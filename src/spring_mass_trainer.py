import torch
import spring_mass_dataset
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import models as models
import spring_mass_model
import math
import numpy as np
import os
from qqtt.utils import logger, cfg
import warp as wp
import os
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

# 3. 如果是 OffscreenRenderer，无头模式仍需保留
# torch.autograd.set_detect_anomaly(True)
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["GALLIUM_DRIVER"] = "llvmpipe"

@wp.kernel
def accum_loss_kernel(loss_accum: wp.array(dtype=float), frame_loss: wp.array(dtype=float)):
    # 简单的累加操作: loss_accum[0] += frame_loss[0]
    wp.atomic_add(loss_accum, 0, frame_loss[0])

class SpringMassTrainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self._create_model()
        mdata = self._create_dataset_offline(mode='train')

        self.mdata = mdata

        if "cloth" in mdata.object_case or "package" in mdata.object_case:
                cfg.load_from_yaml("configs/phystwin_configs/cloth.yaml")
        else:
            cfg.load_from_yaml("configs/phystwin_configs/real.yaml")

        cfg.device = next(self.model.parameters()).device
        
        # create warp simulator
        self.model.module.load_warp_simulator(
            dt = mdata.dt,
            init_vertices = mdata.mesh_pos,
            init_springs = mdata.cells,
            init_spring_Y = mdata.init_spring_Y,
            init_rest_lengths = mdata.spring_reset_length,
            init_masses = mdata.masses,
            num_object_springs = mdata.cells.shape[0],
            init_masks = None,
            init_velocities = mdata.velocity,
            num_all_points = mdata.num_object_points,
            num_surface_points = mdata.num_surface_points,
            num_original_points = mdata.num_original_points,
            controller_points = mdata.controller_point,
            object_points = mdata.object_point,
            object_visibilities = mdata.object_visibilities,
            object_motions_valid = mdata.object_motions_valid,
            collide_elas = mdata.collide_elas,
            collide_fric = mdata.collide_fric,
            dashpot_damping = mdata.dashpot_damping,
            drag_damping = mdata.drag_damping,
            collide_object_elas = mdata.collide_object_elas,
            collide_object_fric = mdata.collide_object_fric,
            collision_dist = mdata.collision_dist,
            reverse_z = cfg.reverse_z,
            spring_Y_min = cfg.spring_Y_min,
            spring_Y_max = cfg.spring_Y_max,
            self_collision = cfg.self_collision,
            num_substeps = cfg.num_substeps,
            device=cfg.device,
        )
        
        # 初始化 Warp 优化器（仅在训练模式且 collision_learn 开启时）
        if cfg.collision_learn:
            # 确保 requires_grad=True
            torch_spring_Y = wp.to_torch(self.model.module.simulator.wp_spring_Y)
            torch_collide_elas = wp.to_torch(self.model.module.simulator.wp_collide_elas)
            torch_collide_fric = wp.to_torch(self.model.module.simulator.wp_collide_fric)
            torch_collide_object_elas = wp.to_torch(self.model.module.simulator.wp_collide_object_elas)
            torch_collide_object_fric = wp.to_torch(self.model.module.simulator.wp_collide_object_fric)
            
            warp_params_list = [torch_collide_elas, torch_collide_fric, 
                                torch_collide_object_elas, torch_collide_object_fric]
        else:
            warp_params_list = []

        self.optimizer = torch.optim.Adam([p for p in self.model.module.parameters()] + warp_params_list, 
                                          lr=self.args.lr, 
                                          betas=(0.9, 0.99))


        def linear_warmup_lr(epoch):
            max_lr = self.args.lr  
            min_lr = max_lr / 10
            if epoch < self.args.warmup_epochs:
                return min_lr + (max_lr - min_lr) * (epoch / self.args.warmup_epochs) 
            else:
                return 1.0  
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_warmup_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.gamma)
        if dist.get_rank() == 0:
            self.writer = SummaryWriter(os.path.join(self.args.dump_dir, 'log'))
        self.pbar = tqdm(range(self.current_epoch, self.args.num_epochs), unit="iters")

        os.makedirs(self.args.dump_dir, exist_ok=True)
        for subdir in ['ckpts', 'log', 'test_RMSE', 'spring_mech_info', 'trajectories']:
            dir = os.path.join(self.args.dump_dir, subdir)
            os.makedirs(dir, exist_ok=True)

    def _create_model(self):
        if self.args.case == 'springmass':
            self.model_class = spring_mass_model.SpringMass
            self.dataset_class = spring_mass_dataset.MeshSpringMassDataset
            self.object_case = self.args.object_case
        else:
            raise NotImplementedError("A Case not wrapped yet")
        self.model = self.model_class(
            pos_dim=self.args.space_dim, 
            ld=self.args.hidden_dim, 
            layer_num=self.args.multi_mesh_layer, 
            pre_layer_num=self.args.pre_layer_num, 
            bottom_layer_num=self.args.bottom_layer_num,
            mlp_hidden_layer=self.args.hidden_depth, 
            MP_times=self.args.mp_time,
            enhance=self.args.enhance, 
            agg_conv_pos=self.args.agg_conv_pos
        )

        if self.args.restart_epoch < 0 and not self.args.scratch:
            try:
                self.model.load_state_dict(
                    torch.load(self.args.path),
                    strict=False
                )
            except:
                checkpoint = torch.load(self.args.path)
                model_dict = self.model.state_dict()
                # Filter out unnecessary keys in checkpoint
                checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
                keys = set(checkpoint.keys())
                # Check if shapes match
                for k in keys:
                    if model_dict[k].shape != checkpoint[k].shape:
                        print(f"\033[31mIgnoring parameter {k} due to shape mismatch:\033[0m {model_dict[k].shape} != {checkpoint[k].shape}")
                        del checkpoint[k]
                self.model.load_state_dict(checkpoint, strict=False)
            print(f"\033[31mLoad pretrained from\033[0m {self.args.path}")

        POST_FIX_1 = '_layernum_' + str(self.args.multi_mesh_layer)
        POST_FIX_2 = POST_FIX_1 + '_MPHIDDENLAYER_' + str(self.args.hidden_depth) + '_MPHIDDENTDIM_' + str(self.args.hidden_dim) + '_MPtime_' + str(self.args.mp_time) + '_NoiseLevel_' + str(
            self.args.noise_level)
        self.checkpt_name = self.args.case + POST_FIX_2 + '.pt'
        if self.args.restart_epoch >= 0:
            print('hi, restarting')
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(torch.load(os.path.join(self.args.dump_dxir, 'ckpts', str(self.args.restart_epoch) + "_" + self.checkpt_name)))
            else:
                self.model.load_state_dict(torch.load(os.path.join(self.args.dump_dir, 'ckpts', str(self.args.restart_epoch) + "_" + self.checkpt_name)))
            
            self.current_epoch = self.args.restart_epoch + 1
            if self.current_epoch < self.args.warmup_epochs:
                min_lr = self.args.lr / 10
                max_lr = self.args.lr
                self.args.lr = min_lr + (max_lr - min_lr) * (self.current_epoch / self.args.warmup_epochs)
            else:
                decay_epochs = self.current_epoch - self.args.warmup_epochs
                self.args.lr *= self.args.gamma**(decay_epochs)
                self.args.lr = max(self.args.lr, 1e-6)
            print('restarted lr is: ', self.args.lr)
        else:
            self.current_epoch = 0

        self.model = nn.parallel.DistributedDataParallel(
            self.model.cuda(self.args.local_rank),
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=True  # [关键修复] 允许部分参数不参与计算
        )

    def _preproc_multi_infos(self, mdata):
        # process the multi-level mesh for batched data here
    
        # no contact, then share the graph between batches
        # only need to reshape input tensor
        m_ids = mdata.m_idx
        m_gs_list = mdata.m_g
        m_gs_parents_list = mdata.m_edge_parents
        m_gs = [torch.tensor(g, dtype=torch.long).to(self.device) for g in m_gs_list]
        m_gs_parents = [torch.tensor(g_parent, dtype=torch.long).to(self.device) for g_parent in m_gs_parents_list]

        return m_ids, m_gs, m_gs_parents

    def _create_dataset_offline(self, mode='train', stride=1):
        if mode == 'train':
            prob = np.random.rand()
            add_noise = False # (prob < 0.667)
            mdata = self.dataset_class(self.args.data_dir,
                                        layer_num=self.args.multi_mesh_layer,
                                        stride=stride,
                                        noise_shuffle=add_noise,
                                        noise_level=self.args.noise_level,
                                        noise_gamma=self.args.noise_gamma,
                                        recal_mesh=self.args.recal_mesh,
                                        consist_mesh=self.args.consist_mesh,
                                        object_case=self.args.object_case,
                                        args=self.args)
        else:
            mdata = self.dataset_class(self.args.data_dir,
                                        layer_num=self.args.multi_mesh_layer,
                                        stride=stride,
                                        noise_shuffle=False,
                                        recal_mesh=self.args.recal_mesh,
                                        consist_mesh=self.args.consist_mesh,
                                        mode=mode,
                                        object_case=self.args.object_case,  
                                        args=self.args)
        return mdata

    def get_single_frame_data(self, frame_idx, m_gs=None, use_gt=False):
        """
        获取单帧的仿真数据（所有substep的位置和速度）

        Args:
            frame_idx: 当前帧索引
            m_gs: 多层级网格信息
            use_gt: 是否使用ground truth数据

        Returns:
            pos: (num_substeps+1, num_points) - 所有substep的位置数组
            vel: (num_substeps+1, num_points) - 所有substep的速度数组
            node_mass: 节点质量
            spring_Y: 弹簧杨氏模量
            spring_reset_length: 弹簧原长
            spring_dashpot_damping: 弹簧阻尼
            drag_damping: 拖拽阻尼
        """
        simulator = self.model.module.simulator

        # 设置当前帧的控制器目标
        simulator.set_controller_target(frame_idx, pure_inference=True)

        # 更新碰撞图（如果需要）
        if simulator.object_collision_flag:
            simulator.update_collision_graph()

        # 执行仿真步骤
        simulator.step()

        # 收集所有substep的位置和速度
        pos_list = []
        vel_list = []

        # 遍历所有wp_states来获取所有substep的数据
        for state in simulator.wp_states:
            # 获取object points的位置和速度
            vertices = torch.cat([
                wp.to_torch(state.wp_x).clone(),
                wp.to_torch(state.wp_control_x).clone()
            ])
            velocities = torch.cat([
                wp.to_torch(state.wp_v).clone(),
                wp.to_torch(state.wp_control_v).clone()
            ])

            # if use_gt:
            #     # 对于最后一个状态，使用当前帧的GT
            #     if state == simulator.wp_states[-1]:
            #         gt_object_points = self.mdata.object_point[frame_idx]
            #         if gt_object_points is not None and len(gt_object_points) > 0:
            #             vertices[:simulator.num_original_points] = torch.FloatTensor(gt_object_points).to(vertices.device)
            #             # 计算速度：用当前帧和前一帧的差分
            #             gt_object_points_prev = self.mdata.object_point[frame_idx - 1]
            #             if gt_object_points_prev is not None and len(gt_object_points_prev) > 0:
            #                 velocities[:simulator.num_original_points] = (
            #                     torch.FloatTensor(gt_object_points).to(velocities.device) -
            #                     torch.FloatTensor(gt_object_points_prev).to(velocities.device)
            #                 ) / (self.mdata.dt * cfg.num_substeps)

            pos_list.append(vertices)
            vel_list.append(velocities)

        # 将列表转换为张量数组 (num_substeps+1, num_points, 3)
        pos = torch.stack(pos_list, dim=0)
        vel = torch.stack(vel_list, dim=0)

        # downsample from pos and vel
        sample_stamp_idx = torch.linspace(0, pos.shape[0]-1, steps=3, dtype=torch.long)

        pos, vel = pos[sample_stamp_idx], vel[sample_stamp_idx]        

        # 获取仿真参数
        node_mass = torch.cat([
            wp.to_torch(simulator.wp_masses).clone(),
            torch.zeros(simulator.num_control_points, device=pos.device)
        ])
        spring_Y = wp.to_torch(simulator.wp_spring_Y).clone()
        spring_rest_length = wp.to_torch(simulator.wp_rest_lengths).clone()
        spring_dashpot_damping = simulator.dashpot_damping
        drag_damping = simulator.drag_damping

        # 计算弹力和dashpot_damping力
        # Get edge indices from graph structure
        edge_index = m_gs[0]  # [2, num_edges]
        idx1 = edge_index[0]  # source nodes
        idx2 = edge_index[1]  # target nodes

        # Get positions and velocities for connected nodes
        x1 = pos[:, idx1]  # [T, num_edges, 3]
        v1 = vel[:, idx1]  # [T, num_edges, 3]
        x2 = pos[:, idx2]  # [T, num_edges, 3]
        v2 = vel[:, idx2]  # [T, num_edges, 3]
        
        # Get spring properties
        spring_Y = torch.exp(spring_Y)
        spring_Y = spring_Y[None, :, None]  # [1, num_edges]
        spring_rest_length = spring_rest_length[None, :, None]  # [num_edges]

        spring_Y = torch.cat([spring_Y]*2, dim=1)
        spring_rest_length = torch.cat([spring_rest_length]*2, dim=1)

        # Calculate displacement vector
        dis = x2 - x1  # [num_edges, 3]
        dis_len = torch.norm(dis, dim=-1, keepdim=True)  # [num_edges, 1]

        # Calculate unit direction vector
        d = dis / torch.clamp(dis_len, min=1e-6)  # [num_edges, 3]

        # Calculate spring force: F = k * (current_length / rest_length - 1.0) * direction
        spring_force = (spring_Y * (dis_len / spring_rest_length - 1.0) * d)

        # Calculate damping force: F = damping * relative_velocity * direction
        v_rel = torch.sum((v2 - v1) * d, dim=-1, keepdim=True)  # [num_edges, 1]
        dashpot_force = spring_dashpot_damping * v_rel * d

        # Total force on each edge
        overall_forces = spring_force + dashpot_force # [num_edges, 3]

        return pos, vel, node_mass, torch.log(spring_Y[0]), spring_rest_length[0], drag_damping, spring_force, dashpot_force, overall_forces

    def simulate_single_step_with_prediction(self, frame_idx, prev_point_pos, prev_point_vel, 
                                             predicted_log_spring_Y, predicted_rest_length, enable_backward=False):
        """
        使用模型预测的机械属性执行单步仿真并计算loss

        Args:
            frame_idx: 当前帧索引
            predicted_log_spring_Y: 模型预测的弹簧杨氏模量
            predicted_rest_length: 模型预测的弹簧原长
            enable_backward: 是否启用反向传播

        Returns:
            如果enable_backward=True，返回(tape, frame_loss)
            否则返回None
        """

        simulator = self.model.module.simulator

        # 使用前一帧状态作为当前初始状态
        prev_point_pos = wp.from_torch(prev_point_pos[:simulator.num_object_points], dtype=wp.vec3)
        prev_point_vel = wp.from_torch(prev_point_vel[:simulator.num_object_points], dtype=wp.vec3)
        simulator.set_init_state(prev_point_pos, prev_point_vel)

        # 使用支持梯度的方法更新simulator的弹簧属性
        simulator.update_spring_properties(predicted_log_spring_Y, 
                                           predicted_rest_length)

        if enable_backward:
            tape = simulator.tape
            with wp.ScopedTimer("backward"):
                # # 手动设置当前帧的ground truth object point
                # if cfg.data_type == "real":
                #     gt_points = simulator.gt_object_points[frame_idx]
                #     if gt_points is not None and len(gt_points) > 0:
                #         wp.copy(simulator.wp_object_points, wp.from_torch(gt_points, dtype=wp.vec3))

                # 设置控制器目标
                simulator.set_controller_target(frame_idx)

                if simulator.object_collision_flag:
                    simulator.update_collision_graph()

                if cfg.use_graph:
                    wp.capture_launch(simulator.graph)
                else:
                    if cfg.data_type == "real":
                        with tape:
                            simulator.step()
                            simulator.calculate_loss()
                        tape.backward(simulator.loss)
                    else:
                        with tape:
                            simulator.step()
                            simulator.calculate_simple_loss()
                        tape.backward(simulator.loss)

            # if cfg.data_type == "real" and frame_idx > 1:
            #     simulator.update_acc()
            #     simulator.set_acc_count(True)

            if cfg.data_type == "real":
                chamfer_loss = wp.to_torch(
                    simulator.chamfer_loss, requires_grad=False
                )
                track_loss = wp.to_torch(
                    simulator.track_loss, requires_grad=False
                )
            else:
                chamfer_loss = 0
                track_loss = 0

            return tape, simulator.loss, chamfer_loss, track_loss
        else:
            # 推理模式
            simulator.set_controller_target(frame_idx, pure_inference=False)

            # # 手动设置当前帧的ground truth object point
            # if cfg.data_type == "real":
            #     gt_points = simulator.gt_object_points[frame_idx]
            #     if gt_points is not None and len(gt_points) > 0:
            #         wp.copy(simulator.wp_object_points, wp.from_torch(gt_points, dtype=wp.vec3))

            if simulator.object_collision_flag:
                simulator.update_collision_graph()

            simulator.step()

            simulator.set_init_state(simulator.wp_states[-1].wp_x, simulator.wp_states[-1].wp_v, pure_inference=True)
            if cfg.data_type == "real" and frame_idx > 1:
                simulator.update_acc()
                simulator.set_acc_count(True)

            return None

    def visualize_point_clouds(self, prev_points, step_points, gt_points, frame_idx, save_path=None):
        """
        可视化前一帧点云、step后的点云和真值点云，并保存为图片

        Args:
            prev_points: 前一帧的点云 (N, 3) numpy array or torch tensor
            step_points: step后的点云 (N, 3) numpy array or torch tensor
            gt_points: 真值点云 (N, 3) numpy array or torch tensor
            frame_idx: 当前帧索引
            save_path: 保存路径，如果为None则使用默认路径
        """
        # 转换为numpy数组
        if isinstance(prev_points, torch.Tensor):
            prev_points = prev_points.detach().cpu().numpy()
        if isinstance(step_points, torch.Tensor):
            step_points = step_points.detach().cpu().numpy()
        if isinstance(gt_points, torch.Tensor):
            gt_points = gt_points.detach().cpu().numpy()

        # 创建图形
        fig = plt.figure(figsize=(20, 6))

        # 计算全局坐标范围以保持一致的视角
        all_points = np.vstack([prev_points, step_points, gt_points])
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

        # 添加边距
        margin = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        # 子图1: 前一帧点云
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(prev_points[:, 0], prev_points[:, 1], prev_points[:, 2],
                   c='blue', marker='o', s=20, alpha=0.6, label='Previous Frame')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'Previous Frame (Frame {frame_idx-1})', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        ax1.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        ax1.set_zlim(z_min - margin * z_range, z_max + margin * z_range)

        # 子图2: Step后的点云
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(step_points[:, 0], step_points[:, 1], step_points[:, 2],
                   c='green', marker='o', s=20, alpha=0.6, label='After Step')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title(f'After Simulation Step (Frame {frame_idx})', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        ax2.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        ax2.set_zlim(z_min - margin * z_range, z_max + margin * z_range)

        # 子图3: 真值点云与预测点云对比
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2],
                   c='red', marker='o', s=20, alpha=0.5, label='Ground Truth')
        ax3.scatter(step_points[:, 0], step_points[:, 1], step_points[:, 2],
                   c='green', marker='^', s=15, alpha=0.5, label='Predicted')
        ax3.scatter(prev_points[:, 0], prev_points[:, 1], prev_points[:, 2],
                   c='blue', marker='o', s=15, alpha=0.2, label='Previous Frame')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title(f'Comparison (Frame {frame_idx})', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        ax3.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        ax3.set_zlim(z_min - margin * z_range, z_max + margin * z_range)


        plt.tight_layout()

        # 保存图片
        if save_path is None:
            vis_dir = os.path.join(self.args.dump_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            save_path = os.path.join(vis_dir, f'frame_{frame_idx:04d}_pointcloud.png')

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Point cloud visualization saved to: {save_path}")
    
        return save_path

    def save_traj(
        self,  mech_info=None, save_path=None
    ):
        # Get simulator reference
        simulator = self.model.module.simulator

        # 1. Set the mech_info to the simulator for recording
        if mech_info is not None:
            logger.info("Setting predicted mechanical properties to simulator")
            if mech_info.dim() == 2:
                predicted_spring_Y = mech_info[:, 0]
                predicted_rest_length = mech_info[:, 1]
                simulator.update_spring_properties(predicted_spring_Y, predicted_rest_length)
            else:
                logger.warning("mech_info has unexpected dimensions, skipping property update")

        # 2. Start simulation and record the trajectory
        frame_len = self.mdata.train_frame + self.mdata.test_frame
        simulator.set_init_state(
            simulator.wp_init_vertices, simulator.wp_init_velocities
        )
        vertices = [
            wp.to_torch(simulator.wp_states[0].wp_x, requires_grad=False).cpu()
        ]

        with wp.ScopedTimer("simulate"):
            for i in tqdm(range(1, frame_len), desc="Saving trajectory"):
                if cfg.data_type == "real":
                    simulator.set_controller_target(i, pure_inference=True)
                if simulator.object_collision_flag:
                    simulator.update_collision_graph()

                if cfg.use_graph:
                    wp.capture_launch(simulator.forward_graph)
                else:
                    simulator.step()
                x = wp.to_torch(simulator.wp_states[-1].wp_x, requires_grad=False)
                vertices.append(x.cpu())
                # Set the intial state for the next step
                simulator.set_init_state(
                    simulator.wp_states[-1].wp_x,
                    simulator.wp_states[-1].wp_v,
                )

        vertices = torch.stack(vertices, dim=0)

        # 3. Save the trajectory to a file
        logger.info(f"Save the trajectory to {save_path}")
        vertices_to_save = vertices.cpu().numpy()
        with open(save_path, "wb") as f:
            pickle.dump(vertices_to_save, f)

        logger.info(f"Trajectory saved successfully with shape {vertices_to_save.shape}")


    def run_epoch(self, epoch,
                  mode='train',
                  mech_info = None,
                  ):

        if mode != 'train':
            self.model.eval()
        else:
            self.model.train()

        mean_loss_insts = 0

        # 单步仿真迭代训练
        simulator = self.model.module.simulator

        # 启动GPU内存记录（仅训练模式）
        # if mode == 'train':
        #     # 创建内存快照保存目录
        #     memory_snapshot_dir = os.path.join(self.args.dump_dir, 'memory_snapshots')
        #     os.makedirs(memory_snapshot_dir, exist_ok=True)

        #     # 开始记录GPU内存历史
        #     torch.cuda.memory._record_memory_history(max_entries=100000)
        #     logger.info(f"Started GPU memory recording for epoch {epoch}")

        # 逐帧迭代
        if mode == 'train':
            max_iter = 50
            m_ids, m_gs, m_gs_parent = \
                self._preproc_multi_infos(self.mdata)

            for frame_idx in range(1, self.mdata.train_frame, 5):
                frame_error = 1
                retry_times = 0
                while ((frame_error > 0.00005) and (retry_times < max_iter)):
                    # 初始化仿真器状态
                    simulator.set_init_state(simulator.wp_init_vertices, 
                                             simulator.wp_init_velocities)
                    frame_error = 0

                    self.optimizer.zero_grad()
                    for sub_frame_idx in range(1, frame_idx+1):
                        
                        print(f"\n--- Processing Frame {sub_frame_idx}/{self.mdata.train_frame-1} ---")
                        
                        # 获取当前帧数据（前一帧和当前帧）
                        pos, vel, node_mass, spring_Y, spring_rest_length, \
                            drag_damping, spring_force, dashpot_force, overall_forces = \
                            self.get_single_frame_data(sub_frame_idx, m_gs, use_gt=True)

                        node_in_feature, edge_in_feature = self.mdata._preprocess(
                            pos, vel, node_mass,
                            spring_Y, spring_rest_length,
                            drag_damping, spring_force, dashpot_force, 
                            overall_forces, device=self.device
                        )

                        # 3. 模型预测机械属性
                        st = time()

                        # spring_Y 和 rest_length 的预测值是基于 edge_in_feature 的，因此我们需要将 edge_in_feature 传入模型进行预测
                        edge_mech_in_bias = self.model(m_ids, m_gs, m_gs_parent, 
                                               node_in_feature, 
                                               edge_in_feature)
                        num_edge = edge_in_feature.shape[1]

                        new_spring_Y = torch.exp(spring_Y[:num_edge // 2, 0]) + edge_mech_in_bias[:, 0]
                        new_rest_length = spring_rest_length[:num_edge // 2, 0] + edge_mech_in_bias[:, 1]

                        new_spring_Y = torch.log(torch.clip(new_spring_Y, 1e2, cfg.spring_Y_max))
                        new_rest_length = torch.clip(new_rest_length, 2e-5, self.mdata.max_radius)

                        print(f"Model inference time: {time() - st:.4f} seconds")

                        # 4. 使用预测的机械属性进行单步仿真并计算loss和反传
                        tape, frame_loss, chamfer_loss, track_loss = \
                            self.simulate_single_step_with_prediction(
                                sub_frame_idx, pos[0], vel[0],
                                new_spring_Y, new_rest_length, 
                                enable_backward=True
                            )

                        # 4.5 可视化点云（每10帧或第一帧）
                        if sub_frame_idx % 10 == 0 or sub_frame_idx == 1:
                            # 获取前一帧点云（只取object points）
                            prev_points = pos[0][:simulator.num_object_points]

                            # 获取step后的点云
                            step_points = torch.cat([
                                wp.to_torch(simulator.wp_states[-1].wp_x).clone(),
                            ])[:simulator.num_object_points]

                            # 获取真值点云
                            gt_points = self.mdata.object_point[sub_frame_idx]
                            if gt_points is not None and len(gt_points) > 0:
                                gt_points = torch.FloatTensor(gt_points).to(prev_points.device)

                                # 调用可视化函数
                                try:
                                    self.visualize_point_clouds(
                                        prev_points=prev_points,
                                        step_points=step_points,
                                        gt_points=gt_points,
                                        frame_idx=sub_frame_idx
                                    )
                                except:
                                    import pdb; pdb.set_trace()

                        # 5. 提取梯度
                        wp_predicted_spring_Y = simulator.wp_spring_Y
                        wp_predicted_rest_length = simulator.wp_rest_lengths

                        grad_spring_Y = wp.to_torch(wp_predicted_spring_Y.grad).clone() if wp_predicted_spring_Y.grad else torch.zeros_like(new_spring_Y)
                        grad_rest_length = wp.to_torch(wp_predicted_rest_length.grad).clone() if wp_predicted_rest_length.grad else torch.zeros_like(new_rest_length)

                        # -------------------------------------------------------
                        # [核心] 手动梯度桥接流程
                        # -------------------------------------------------------
                        
                        torch.autograd.backward(
                            tensors=[new_spring_Y, new_rest_length],
                            grad_tensors=[grad_spring_Y, grad_rest_length],
                        )

                        # 获取总Loss的数值用于统计
                        loss_val = wp.to_torch(frame_loss).item()

                        print(f"Spring_Y Grad Norm: {grad_spring_Y.norm().item():.6f}")
                        print(f"Rest_Length Grad Norm: {grad_rest_length.norm().item():.6f}")

                        # 打印碰撞参数梯度（如果启用了collision_learn）
                        if cfg.collision_learn:
                            print(f"Collision Elas Grad: {wp.to_torch(simulator.wp_collide_elas.grad)}")
                            print(f"Collision Fric Grad: {wp.to_torch(simulator.wp_collide_fric.grad)}")
                            print(f"Collision Object Elas Grad: {wp.to_torch(simulator.wp_collide_object_elas.grad)}")
                            print(f"Collision Object Fric Grad: {wp.to_torch(simulator.wp_collide_object_fric.grad)}")

                        # 打印仿真参数
                        print(f"\n{'='*60}")
                        print(f"Spring_Y Average: {new_spring_Y.mean().item():.6f}")
                        print(f"Rest_Length Average: {new_rest_length.mean().item():.6f}")
                        print(f"\nCollision Parameters:")
                        print(f"  collide_elas: {wp.to_torch(self.model.module.simulator.wp_collide_elas).item():.6f}")
                        print(f"  collide_fric: {wp.to_torch(self.model.module.simulator.wp_collide_fric).item():.6f}")
                        print(f"  collide_object_elas: {wp.to_torch(self.model.module.simulator.wp_collide_object_elas).item():.6f}")
                        print(f"  collide_object_fric: {wp.to_torch(self.model.module.simulator.wp_collide_object_fric).item():.6f}")

                        print(f"\nLoss Breakdown:")
                        print(f"{'='*60}\n")
                        print(f"Frame {sub_frame_idx} Loss: {wp.to_torch(frame_loss).item():.6f}")
                        print(f"Chamfer Loss: {chamfer_loss.item():.6f}")
                        print(f"Track Loss: {track_loss.item():.6f}")

                        if cfg.use_graph:
                            # Only need to clear the gradient, the tape is created in the graph
                            tape.zero()
                        else:
                            # Need to reset the compute graph and clear the gradient
                            tape.reset()

                        # PyTorch 优化器更新神经网络参数
                        self.optimizer.step()
                            
                        simulator.clear_loss()
                        # 更新状态为下一帧的初始状态
                        simulator.set_init_state(simulator.wp_states[-1].wp_x, 
                                                simulator.wp_states[-1].wp_v)
                     
                        # stats
                        frame_error += loss_val
                    frame_error /= frame_idx
                    retry_times += 1

                    # 保存内存快照（每次重试后）
                    # snapshot_path = os.path.join(memory_snapshot_dir,
                    #                             f'epoch_{epoch}_frame_{frame_idx}_retry_{retry_times}.pickle')
                    # try:
                    #     torch.cuda.memory._dump_snapshot(snapshot_path)
                    #     logger.info(f"Saved memory snapshot: {snapshot_path}")
                    # except Exception as e:
                    #     logger.warning(f"Failed to save memory snapshot: {e}")

                # import pdb; pdb.set_trace()
                mean_loss_insts = frame_error
        else:
            assert mech_info is not None, "In test mode, mech_info must be provided."

            # set预测的机械属性
            simulator.update_spring_properties(
                mech_info[:, 0],
                mech_info[:, 1]
            )

            # vertices = []

            # 验证模式
            for frame_idx in range(1, self.mdata.train_frame + self.mdata.test_frame):
                simulator.set_controller_target(frame_idx)
                if simulator.object_collision_flag:
                    simulator.update_collision_graph()

                if cfg.use_graph:
                    wp.capture_launch(simulator.graph)
                else:
                    if cfg.data_type == "real":
                        with simulator.tape:
                            simulator.step()
                            simulator.calculate_loss()
                    else:
                        with simulator.tape:
                            simulator.step()
                            simulator.calculate_simple_loss()

                x = wp.to_torch(simulator.wp_states[-1].wp_x, requires_grad=False)
                # vertices.append(x.cpu())
                # Set the intial state for the next step
                simulator.set_init_state(
                    simulator.wp_states[-1].wp_x,
                    simulator.wp_states[-1].wp_v,
                )

                simulator.clear_loss()
                # stats
                mean_loss_insts += wp.to_torch(frame_loss).item()
            
            return mean_loss_insts #, vertices 
            
        if mode == 'train':
            # opt scheduler
            if epoch < self.args.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                if self.optimizer.param_groups[0]['lr'] > 1e-6:
                    self.scheduler.step()
                    if self.optimizer.param_groups[0]['lr'] < 1e-6:
                        self.optimizer.param_groups[0]['lr'] = 1e-6
        else:
            self.model.eval()

        # 停止GPU内存记录（仅训练模式）
        # if mode == 'train':
        #     try:
        #         # 保存最终的内存快照
        #         final_snapshot_path = os.path.join(memory_snapshot_dir,
        #                                            f'epoch_{epoch}_final.pickle')
        #         torch.cuda.memory._dump_snapshot(final_snapshot_path)
        #         logger.info(f"Saved final memory snapshot: {final_snapshot_path}")
        #     except Exception as e:
        #         logger.warning(f"Failed to save final memory snapshot: {e}")

        #     # 停止记录
        #     torch.cuda.memory._record_memory_history(enabled=None)
        #     logger.info(f"Stopped GPU memory recording for epoch {epoch}")

        # 返回最后一帧的mech_info
        final_mech_info = torch.stack([new_spring_Y, new_rest_length], dim=1).detach() if mode == 'train' else None
        return mean_loss_insts, final_mech_info

    def train(self):
        # load phystwin zero-grad optimization results and warp simulator here
        best_loss = 1e8
        best_mech_info = None
        best_epoch = None

        # Early stopping parameters for mech_info convergence
        prev_mech_info = None
        mech_info_change_threshold = 1e-7  # Relative change threshold
        patience = 5  # Number of epochs to wait before stopping
        patience_counter = 0

        for epoch in self.pbar:
            mean_loss_train, mech_info = self.run_epoch(epoch, 'train')
            with torch.autograd.no_grad():
                # train loss record
                if dist.get_rank() == 0:
                    self.pbar.set_description(f"Epoch: {epoch}, Training RMSE: {math.sqrt(mean_loss_train)}")
                    self.writer.add_scalar('RMSE/train', math.sqrt(mean_loss_train), epoch)
                    self.writer.add_scalar('lr/train', self.optimizer.param_groups[0]['lr'], epoch)
                    
                    # dump ckpt
                    ckpt_path = os.path.join(self.args.dump_dir, 'ckpts', str(epoch) + "_" + self.checkpt_name)
                    torch.save(self.model.module.state_dict(), ckpt_path)
                    
                    # dump spring emch info
                    mech_info_path = os.path.join(self.args.dump_dir, 'spring_mech_info', str(epoch) + "_" + self.checkpt_name)
                    torch.save(mech_info.clone(), mech_info_path)

                    # save trajectory with current mech_info
                    traj_save_path = os.path.join(self.args.dump_dir, 'trajectories', f'epoch_{epoch}_trajectory.pkl')
                    logger.info(f"Saving trajectory for epoch {epoch}")
                    self.save_traj(mech_info=mech_info.clone(), save_path=traj_save_path)

                    if best_loss > mean_loss_train:
                        best_mech_info = mech_info.clone()
                        best_epoch = epoch
                        best_loss = mean_loss_train

                        # Save best trajectory
                        best_traj_save_path = os.path.join(self.args.dump_dir, 'trajectories', 'best_trajectory.pkl')
                        logger.info(f"New best loss {best_loss:.6f} at epoch {epoch}, saving best trajectory")
                        self.save_traj(mech_info=mech_info.clone(), save_path=best_traj_save_path)

                    # Check mech_info convergence for early stopping
                    if prev_mech_info is not None:
                        # Calculate relative change in mech_info
                        mech_info_diff = torch.abs(mech_info - prev_mech_info)
                        mech_info_norm = torch.abs(prev_mech_info) + 1e-8  # Avoid division by zero
                        relative_change = (mech_info_diff / mech_info_norm).mean().item()

                        logger.info(f"Epoch {epoch}: mech_info relative change = {relative_change:.6f}")
                        self.writer.add_scalar('mech_info/relative_change', relative_change, epoch)

                        if relative_change < mech_info_change_threshold:
                            patience_counter += 1
                            logger.info(f"mech_info change below threshold. Patience counter: {patience_counter}/{patience}")

                            if patience_counter >= patience:
                                logger.info(f"Early stopping triggered at epoch {epoch}. mech_info has converged.")
                                self.pbar.close()
                                # Save final best mech_info before stopping
                                mech_info_path = os.path.join(self.args.dump_dir, 'spring_mech_info', "best_{}.pth".format(best_epoch))
                                torch.save(best_mech_info, mech_info_path)
                                return
                        else:
                            patience_counter = 0  # Reset counter if change is above threshold

                    # Update prev_mech_info for next epoch
                    prev_mech_info = mech_info.clone()

                # valid loss record
                if self.args.n_valid > 0 and (epoch+1) % 5 == 0:
                    mean_loss_valid, _ = self.run_epoch(epoch, 'test', mech_info=mech_info)
                    if dist.get_rank() == 0:
                        self.writer.add_scalar('RMSE/test', math.sqrt(mean_loss_valid), epoch)
                dist.barrier()
        
        # dump best spring emch info estimation
        mech_info_path = os.path.join(self.args.dump_dir, 'spring_mech_info', "best_{}.pth".format(best_epoch))
        torch.save(best_mech_info, mech_info_path)

    def test(self, epoch=None):
        self.model.eval()
        epoch = self.args.restart_epoch if epoch==None else epoch
        rmse = []
        with torch.autograd.no_grad():
            # reload the model
            print('test on epoch, ', epoch)
            if self.args.n_test > 0:
                mean_loss_test, _ = self.run_epoch(epoch, mode='test')
                print('epoch', epoch, 'RMSE/test', math.sqrt(mean_loss_test))
                if dist.get_rank() == 0:
                    self.writer.add_scalar('RMSE/test', math.sqrt(mean_loss_test), epoch)

            rmse.append(math.sqrt(mean_loss_test))

        if dist.get_rank() == 0:
            RMSE_cell = np.array([np.mean(rmse)])
            dump_path = os.path.join(self.args.dump_dir, 'test_RMSE', 'epoch_' + str(epoch) + '.csv')
            np.savetxt(dump_path, RMSE_cell, delimiter=',')
        
    def visualize_simulation(self, mech_info_path, output_video_path='simulation_visualization.mp4'):
        """
        Run pure inference simulation and create PyVista visualization video.

        Args:
            mech_info_path: Path to the best_mech_info .pth file
            output_video_path: Path to save the output video
        """
        import pyvista as pv
        import h5py

        self.model.eval()

        with torch.autograd.no_grad():
            # Load dataset
            mdata = self._create_dataset_offline(mode='train')

            # Load config
        if "cloth" in mdata.object_case or "package" in mdata.object_case:
            cfg.load_from_yaml("configs/phystwin_configs/cloth.yaml")
        else:
            cfg.load_from_yaml("configs/phystwin_configs/real.yaml")

        cfg.device = next(self.model.parameters()).device

        # Load warp simulator
        self.model.module.load_warp_simulator(
            dt=mdata.dt,
            init_vertices=mdata.mesh_pos,
            init_springs=mdata.cells,
            init_spring_Y=mdata.init_spring_Y,
            init_rest_lengths=mdata.spring_reset_length,
            init_masses=mdata.masses,
            num_object_springs=mdata.cells.shape[0],
            init_masks=None,
            init_velocities=mdata.velocity,
            num_all_points=mdata.num_object_points,
            num_surface_points=mdata.num_surface_points,
            num_original_points=mdata.num_original_points,
            controller_points=mdata.controller_point,
            object_points=mdata.object_point,
            object_visibilities=mdata.object_visibilities,
            object_motions_valid=mdata.object_motions_valid,
            collide_elas=mdata.collide_elas,
            collide_fric=mdata.collide_fric,
            dashpot_damping=mdata.dashpot_damping,
            drag_damping=mdata.drag_damping,
            collide_object_elas=mdata.collide_object_elas,
            collide_object_fric=mdata.collide_object_fric,
            collision_dist=mdata.collision_dist,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            self_collision=cfg.self_collision,
            num_substeps=cfg.num_substeps,
            device=cfg.device,
        )

        # Load best_mech_info
        logger.info(f"Loading mech_info from: {mech_info_path}")
        mech_info = torch.load(mech_info_path, map_location=cfg.device)

        # Update simulator with predicted mechanical properties
        if mech_info is not None and mech_info.dim() == 2:
            predicted_spring_Y = mech_info[:, 0]
            self.model.module.simulator.wp_spring_Y = wp.from_torch(predicted_spring_Y.contiguous(), dtype=wp.float32)

            predicted_rest_length = mech_info[:, 1]
            self.model.module.simulator.wp_rest_lengths = wp.from_torch(predicted_rest_length.contiguous(), dtype=wp.float32)

        # Run pure inference simulation and collect all points data
        simulator = self.model.module.simulator

        # Storage for visualization data
        object_points_sequence = []  # Simulated object points
        controller_points_sequence = []  # Controller points
        gt_object_points_sequence = []  # Ground truth object points

        # Initialize simulation
        simulator.set_init_state(simulator.wp_init_vertices, simulator.wp_init_velocities)

        # Collect initial state
        object_points_sequence.append(wp.to_torch(simulator.wp_states[0].wp_x).detach().cpu().numpy())
        controller_points_sequence.append(wp.to_torch(simulator.wp_states[0].wp_control_x).detach().cpu().numpy())
        gt_object_points_sequence.append(mdata.object_point[0])

        # Run simulation
        logger.info(f"Running pure inference simulation for {mdata.train_frame} frames...")
        
        for frame_idx in tqdm(range(1, mdata.train_frame + mdata.test_frame)):
            # Set controller target with pure_inference=True
            simulator.set_controller_target(frame_idx, pure_inference=True)

            if simulator.object_collision_flag:
                simulator.update_collision_graph()

            # Run simulation step
            simulator.step()

            # Collect data
            object_points_sequence.append(wp.to_torch(simulator.wp_states[-1].wp_x).detach().cpu().numpy())
            controller_points_sequence.append(wp.to_torch(simulator.wp_states[-1].wp_control_x).detach().cpu().numpy())
            gt_object_points_sequence.append(mdata.object_point[frame_idx])

            # Reset for next step
            simulator.set_init_state(simulator.wp_states[-1].wp_x, simulator.wp_states[-1].wp_v)
            if cfg.data_type == "real" and frame_idx > 1:
                simulator.update_acc()
                simulator.set_acc_count(True)

        logger.info("Simulation completed. Creating PyVista visualization...")
        # Start a virtual framebuffer (requires xvfb to be installed)
        pv.start_xvfb()

        # Create PyVista visualization
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        plotter.set_background('white')
        # Set camera view
        plotter.camera_position = 'xy'
        plotter.camera.azimuth = 45
        plotter.camera.elevation = 30

        plotter.open_movie(output_video_path, framerate=10)
        
        # Visualize each frame
        for frame_idx in tqdm(range(len(object_points_sequence)), desc="Rendering frames"):
            plotter.clear()

            # Add simulated object points (green)
            obj_points = object_points_sequence[frame_idx]
            if len(obj_points) > 0:
                obj_cloud = pv.PolyData(obj_points)
                plotter.add_mesh(obj_cloud, color='green', point_size=8.0,
                                render_points_as_spheres=True, label='Simulated Object')

            # Add controller points (blue)
            ctrl_points = controller_points_sequence[frame_idx]
            if len(ctrl_points) > 0:
                ctrl_cloud = pv.PolyData(ctrl_points)
                plotter.add_mesh(ctrl_cloud, color='blue', point_size=10.0,
                                render_points_as_spheres=True, label='Controller')

            # Add ground truth object points (red, semi-transparent)
            gt_points = gt_object_points_sequence[frame_idx]
            if len(gt_points) > 0:
                gt_cloud = pv.PolyData(gt_points)
                plotter.add_mesh(gt_cloud, color='red', point_size=6.0,
                                opacity=0.5, render_points_as_spheres=True,
                                label='Ground Truth')

            # Add legend and text
            plotter.add_text(f'Frame: {frame_idx}/{len(object_points_sequence)-1}',
                            position='upper_left', font_size=12, color='black')

            # Write frame
            plotter.write_frame()

        plotter.close()
        logger.info(f"Video saved to: {output_video_path}")

        logger.info("Visualization complete!")

        return object_points_sequence, controller_points_sequence, gt_object_points_sequence