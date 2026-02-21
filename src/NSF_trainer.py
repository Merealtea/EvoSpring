import torch
import datasets as cdata
import spring_mass_dataset
import torch.nn as nn
import torch.distributed as dist
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import models as models
import spring_mass_model 
import neural_spring
import math
import random
import numpy as np
import os
from qqtt.utils import logger, cfg
import warp as wp
import os
from time import time

import logging

# 禁用所有 warp 相关的 logger
logging.getLogger("warp").setLevel(logging.ERROR)

@wp.kernel
def accum_loss_kernel(loss_accum: wp.array(dtype=float), frame_loss: wp.array(dtype=float)):
    # 简单的累加操作: loss_accum[0] += frame_loss[0]
    wp.atomic_add(loss_accum, 0, frame_loss[0])

class NSFTrainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        mdata = self._create_dataset_offline(mode='train')

        num_springs = len(mdata.spring_reset_length)
        init_spring_Y = torch.ones(num_springs).to(self.device) * mdata.init_spring_Y
        spring_rest_length = torch.tensor(mdata.spring_reset_length).to(self.device)

        self.spring_rest_length = spring_rest_length

        self._create_model(num_springs, init_spring_Y)

        self.mdata = mdata
        self.total_update = 0

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
            self.torch_collide_elas = wp.to_torch(self.model.module.simulator.wp_collide_elas)
            self.torch_collide_fric = wp.to_torch(self.model.module.simulator.wp_collide_fric)
            self.torch_collide_object_elas = wp.to_torch(self.model.module.simulator.wp_collide_object_elas)
            self.torch_collide_object_fric = wp.to_torch(self.model.module.simulator.wp_collide_object_fric)
            
            warp_params_list = [self.torch_collide_elas, self.torch_collide_fric, 
                                self.torch_collide_object_elas, self.torch_collide_object_fric]
        else:
            warp_params_list = []

        self.optimizer = torch.optim.Adam([p for p in self.model.module.parameters()],
                                          lr=self.args.lr, 
                                          betas=(0.9, 0.99))

        self.collide_optimizer = torch.optim.Adam(warp_params_list, 
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

    def _create_model(self, num_springs=None, init_spring_Y=None):
        self.object_case = self.args.object_case

        self.model = neural_spring.NeuralSpringField(
            num_springs=num_springs,
            s0_init=init_spring_Y
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

    def _preproc_multi_infos(self, mdata, node_in_feature, edge_mech_in_feature, node_tar):
        # process the multi-level mesh for batched data here
    
        # no contact, then share the graph between batches
        # only need to reshape input tensor
        m_ids = mdata.m_idx
        m_gs_list = mdata.m_g
        m_gs_parents_list = mdata.m_edge_parents
        m_gs = [torch.tensor(g, dtype=torch.long).to(self.device) for g in m_gs_list]
        m_gs_parents = [torch.tensor(g_parent, dtype=torch.long).to(self.device) for g_parent in m_gs_parents_list]
        
        # Load data to GPU
        node_in_feature = node_in_feature.to(self.device)
        edge_mech_in_feature = edge_mech_in_feature.to(self.device)
        node_tar = node_tar.to(self.device)

        return m_ids, m_gs, m_gs_parents, node_in_feature, edge_mech_in_feature, node_tar

    def get_spring_midpoints(self):
        """
        Calculate the midpoint coordinates of springs in 3D space.

        Args:
            frame_idx: Frame index to use for positions (default: 0 for first frame)

        Returns:
            midpoints: Spring midpoint coordinates, shape (num_springs, 3)
        """
        # Extract cells and positions from self.mdata
        cells = self.mdata.cells
        positions = self.mdata.mesh_pos 

        # Get the two endpoints for each spring
        point1_indices = cells[0]
        point2_indices = cells[1]

        # Get coordinates of the two endpoints
        point1_coords = positions[point1_indices]  # (num_springs, 3)
        point2_coords = positions[point2_indices]  # (num_springs, 3)

        # Calculate midpoints
        midpoints = (point1_coords + point2_coords) / 2.0
        midpoints = torch.tensor(midpoints, dtype=torch.float32).to(self.device)  # 转换为 PyTorch 张量并移动到设备上

        # Normalize midpoints to [-1, 1] 
        max_x, max_y, max_z = positions.max(axis=0)
        min_x, min_y, min_z = positions.min(axis=0)
        midpoints[:, 0] = 2 * (midpoints[:, 0] - min_x) / (max_x - min_x) - 1
        midpoints[:, 1] = 2 * (midpoints[:, 1] - min_y) / (max_y - min_y) - 1
        midpoints[:, 2] = 2 * (midpoints[:, 2] - min_z) / (max_z - min_z) - 1
        return midpoints

    def _create_dataset_offline(self, mode='train', stride=1):
        if mode == 'train':
            prob = np.random.rand()
            add_noise = False # (prob < 0.667)
            mdata = spring_mass_dataset.MeshSpringMassDataset(self.args.data_dir,
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
            mdata = spring_mass_dataset.MeshSpringMassDataset(self.args.data_dir,
                                        layer_num=self.args.multi_mesh_layer,
                                        stride=stride,
                                        noise_shuffle=False,
                                        recal_mesh=self.args.recal_mesh,
                                        consist_mesh=self.args.consist_mesh,
                                        mode=mode,
                                        object_case=self.args.object_case,  
                                        args=self.args)
        return mdata
    
    def generate_data_point_sequence(self, update_frame_num, enable_backward=False, default_loss=None):
        simulator = self.model.module.simulator
        vertices_sequence = []
        velocities_sequence = []
        grad_spring_Y_sequence = []

        # [修改] 定义 Warp 端的 Loss 累加器
        losses = []
        
        # 将 cfg.device 更新为 Warp 兼容的字符串或直接获取 Warp device 对象
   
        # 初始状态记录 ... (保持不变)
        simulator.set_init_state(simulator.wp_init_vertices, simulator.wp_init_velocities)
        vertices_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[0].wp_x).clone(), wp.to_torch(simulator.wp_states[0].wp_control_x).clone()]))
        velocities_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[0].wp_v).clone(), wp.to_torch(simulator.wp_states[0].wp_control_v).clone()]))

        context_manager = simulator.tape if enable_backward else open(os.devnull) # 简单的上下文占位符
        
        valid_frames = 0
        for frame_idx in tqdm(range(1, update_frame_num)):
            print(f"Simulating frame {frame_idx}/{update_frame_num}...")
            simulator.set_controller_target(frame_idx)
            
            # Record data
            vertices_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[-1].wp_x).clone(), wp.to_torch(simulator.wp_states[-1].wp_control_x).clone()]))
            velocities_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[-1].wp_v).clone(), wp.to_torch(simulator.wp_states[-1].wp_control_v).clone()]))


            if simulator.object_collision_flag:
                simulator.update_collision_graph()

            # 计算 Loss
            if cfg.data_type == "real":
                with context_manager:
                    simulator.step()
                    simulator.calculate_loss()
            else:
                with context_manager:
                    simulator.step()
                    simulator.calculate_simple_loss()
                    
            if default_loss is not None:
                simulator.tape.backward(simulator.loss)

            valid_frames += 1

            # 2. 提取 spring_Y 的梯度
            grad_spring_Y = wp.to_torch(simulator.wp_spring_Y.grad).clone()
            loss = wp.to_torch(simulator.loss, requires_grad=False)

            if (default_loss and (loss.item() > (default_loss[frame_idx-1] * 1.005))) or (torch.isnan(grad_spring_Y).any()):
                break

            losses.append(loss.item())

            # grad_spring_Y clip
            grad_spring_Y = torch.clamp(grad_spring_Y, min=-1000.0, max=1000.0)            
            grad_spring_Y_sequence.append(grad_spring_Y)

            if cfg.use_graph:
                # Only need to clear the gradient, the tape is created in the graph
                simulator.tape.zero()
            else:
                # Need to reset the compute graph and clear the gradient
                simulator.tape.reset()

            simulator.clear_loss()

            simulator.set_init_state(
                simulator.wp_states[-1].wp_x, 
                simulator.wp_states[-1].wp_v
            )
                
        # 转换输出张量 ... (保持不变)
        vertices_tensor = torch.stack(vertices_sequence, dim=0)
        velocities_tensor = torch.stack(velocities_sequence, dim=0)
        node_mass = torch.cat([wp.to_torch(simulator.wp_masses).clone(), torch.zeros(simulator.num_control_points, device=vertices_tensor.device)])
        spring_Y = wp.to_torch(simulator.wp_spring_Y).clone()

        spring_reset_length = wp.to_torch(simulator.wp_rest_lengths).clone()
        spring_dashpot_damping = simulator.dashpot_damping
        drag_damping = simulator.drag_damping

        if enable_backward:
            if len(grad_spring_Y_sequence) == 0:
                import pdb; pdb.set_trace()
            grad_spring_Y_avg = torch.mean(torch.stack(grad_spring_Y_sequence, dim=0), dim=0)  # 计算平均梯度

            # 返回平均梯度
            return vertices_tensor, velocities_tensor, node_mass, spring_Y, spring_reset_length, spring_dashpot_damping, drag_damping, losses, grad_spring_Y_avg, valid_frames
        else:
            return vertices_tensor, velocities_tensor, node_mass, spring_Y, spring_reset_length, spring_dashpot_damping, drag_damping


    def run_epoch(self, epoch, 
                  mode='train', 
                  ):
        torch.cuda.memory._record_memory_history(max_entries=100000)

        self.model.module.simulator.set_init_state(
            self.model.module.simulator.wp_init_vertices, 
            self.model.module.simulator.wp_init_velocities
        )

        _, _, _, _, _, _, _, losses, grad_spring_Y_avg, valid_frames = self.generate_data_point_sequence(
                update_frame_num=self.mdata.train_frame,
                enable_backward=True,
        )


        if mode != 'train':
            self.model.eval()
        mean_loss_insts = 0

        for interval in range(min(10, self.mdata.train_frame), self.mdata.train_frame, 10):

            # optimization
            self.optimizer.zero_grad()

            # Update simulator with predicted mechanical properties
            # Note: You may need to adjust this based on actual mech_info structure
            st = time()
            spring_pos = self.get_spring_midpoints()
            new_spring_Y = self.model(spring_pos)[:, 0]
            log_new_spring_Y = torch.log(new_spring_Y + 1e-8)  # 添加小值避免 log(0)
            
            
            # [关键] 映射到 Warp 时，必须设置 requires_grad=True
            # 这样 Warp 才会为这些数组分配梯度缓冲区，供 Tape 使用
            wp_predicted_spring_Y = wp.from_torch(log_new_spring_Y.contiguous(), dtype=wp.float32, requires_grad=True)

            self.model.module.simulator.wp_spring_Y = wp_predicted_spring_Y

            print("Forward time is : {}".format(time() - st))
            st = time()
            # 运行仿真
            if mode == 'train':
                # 获取平均梯度
                pos, vel, _, _, _, _, _, loss_val, grad_spring_Y_avg, valid_frames = \
                    self.generate_data_point_sequence(
                    update_frame_num=interval,
                    enable_backward=True,
                    default_loss=losses,
                )

                # Print spring_Y average
                print(f"\n{'='*60}")
                print(f"Spring_Y Average: {new_spring_Y.mean().item():.6f}")
                print(f"Loss value: {np.mean(loss_val):.10f}, default loss: {np.mean(losses):.10f}")
                print(f"Valid frames: {valid_frames}")
                print(f"{'='*60}\n")

                # -------------------------------------------------------
                # [核心修复] 使用平均梯度更新模型
                # -------------------------------------------------------

                # 将平均梯度手动注入 PyTorch 计算图
                torch.autograd.backward(
                    tensors=[log_new_spring_Y],
                    grad_tensors=[grad_spring_Y_avg]
                )

                # PyTorch 优化器更新神经网络参数
                self.optimizer.step()
                print(f"Model updated with average gradient per frame")

            else:
                # 验证模式不需要 tape
                _, _, _, _, _, _, _, loss = self.generate_data_point_sequence(
                    update_frame_num=self.mdata.train_frame,
                    enable_backward=False,
                    set_object_point=False
                )
                loss_val = loss if isinstance(loss, float) else loss # 处理验证集返回值

            print("Backward time : {}".format(time() - st))
            # stats
            mean_loss = np.mean(loss_val) if isinstance(loss_val, list) else loss_val

        # stats
        with torch.autograd.no_grad():
            mean_loss_insts += mean_loss

        # stats
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

        final_mech_info = torch.stack([new_spring_Y, self.spring_rest_length]
                                       , dim=0).detach()  # 这里根据实际 mech_info 结构调整

        return mean_loss_insts, final_mech_info

    
    def train(self):
        # load phystwin zero-grad optimization results and warp simulator here
        best_loss = 1e8
        best_mech_info = None
        best_epoch = None
        mech_info = None

        for epoch in self.pbar:
            mean_loss_train, mech_info = self.run_epoch(epoch, 
                                                        'train')

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

                    if best_loss > mean_loss_train:
                        best_mech_info = mech_info.clone()
                        best_epoch = epoch
                        best_loss = mean_loss_train

                # valid loss record
                if self.args.n_valid > 0 and (epoch+1) % 5 == 0:
                    mean_loss_valid = self.run_epoch(epoch, 
                                                    mech_info,
                                                    'test',
                                                    self.mdata)
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
                mean_loss_test = self.run_epoch(epoch, mode='test')
                print('epoch', epoch, 'RMSE/test', math.sqrt(mean_loss_test))
                if dist.get_rank() == 0:
                    self.writer.add_scalar('RMSE/test', math.sqrt(mean_loss_test), epoch)

            rmse.append(math.sqrt(mean_loss_test))

        if dist.get_rank() == 0:
            RMSE_cell = np.array([np.mean(rmse)])
            dump_path = os.path.join(self.args.dump_dir, 'test_RMSE', 'epoch_' + str(epoch) + '.csv')
            np.savetxt(dump_path, RMSE_cell, delimiter=',')
        
    
    def rollout(self, epoch=None, time_stps=None):
        epoch = self.args.restart_epoch if epoch == None else epoch
        instance_list = list(range(self.args.n_test))
        errors = []
        self.model.eval()
        with torch.autograd.no_grad():
            for i in tqdm(range(len(instance_list))):
                id = instance_list[i]
                # rollout for this instance, then record into a file
                mdata = self._create_datset_offline(id, mode='test')
                if time_stps is None:
                    L = mdata.in_feature.shape[0]
                    instance_rollout_error = np.zeros(L)
                    time_stps = L
                else:
                    L = time_stps
                    instance_rollout_error = np.zeros(L)
                for id_batch, b_data in enumerate(DataLoader(mdata, batch_size=1, shuffle=False)):
                    # print('id_batch', id_batch)
                    _, n, _ = mdata.in_feature.shape
                    m_ids = mdata.m_idx
                    m_gs_list = mdata.m_g
                    m_gs = [torch.tensor(g, dtype=torch.long).to(self.device) for g in m_gs_list]
                    if mdata.has_contact:
                        m_cgs = [torch.tensor(g, dtype=torch.long).to(self.device) for g in mdata.m_cgs[id_batch]]
                        m_g_cg = [[g, cg] for g, cg in zip(m_gs, m_cgs)]
                        m_gs = m_g_cg
                    pen_coeff = mdata.suggested_pen_coef().to(self.device)
                    if id_batch == 0:
                        current_stat = mdata[0].x.reshape(1, n, -1).to(self.device)
                    
                    b_data.y = b_data.y.reshape(1, n, -1).to(self.device)
                    loss, out, _ = self.model(m_ids, m_gs, current_stat, b_data.y, pen_coeff)
                    # record global error
                    instance_rollout_error[id_batch] = math.sqrt(loss.item())
                    # push forward state
                    current_stat = mdata._push_forward(out, current_stat)
                    current_stat = current_stat.detach()

                dir = os.path.join(self.args.dump_dir, f'rollout_RMSE_epoch_' + str(epoch))
                os.makedirs(dir, exist_ok=True)
                dump_path = os.path.join(dir, str(id) + '.csv')
                np.savetxt(dump_path, instance_rollout_error, delimiter=',')
                errors.append(np.mean(instance_rollout_error))

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
            mdata = self._create_datset_offline(mode='train')

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
            init_spring_Y=mdata.spring_Y[0],
            init_rest_lengths=mdata.spring_reset_length,
            init_masses=mdata.masses,
            num_object_springs=mdata.spring_Y.shape[0],
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
        mdata.train_frame = 58
        logger.info(f"Running pure inference simulation for {mdata.train_frame} frames...")
        for frame_idx in tqdm(range(1, mdata.train_frame)):
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

        plotter.open_movie(output_video_path, framerate=30)
        
        # plotter.show(auto_close=False)  # 必须先 show，建立窗口上下文

        # import pdb; pdb.set_trace()
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