import torch
from qqtt.utils import logger, cfg
import torch.nn as nn
import torch.distributed as dist
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import models as models
from End2End_EvoSpring_Reduction import End2EndReduction_EvoSpring
from End2End_Reduction_Dataset import End2EndReductionDataset
import math
import numpy as np
import os
import json
import warp as wp
import os
from time import time
import pickle
import logging

# 禁用所有 warp 相关的 logger
logging.getLogger("warp").setLevel(logging.ERROR)


class E2EReductionTrainer:
    def __init__(self, args, 
            device="cuda:0",
            
        ):
        self.args = args
        cfg.device = device

        self.mdata = self._create_dataset_offline(mode='train')
        self._create_model(cfg.init_spring_Y,
                            cfg.drag_damping,
                            cfg.dashpot_damping,
                            cfg.collide_elas,
                            cfg.collide_fric,
                            cfg.collide_object_elas,
                            cfg.collide_object_fric)
        
        # 初始化 Warp 优化器（仅在训练模式且 collision_learn 开启时）
        # Update without warp, only by NN for multilayer design
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.args.lr * min(np.sqrt(cfg.train_frame), 5), 
                                          betas=(0.9, 0.99))
        # max lr change to 5 for non cloth case

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
        
        self.pbar = tqdm(range(self.args.num_epochs), unit="iters")

        os.makedirs(self.args.dump_dir, exist_ok=True)
        for subdir in ['ckpts', 'log', 'test_RMSE', 'spring_mech_info', 'trajectories']:
            dir = os.path.join(self.args.dump_dir, subdir)
            os.makedirs(dir, exist_ok=True)

        self.total_update = 0

        # Track best results across all iterations
        self.global_best_loss = float('inf')
        self.global_best_mech_info = None
        self.global_best_epoch = None
        self.global_best_iteration = None

    def _create_model(self, init_spring_Y=None, init_drag_damping=None, init_dashpot_damping=None, init_collision_elas=None, init_collision_fric=None, init_collision_object_elas=None, init_collision_object_fric=None):
        self.object_case = self.args.object_case

        self.model = End2EndReduction_EvoSpring(
            pos_dim=self.args.space_dim,
            ld=self.args.hidden_dim,
            layer_num=self.args.multi_mesh_layer,
            pre_layer_num=self.args.pre_layer_num,
            bottom_layer_num=self.args.bottom_layer_num,
            mlp_hidden_layer=self.args.hidden_depth,
            MP_times=self.args.mp_time,
            enhance=self.args.enhance,
            agg_conv_pos=self.args.agg_conv_pos,
            default_spring_Y=init_spring_Y,
            default_drag_damping=init_drag_damping,
            default_dashpot_damping=init_dashpot_damping,
            default_collision_elas=init_collision_elas,
            default_collision_fric=init_collision_fric,
            default_collision_object_elas=init_collision_object_elas,
            default_collision_object_fric=init_collision_object_fric
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model.cuda(self.args.local_rank),
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=True  # [关键修复] 允许部分参数不参与计算
        )

    def _create_dataset_offline(self, mode='train', stride=1):
        if mode == 'train':
            mdata = End2EndReductionDataset(self.args.data_dir,
                                        layer_num=self.args.multi_mesh_layer,
                                        stride=stride,
                                        recal_mesh=self.args.recal_mesh,
                                        consist_mesh=self.args.consist_mesh,
                                        object_case=self.args.object_case,
                                        args=self.args, device=cfg.device)
        else:
            mdata = End2EndReductionDataset(self.args.data_dir,
                                        layer_num=self.args.multi_mesh_layer,
                                        stride=stride,
                                        recal_mesh=self.args.recal_mesh,
                                        consist_mesh=self.args.consist_mesh,
                                        mode=mode,
                                        object_case=self.args.object_case,  
                                        args=self.args, device=cfg.device)
        return mdata

    def positional_encoding(self, positions, num_freq_bands=10):
        """
        NeRF-style positional encoding
        Args:
            positions: [N, 3] normalized positions in [-1, 1]
            num_freq_bands: number of frequency bands (L)
        Returns:
            encoded_positions: [N, 3 * 2 * L] encoded positions
        """
        # positions shape: [N, 3]
        freq_bands = 2.0 ** torch.arange(num_freq_bands, dtype=positions.dtype, device=positions.device)  # [L]
        # freq_bands shape: [L]

        # Expand dimensions for broadcasting: positions [N, 3, 1], freq_bands [1, 1, L]
        pos_expanded = positions.unsqueeze(-1)  # [N, 3, 1]
        freq_expanded = freq_bands.unsqueeze(0).unsqueeze(0)  # [1, 1, L]

        # Compute scaled positions: [N, 3, L]
        scaled_pos = math.pi * pos_expanded * freq_expanded

        # Apply sin and cos
        sin_encoded = torch.sin(scaled_pos)  # [N, 3, L]
        cos_encoded = torch.cos(scaled_pos)  # [N, 3, L]

        # Interleave sin and cos: [N, 3, 2*L]
        encoded = torch.stack([sin_encoded, cos_encoded], dim=-1)  # [N, 3, L, 2]
        encoded = encoded.reshape(positions.shape[0], positions.shape[1], -1)  # [N, 3, 2*L]

        # Flatten to [N, 3 * 2 * L]
        encoded = encoded.reshape(positions.shape[0], -1)

        return encoded

    def save_traj(
        self, mlvl_mech_info=None, mlvl_topo_info=None, save_path=None, compute_loss=False
    ):
        if mlvl_topo_info is None or len(mlvl_topo_info) == 0:
            logger.error("Topology info is missing. Cannot initialize simulator.")
            return

        # ====================================================================
        # [核心修改 1]: 直接在内部初始化第一层 (Layer 0) 的 Simulator
        # ====================================================================
        logger.info("Initializing Level 0 simulator for trajectory rollout...")
        topo_0 = mlvl_topo_info[0]
        node_idx = topo_0['node_idx']
        spring_graph = topo_0['spring_graph']
        masses = topo_0['masses']
        node_type = topo_0['node_type']

        # 从 dataset (self.mdata) 中获取对应的初始物理状态
        init_node = self.mdata.init_vertices[node_idx]
        init_springs = spring_graph.T.int().contiguous()
        init_rest_lengths = torch.norm((init_node[spring_graph[0]] - init_node[spring_graph[1]]), dim=1)
        init_masses = masses[:, 0]

        # 提取 gt object points (处理碰撞体交互)
        object_node_idx = np.array(node_idx)[node_type[0].detach().cpu().numpy() == 0].tolist()
        gt_object_points = self.mdata.object_points[:, object_node_idx]
        gt_object_visibilities = self.mdata.object_visibilities[:, object_node_idx]
        gt_object_motions_valid = self.mdata.object_motions_valid[:, object_node_idx]

        # 实例化模拟器
        simulator = self.mdata.create_spring_mass_sim(
            init_node, 
            init_springs, 
            init_rest_lengths, 
            init_masses,
            node_type[0],
            gt_object_points, 
            gt_object_visibilities, 
            gt_object_motions_valid
        )

        # ====================================================================
        # 2. 将预测的第一层力学参数 (mech_info) 赋予模拟器
        # ====================================================================
        if mlvl_mech_info is not None and len(mlvl_mech_info) > 0:
            logger.info("Setting predicted mechanical properties to simulator (Level 0)")
            first_mech_info = mlvl_mech_info[0] 

            if isinstance(first_mech_info, dict):
                if 'log_spring_Y' in first_mech_info:
                    wp_predicted_spring_Y = wp.from_torch(first_mech_info['log_spring_Y'].contiguous(), dtype=wp.float32, requires_grad=False)
                    simulator.set_spring_Y(wp_predicted_spring_Y)

                if 'drag_damping' in first_mech_info:
                    wp_predicted_drag_damping = wp.from_torch(first_mech_info['drag_damping'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                    simulator.set_drag_damping(wp_predicted_drag_damping)

                if 'dashpot_damping' in first_mech_info:
                    wp_predicted_dashpot_damping = wp.from_torch(first_mech_info['dashpot_damping'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                    simulator.set_dashpot_damping(wp_predicted_dashpot_damping)

                if 'collision_elas' in first_mech_info:
                    wp_predicted_collision_elas = wp.from_torch(first_mech_info['collision_elas'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                    simulator.set_collision_elas(wp_predicted_collision_elas)

                if 'collision_fric' in first_mech_info:
                    wp_predicted_collision_fric = wp.from_torch(first_mech_info['collision_fric'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                    simulator.set_collision_fric(wp_predicted_collision_fric)

                if 'collision_object_elas' in first_mech_info:
                    wp_predicted_collision_object_elas = wp.from_torch(first_mech_info['collision_object_elas'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                    simulator.set_collision_object_elas(wp_predicted_collision_object_elas)

                if 'collision_object_fric' in first_mech_info:
                    wp_predicted_collision_object_fric = wp.from_torch(first_mech_info['collision_object_fric'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                    simulator.set_collision_object_fric(wp_predicted_collision_object_fric)
            else:
                logger.warning("first_mech_info has unexpected format (expected dict), skipping property update")

        # ====================================================================
        # 3. 运行前向仿真，记录轨迹
        # ====================================================================
        frame_len = cfg.train_frame + cfg.test_frame
        simulator.set_init_state(
            simulator.wp_init_vertices, simulator.wp_init_velocities
        )
        
        vertices = [wp.to_torch(simulator.wp_states[0].wp_x, requires_grad=False).cpu()]
        velocities = [wp.to_torch(simulator.wp_states[0].wp_v, requires_grad=False).cpu()]

        frame_losses, chamfer_losses, track_losses = [0], [0], [0]

        for i in tqdm(range(1, frame_len), desc="Saving trajectory"):
            if cfg.data_type == "real":
                simulator.set_controller_target(i, pure_inference=False)
            if simulator.object_collision_flag:
                simulator.update_collision_graph()

            if cfg.use_graph:
                wp.capture_launch(simulator.forward_graph)
            else:
                simulator.step()

            if compute_loss:
                if cfg.data_type == "real":
                    simulator.calculate_loss()
                    chamfer_losses.append(wp.to_torch(simulator.chamfer_loss, requires_grad=False).item())
                    track_losses.append(wp.to_torch(simulator.track_loss, requires_grad=False).item())
                else:
                    simulator.calculate_simple_loss()
                    chamfer_losses.append(0.0)
                    track_losses.append(0.0)

                frame_losses.append(wp.to_torch(simulator.loss, requires_grad=False).item())
                simulator.clear_loss()

            x = wp.to_torch(simulator.wp_states[-1].wp_x, requires_grad=False)
            v = wp.to_torch(simulator.wp_states[-1].wp_v, requires_grad=False)
            vertices.append(x.cpu())
            velocities.append(v.cpu())
            simulator.set_init_state(simulator.wp_states[-1].wp_x, simulator.wp_states[-1].wp_v)

        vertices = torch.stack(vertices, dim=0)
        velocities = torch.stack(velocities, dim=0)

        # ====================================================================
        # 4. 打包并保存结果 (.pkl)
        # ====================================================================
        logger.info(f"Save the trajectory to {save_path}")
        
        mlvl_data_to_save = []
        if mlvl_mech_info is not None and mlvl_topo_info is not None:
            for lvl_idx, (mech, topo) in enumerate(zip(mlvl_mech_info, mlvl_topo_info)):
                lvl_dict = {
                    'level': lvl_idx,
                    'mechanics': {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in mech.items()},
                    'topology': {
                        'masses': topo['masses'].detach().cpu().numpy() if isinstance(topo['masses'], torch.Tensor) else topo['masses'],
                        'spring_graph': topo['spring_graph'].detach().cpu().numpy() if isinstance(topo['spring_graph'], torch.Tensor) else topo['spring_graph'],
                        'node_idx': topo['node_idx'],
                        'node_type': topo['node_type'].detach().cpu().numpy() if isinstance(topo['node_type'], torch.Tensor) else topo['node_type']
                    }
                }
                mlvl_data_to_save.append(lvl_dict)

        save_data = {
            'vertices': vertices.cpu().numpy(),       # 仅第一层的轨迹
            'velocities': velocities.cpu().numpy(),   # 仅第一层的速度
            'multi_level_info': mlvl_data_to_save     # 保存所有层级的力学和拓扑信息
        }

        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Trajectory saved successfully with shape {save_data['vertices'].shape}")
        if compute_loss:
            return frame_losses, chamfer_losses, track_losses, save_data

    def _preproc_multi_infos(self, mdata):
        # process the multi-level mesh for batched data here
    
        # no contact, then share the graph between batches
        # only keep the first layer
        m_ids = []
        m_gs_list = [np.concatenate([mdata.cells, mdata.cells[[1,0]]], axis = 1)]
        # m_gs_parents_list = mdata.m_edge_parents[0:1]
        m_gs = [torch.tensor(g, dtype=torch.long).to(cfg.device) for g in m_gs_list]
        # m_gs_parents = [torch.tensor(g_parent, dtype=torch.long).to(cfg.device) for g_parent in m_gs_parents_list]

        return m_ids, m_gs
    
    def generate_data_point_sequence(self, simulator, update_frame_num, enable_backward=False):
        vertices_sequence = []
        velocities_sequence = []

        # 存储所有可微分参数的梯度序列
        grad_sequences = {
            'log_spring_Y': [],
            'collision_elas': [],
            'collision_fric': [],
            'collision_object_elas': [],
            'collision_object_fric': [],
            'drag_damping': [],
            'dashpot_damping': []
        }

        # [修改] 定义 Warp 端的 Loss 累加器
        losses = []
        chamfer_losses = []
        track_losses = []

        # 将 cfg.device 更新为 Warp 兼容的字符串或直接获取 Warp device 对象
        simulator.set_init_state(
            simulator.wp_init_vertices, 
            simulator.wp_init_velocities
        )
        
        vertices_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[0].wp_x).clone(), wp.to_torch(simulator.wp_states[0].wp_control_x).clone()]))
        velocities_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[0].wp_v).clone(), wp.to_torch(simulator.wp_states[0].wp_control_v).clone()]))

        valid_frames = 0
        
        for frame_idx in tqdm(range(1, update_frame_num)):
            simulator.set_controller_target(frame_idx)

            # Record data
            vertices_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[-1].wp_x).clone(), wp.to_torch(simulator.wp_states[-1].wp_control_x).clone()]))
            velocities_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[-1].wp_v).clone(), wp.to_torch(simulator.wp_states[-1].wp_control_v).clone()]))

            if simulator.object_collision_flag:
                simulator.update_collision_graph()

            # 计算 Loss
            if cfg.use_graph:
                wp.capture_launch(simulator.graph)
            else:
                if cfg.data_type == "real":
                    with simulator.tape:
                        simulator.step()
                        simulator.calculate_loss()
                    simulator.tape.backward(simulator.loss)
                else:
                    with simulator.tape:
                        simulator.step()
                        simulator.calculate_simple_loss()
                    simulator.tape.backward(simulator.loss)
            # simulator.export_forces_to_txt(frame_idx=frame_idx, filename="simulation_forces.txt")
            # import pdb; pdb.set_trace()
            valid_frames += 1

            # 提取所有可微分参数的梯度
            grad_dict = {}

            # spring_Y 梯度
            if simulator.wp_spring_Y.grad is not None:
                grad_dict['log_spring_Y'] = wp.to_torch(simulator.wp_spring_Y.grad).clone()
            else:
                grad_dict['log_spring_Y'] = torch.zeros_like(wp.to_torch(simulator.wp_spring_Y))

            # 碰撞参数梯度
            if cfg.collision_learn:
                if simulator.wp_collide_elas.grad is not None:
                    grad_dict['collision_elas'] = wp.to_torch(simulator.wp_collide_elas.grad).clone()
                else:
                    grad_dict['collision_elas'] = torch.zeros_like(wp.to_torch(simulator.wp_collide_elas))

                if simulator.wp_collide_fric.grad is not None:
                    grad_dict['collision_fric'] = wp.to_torch(simulator.wp_collide_fric.grad).clone()
                else:
                    grad_dict['collision_fric'] = torch.zeros_like(wp.to_torch(simulator.wp_collide_fric))

                if simulator.wp_collide_object_elas.grad is not None:
                    grad_dict['collision_object_elas'] = wp.to_torch(simulator.wp_collide_object_elas.grad).clone()
                else:
                    grad_dict['collision_object_elas'] = torch.zeros_like(wp.to_torch(simulator.wp_collide_object_elas))

                if simulator.wp_collide_object_fric.grad is not None:
                    grad_dict['collision_object_fric'] = wp.to_torch(simulator.wp_collide_object_fric.grad).clone()
                else:
                    grad_dict['collision_object_fric'] = torch.zeros_like(wp.to_torch(simulator.wp_collide_object_fric))

                # 阻尼参数梯度 (Damping Parameters)
                # 检查 simulator 是否有 Warp 数组版本的阻尼参数
                if hasattr(simulator, 'wp_drag_damping') and simulator.wp_drag_damping.grad is not None:
                    grad_dict['drag_damping'] = wp.to_torch(simulator.wp_drag_damping.grad).clone()
                elif hasattr(simulator, 'wp_drag_damping'):
                    grad_dict['drag_damping'] = torch.zeros_like(wp.to_torch(simulator.wp_drag_damping))
                else:
                    # 如果没有 Warp 数组版本，创建标量零梯度
                    grad_dict['drag_damping'] = torch.zeros(1, dtype=torch.float32, device=cfg.device)

                if hasattr(simulator, 'wp_dashpot_damping') and simulator.wp_dashpot_damping.grad is not None:
                    grad_dict['dashpot_damping'] = wp.to_torch(simulator.wp_dashpot_damping.grad).clone()
                elif hasattr(simulator, 'wp_dashpot_damping'):
                    grad_dict['dashpot_damping'] = torch.zeros_like(wp.to_torch(simulator.wp_dashpot_damping))
                else:
                    # 如果没有 Warp 数组版本，创建标量零梯度
                    grad_dict['dashpot_damping'] = torch.zeros(1, dtype=torch.float32, device=cfg.device)


            # 提取 chamfer_loss 和 track_loss (仅对 real 数据)
            if cfg.data_type == "real":
                chamfer_loss = wp.to_torch(simulator.chamfer_loss, requires_grad=False)
                track_loss = wp.to_torch(simulator.track_loss, requires_grad=False)
            else:
                chamfer_loss = torch.tensor(0.0)
                track_loss = torch.tensor(0.0)

            loss = wp.to_torch(simulator.loss, requires_grad=False)

            # 检查梯度是否有 NaN
            has_nan = False

            for param_name, grad in grad_dict.items():
                if torch.isnan(grad).any():
                    print(f'Gradient of {param_name} turns to nan in step {frame_idx}')
                    has_nan = True

            if has_nan:
                break
            
            losses.append(loss.item())
            chamfer_losses.append(chamfer_loss.item())
            track_losses.append(track_loss.item())

            # 保存所有梯度
            for param_name, grad in grad_dict.items():
                grad_sequences[param_name].append(grad)
    
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

        # # 转换输出张量 ... (保持不变)
        vertices_tensor = [] # torch.stack(vertices_sequence, dim=0)
        velocities_tensor =  [] # torch.stack(velocities_sequence, dim=0)
        node_mass = [] # torch.cat([wp.to_torch(simulator.wp_masses).clone(), torch.zeros(simulator.num_control_points, device=vertices_tensor.device)])
        spring_Y = [] # wp.to_torch(simulator.wp_spring_Y).clone()

        spring_rest_length = [] # wp.to_torch(simulator.wp_rest_lengths).clone()
        spring_dashpot_damping = None # simulator.wp_dashpot_damping
        drag_damping = None # simulator.wp_drag_damping

        if enable_backward:
            # 计算所有参数的平均梯度
            grad_avg_dict = {}
            for param_name, grad_list in grad_sequences.items():
                if len(grad_list) == 0:
                    # 如果没有梯度，创建零梯度
                    if param_name == 'log_spring_Y':
                        grad_avg_dict[param_name] = torch.zeros_like(spring_Y)
                    elif param_name in ['collision_elas', 'collision_fric', 'collision_object_elas', 'collision_object_fric']:
                        if cfg.collision_learn:
                            # 将 collision_ 前缀转换为 collide_ 以匹配 simulator 属性名
                            wp_param_name = 'wp_' + param_name.replace('collision_', 'collide_')
                            grad_avg_dict[param_name] = torch.zeros_like(wp.to_torch(getattr(simulator, wp_param_name)))
                    elif param_name in ['drag_damping', 'dashpot_damping']:
                        if cfg.collision_learn:
                            # 为阻尼参数创建标量零梯度
                            grad_avg_dict[param_name] = torch.zeros(1, dtype=torch.float32, device=cfg.device)
                else:
                    # 计算累积梯度
                    grad_avg_dict[param_name] = torch.sum(torch.stack(grad_list, dim=0), dim=0)

            # 返回平均梯度字典
            return vertices_tensor, velocities_tensor, node_mass, spring_Y, spring_rest_length, spring_dashpot_damping, drag_damping, losses, chamfer_losses, track_losses, grad_avg_dict, valid_frames
        else:
            return vertices_tensor, velocities_tensor, node_mass, spring_Y, spring_rest_length, spring_dashpot_damping, drag_damping, losses, chamfer_losses, track_losses


    def run_epoch(self, epoch, 
                  mode='train', 
                  ):
        # get node_pos, node_mass, rest_length , drag_damping, dashpot_damping in first frame from mdata
        train_node_pos = self.mdata.init_vertices
        train_node_type = torch.FloatTensor(self.mdata.node_type[0]).to(cfg.device)
        train_node_mass = self.mdata.init_masses[:, None]

        # normalize train node pos into [-1,1]
        pos_min = train_node_pos.min(dim=0, keepdim=True)[0]
        pos_max = train_node_pos.max(dim=0, keepdim=True)[0]
        normalized_train_node_pos = 2 * (train_node_pos - pos_min) / (pos_max - pos_min) - 1

        # apply NeRF-style positional encoding to node positions
        pos_encoded = self.positional_encoding(normalized_train_node_pos, num_freq_bands=10)

        node_in_feature = torch.cat([
            normalized_train_node_pos, pos_encoded,  train_node_mass, train_node_type
        ], dim = 1)

        if mode != 'train':
            self.model.eval()
            
        # 运行仿真
        if mode == 'train':
            for i in range(10):
                # 获取平均梯度
                self.optimizer.zero_grad()
                m_ids, m_gs = self._preproc_multi_infos(self.mdata)

                st = time()

                mlvl_info = self.model(m_ids, m_gs, node_in_feature)

                # [关键] 映射到 Warp 时，必须设置 requires_grad=True
                # 这样 Warp 才会为这些数组分配梯度缓冲区，供 Tape 使用

                gradients, losses = [], []

                # -------------------------------------------------------
                # 初始化收集器，用于累加所有层级的信息
                # -------------------------------------------------------
                all_tensors_to_backward = []
                all_grad_tensors_to_backward = []
                mlvl_current_mech_info = [] # 保存当前迭代的所有层级力学参数
                mlvl_current_topo_info = [] # [新增] 保存当前迭代的所有层级拓扑参数

                # [修复] 加上星号 * 进行正确的 tuple 解包
                for new_spring_Y, drag_damping_out, \
                    dashpot_damping_out, collide_fric, collide_elas,\
                          masses, spring_graph, node_idx, node_type \
                            in zip(*mlvl_info):
                    
                    # [新增] 收集并保存当前层级的拓扑与质量信息
                    mlvl_current_topo_info.append({
                        'masses': masses.clone(),
                        'spring_graph': spring_graph.clone(),
                        'node_idx': node_idx,
                        'node_type': node_type.clone()
                    })
                    
                    init_node = self.mdata.init_vertices[node_idx]
                    init_springs = spring_graph.T.int().contiguous()
                    init_rest_lengths = torch.norm((init_node[spring_graph[0]]-init_node[spring_graph[1]]), dim = 1)
                    init_masses = masses[:, 0]

                    # downsample for gt points
                    object_node_idx = np.array(node_idx)[node_type[0].detach().cpu().numpy()==0].tolist()
                    gt_object_points = self.mdata.object_points[:, object_node_idx]
                    gt_object_visibilities = self.mdata.object_visibilities[:, object_node_idx]
                    gt_object_motions_valid = self.mdata.object_motions_valid[:, object_node_idx]

                    # create simulator
                    simulator = self.mdata.create_spring_mass_sim(init_node, 
                                                                  init_springs, 
                                                                  init_rest_lengths, 
                                                                  init_masses,
                                                                  node_type[0],
                                                                  gt_object_points, 
                                                                  gt_object_visibilities, 
                                                                  gt_object_motions_valid)

                    # set up spring mass parameters
                    log_new_spring_Y = torch.log(new_spring_Y)

                    wp_predicted_spring_Y = wp.from_torch(
                    log_new_spring_Y.contiguous(), dtype=wp.float32, requires_grad=True
                    )

                    # Convert predicted damping values to warp
                    wp_predicted_drag_damping = wp.from_torch(
                        drag_damping_out.contiguous(), dtype=wp.float32, requires_grad=True
                    )
                    wp_predicted_dashpot_damping = wp.from_torch(
                        dashpot_damping_out.contiguous(), dtype=wp.float32, requires_grad=True
                    )

                    # Convert predicted collision values to warp
                    wp_predicted_collision_elas = wp.from_torch(
                        collide_elas.contiguous(), dtype=wp.float32, requires_grad=True
                    )
                    wp_predicted_collision_fric = wp.from_torch(
                        collide_fric.contiguous(), dtype=wp.float32, requires_grad=True
                    )

                    # Set predicted spring_Y and damping to simulator
                    simulator.set_spring_Y(wp_predicted_spring_Y)
                    simulator.set_drag_damping(wp_predicted_drag_damping)
                    simulator.set_dashpot_damping(wp_predicted_dashpot_damping)
                    simulator.set_collision_elas(wp_predicted_collision_elas)
                    simulator.set_collision_fric(wp_predicted_collision_fric)
                        
                    pos, vel, _, _, _, _, _, loss_val, chamfer_loss_val, track_loss_val, grad_avg_dict, valid_frames = \
                        self.generate_data_point_sequence(
                            simulator,
                            update_frame_num=cfg.train_frame,
                            enable_backward=True,
                        )
                    
                    gradients.append(grad_avg_dict)
                    losses.append(loss_val)

                    # 将这一层的 tensor 和梯度追加到总列表中
                    all_tensors_to_backward.extend([
                        log_new_spring_Y,
                        drag_damping_out,
                        dashpot_damping_out,
                        collide_elas,
                        collide_fric
                    ])
                    
                    all_grad_tensors_to_backward.extend([
                        grad_avg_dict['log_spring_Y'],
                        grad_avg_dict['drag_damping'],
                        grad_avg_dict['dashpot_damping'],
                        grad_avg_dict['collision_elas'],
                        grad_avg_dict['collision_fric']
                    ])

                    # 将这一层的力学状态保存为一个独立的字典
                    mlvl_current_mech_info.append({
                        'log_spring_Y': log_new_spring_Y.detach().clone(),
                        'drag_damping': drag_damping_out.detach().clone(),
                        'dashpot_damping': dashpot_damping_out.detach().clone(),
                        'collision_elas': collide_elas.detach().clone(),
                        'collision_fric': collide_fric.detach().clone(),
                    })
                    # --- 结束当前层级的循环 ---

                    break # DEBUG(cxy) : Only supervision with first one layer

                print("Forward time is : {}".format(time() - st))
                st = time()
                
                self.total_update += 1

                # Print params (using the last level evaluated for printing)
                print(f"\n{'='*60}")
                print(f"Spring Y iteration {i+1}/10 epoch {epoch}")
                print(f"spring_Y grad Average: {grad_avg_dict['log_spring_Y'].mean().item()}")
                print(f"Spring_Y Average: {log_new_spring_Y.mean().item():.6f}")
                print(f"drag_damping grad Average: {grad_avg_dict['drag_damping'].mean().item()}")
                print(f"Drag_Damping: {drag_damping_out.item():.6f}")
                print(f"dashpot_damping grad Average: {grad_avg_dict['dashpot_damping'].mean().item()}")
                print(f"Dashpot_Damping: {dashpot_damping_out.item():.6f}")
                print(f"collision_elas grad Average: {grad_avg_dict['collision_elas'].mean().item()}")
                print(f"Collision_Elas: {collide_elas.item():.6f}")
                print(f"collision_fric grad Average: {grad_avg_dict['collision_fric'].mean().item()}")
                print(f"Collision_Fric: {collide_fric.item():.6f}")
                print(f"Loss sum: {np.sum(loss_val):.10f}")
                print(f"Chamfer loss sum: {np.sum(chamfer_loss_val):.10f}")
                print(f"Track loss sum: {np.sum(track_loss_val):.10f}")
                print(f"Valid frames: {valid_frames}")
                print(f"{'='*60}\n")

                self.writer.add_scalar('Spring_Y_update/overall_Loss', np.sum(loss_val), self.total_update)
                self.writer.add_scalar('Spring_Y_update/chamfer_Loss', np.sum(chamfer_loss_val), self.total_update)
                self.writer.add_scalar('Spring_Y_update/track_Loss', np.sum(track_loss_val), self.total_update)
                self.writer.add_scalar('Spring_Y_update/Spring_Y_Average', log_new_spring_Y.mean().item(), self.total_update)
                self.writer.add_scalar('Spring_Y_update/Drag_Damping', drag_damping_out.item(), self.total_update)
                self.writer.add_scalar('Spring_Y_update/Dashpot_Damping', dashpot_damping_out.item(), self.total_update)
                self.writer.add_scalar('Spring_Y_update/Collision_Elas', collide_elas.item(), self.total_update)
                self.writer.add_scalar('Spring_Y_update/Collision_Fric', collide_fric.item(), self.total_update)

                # -------------------------------------------------------
                # 使用平均梯度更新模型
                # -------------------------------------------------------
                print("Forward processing overhead time is : {}".format(time() - st))
                st = time()
                
                # 计算所有层级的总 Loss，用于判定全局最优和记录
                total_loss_val = np.sum([np.sum(l) for l in losses])

                # -------------------------------------------------------
                # 在循环外部，一次性对所有收集到的 Tensor 计算图反向传播
                # -------------------------------------------------------
                torch.autograd.backward(
                    tensors=all_tensors_to_backward,
                    grad_tensors=all_grad_tensors_to_backward
                )

                # PyTorch 优化器更新神经网络参数
                self.optimizer.step()
                print(f"Model updated with average gradient per frame for ALL levels")

                # Check if this is the best iteration and save immediately
                current_loss = np.sum(total_loss_val)
                if current_loss < self.global_best_loss:
                    prev_best_loss = self.global_best_loss
                    self.global_best_loss = current_loss
                    self.global_best_epoch = epoch
                    self.global_best_iteration = i

                    # Save best model checkpoint
                    best_ckpt_path = os.path.join(self.args.dump_dir, 'ckpts', f'best_iter_epoch{epoch}_iter{i}')
                    torch.save(self.model.module.state_dict(), best_ckpt_path)

                    # [修复]: 保存包含所有层级的 mech_info 列表和 topo_info 列表
                    self.global_best_mech_info = [info.copy() for info in mlvl_current_mech_info]
                    self.global_best_topo_info = [info.copy() for info in mlvl_current_topo_info]
                    
                    best_mech_info_path = os.path.join(self.args.dump_dir, 'spring_mech_info', f'best_iter_epoch{epoch}_iter{i}')
                    torch.save({'mech': self.global_best_mech_info, 'topo': self.global_best_topo_info}, best_mech_info_path)

                    # Save best trajectory
                    best_traj_path = os.path.join(self.args.dump_dir, 'trajectories', 'best_trajectory.pkl')
                    logger.info(f"New best loss {total_loss_val:.10f}. Saving checkpoint and trajectory...")
                    
                    # [修复]: 不再传入 simulator，直接传入多层结构信息，让其内部去初始化
                    self.save_traj(mlvl_mech_info=self.global_best_mech_info, 
                                   mlvl_topo_info=self.global_best_topo_info,
                                   save_path=best_traj_path, 
                                   compute_loss=True)

                    # Log to tensorboard
                    self.writer.add_scalar('Best/loss', current_loss, self.total_update)
                    self.writer.add_scalar('Best/epoch', epoch, self.total_update)
                    self.writer.add_scalar('Best/iteration', i, self.total_update)

                    print(f"\n{'='*60}")
                    print(f"NEW BEST RESULT SAVED!")
                    print(f"Epoch: {epoch}, Iteration: {i+1}/10")
                    print(f"Loss: {current_loss:.10f}")
                    print(f"Previous best: {prev_best_loss:.10f}")
                    print(f"{'='*60}\n")
           
        else:
            # 验证模式不需要 tape
            _, _, _, _, _, _, _, loss, chamfer_loss, track_loss = self.generate_data_point_sequence(
                update_frame_num=self.mdata.train_frame,
                enable_backward=False,
                set_object_point=False
            )
            loss_val = loss if isinstance(loss, float) else loss # 处理验证集返回值
            total_loss_val = loss_val

        print("Backward/Overhead time : {}".format(time() - st))
        # stats
        mean_loss = np.sum(total_loss_val) if isinstance(total_loss_val, list) else total_loss_val

        # opt scheduler
        if mode == 'train':
            if epoch < self.args.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                if self.optimizer.param_groups[0]['lr'] > 1e-6:
                    self.scheduler.step()
                    if self.optimizer.param_groups[0]['lr'] < 1e-6:
                        self.optimizer.param_groups[0]['lr'] = 1e-6
        else:
            self.model.eval()

        # [修复]: 返回总 loss，以及包含所有层的拓扑结构和力学参数列表
        if mode == 'train':
            return mean_loss, mlvl_current_mech_info, mlvl_current_topo_info
        else:
            return mean_loss, None, None

    
    def train(self):
        # Initialize global_best_loss with the initial trajectory loss if not set
        for epoch in self.pbar:
            self.run_epoch(epoch, 'train')

            
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
                    m_gs = [torch.tensor(g, dtype=torch.long).to(cfg.device) for g in m_gs_list]
                    if mdata.has_contact:
                        m_cgs = [torch.tensor(g, dtype=torch.long).to(cfg.device) for g in mdata.m_cgs[id_batch]]
                        m_g_cg = [[g, cg] for g, cg in zip(m_gs, m_cgs)]
                        m_gs = m_g_cg
                    pen_coeff = mdata.suggested_pen_coef().to(cfg.device)
                    if id_batch == 0:
                        current_stat = mdata[0].x.reshape(1, n, -1).to(cfg.device)
                    
                    b_data.y = b_data.y.reshape(1, n, -1).to(cfg.device)
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
