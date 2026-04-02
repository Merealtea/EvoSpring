import torch
from qqtt.utils import logger, cfg
import torch.nn as nn
import torch.distributed as dist
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import models as models
from End2End_EvoSpring import End2End_EvoSpring
from End2End_Dataset import End2EndDataset
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

@wp.kernel
def accum_loss_kernel(loss_accum: wp.array(dtype=float), frame_loss: wp.array(dtype=float)):
    # 简单的累加操作: loss_accum[0] += frame_loss[0]
    wp.atomic_add(loss_accum, 0, frame_loss[0])

class E2ETrainer:
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
        
        # create warp simulator
        self.model.module.load_warp_simulator(self.mdata.simulator)

        # 初始化 Warp 优化器（仅在训练模式且 collision_learn 开启时）
        if cfg.collision_learn:
            # 确保 requires_grad=True
            # 碰撞参数 (Collision Parameters)
            self.torch_collide_elas = wp.to_torch(self.model.module.simulator.wp_collide_elas)
            self.torch_collide_fric = wp.to_torch(self.model.module.simulator.wp_collide_fric)
            self.torch_collide_object_elas = wp.to_torch(self.model.module.simulator.wp_collide_object_elas)
            self.torch_collide_object_fric = wp.to_torch(self.model.module.simulator.wp_collide_object_fric)

            # 将所有可微分参数添加到列表
            warp_params_list = [
                self.torch_collide_elas,
                self.torch_collide_fric,
                self.torch_collide_object_elas,
                self.torch_collide_object_fric,
            ]
        else:
            warp_params_list = []

        self.default_spring_Y = wp.to_torch(self.model.module.simulator.wp_spring_Y).detach().clone()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr * min(np.sqrt(cfg.train_frame), 5), betas=(0.9, 0.99))
        # max lr change to 5 for non cloth case

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

        self.total_update = 0

        # Track best results across all iterations
        self.global_best_loss = float('inf')
        self.global_best_mech_info = None
        self.global_best_epoch = None
        self.global_best_iteration = None


    def _create_dataset_offline(self, mode='train', stride=1):
        if mode == 'train':
            mdata = End2EndDataset(self.args.data_dir,
                                        layer_num=self.args.multi_mesh_layer,
                                        stride=stride,
                                        recal_mesh=self.args.recal_mesh,
                                        consist_mesh=self.args.consist_mesh,
                                        object_case=self.args.object_case,
                                        args=self.args, device=cfg.device)
        else:
            mdata = End2EndDataset(self.args.data_dir,
                                        layer_num=self.args.multi_mesh_layer,
                                        stride=stride,
                                        recal_mesh=self.args.recal_mesh,
                                        consist_mesh=self.args.consist_mesh,
                                        mode=mode,
                                        object_case=self.args.object_case,  
                                        args=self.args, device=cfg.device)
        return mdata


    def _create_model(self, init_spring_Y=None, init_drag_damping=None, init_dashpot_damping=None, init_collision_elas=None, init_collision_fric=None, init_collision_object_elas=None, init_collision_object_fric=None):
        self.object_case = self.args.object_case

        self.model = End2End_EvoSpring(
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
        self,  mech_info=None, save_path=None, compute_loss=False
    ):
        # Get simulator reference
        simulator = self.model.module.simulator
        

        # 1. Set the mech_info to the simulator for recording
        if mech_info is not None:
            logger.info("Setting predicted mechanical properties to simulator")
            if isinstance(mech_info, dict):
                # Set spring_Y if available
                if 'log_spring_Y' in mech_info:
                    predicted_spring_Y = mech_info['log_spring_Y']
                    wp_predicted_spring_Y = wp.from_torch(predicted_spring_Y.contiguous(), dtype=wp.float32, requires_grad=False)
                    simulator.set_spring_Y(wp_predicted_spring_Y)

                # Set drag_damping if available
                if 'drag_damping' in mech_info:
                    predicted_drag_damping = mech_info['drag_damping']
                    wp_predicted_drag_damping = wp.from_torch(predicted_drag_damping.reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                    simulator.set_drag_damping(wp_predicted_drag_damping)

                # Set dashpot_damping if available
                if 'dashpot_damping' in mech_info:
                    predicted_dashpot_damping = mech_info['dashpot_damping']
                    wp_predicted_dashpot_damping = wp.from_torch(predicted_dashpot_damping.reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                    simulator.set_dashpot_damping(wp_predicted_dashpot_damping)

                # Set collision parameters if available
                if 'collision_elas' in mech_info:
                    predicted_collision_elas = mech_info['collision_elas']
                    wp_predicted_collision_elas = wp.from_torch(predicted_collision_elas.reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                    simulator.set_collision_elas(wp_predicted_collision_elas)

                if 'collision_fric' in mech_info:
                    predicted_collision_fric = mech_info['collision_fric']
                    wp_predicted_collision_fric = wp.from_torch(predicted_collision_fric.reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                    simulator.set_collision_fric(wp_predicted_collision_fric)

                if 'collision_object_elas' in mech_info:
                    predicted_collision_object_elas = mech_info['collision_object_elas']
                    wp_predicted_collision_object_elas = wp.from_torch(predicted_collision_object_elas.reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                    simulator.set_collision_object_elas(wp_predicted_collision_object_elas)

                if 'collision_object_fric' in mech_info:
                    predicted_collision_object_fric = mech_info['collision_object_fric']
                    wp_predicted_collision_object_fric = wp.from_torch(predicted_collision_object_fric.reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                    simulator.set_collision_object_fric(wp_predicted_collision_object_fric)
            else:
                logger.warning("mech_info has unexpected format (expected dict), skipping property update")

        # 2. Start simulation and record the trajectory
        frame_len = cfg.train_frame + cfg.test_frame
        simulator.set_init_state(
            simulator.wp_init_vertices, simulator.wp_init_velocities
        )
        
        vertices = [
            wp.to_torch(simulator.wp_states[0].wp_x, requires_grad=False).cpu()
        ]
        velocities = [
            wp.to_torch(simulator.wp_states[0].wp_v, requires_grad=False).cpu()
        ]

        
        # Initialize loss storage if compute_loss is True
        frame_losses = [0]
        chamfer_losses = [0]
        track_losses = [0]

        for i in tqdm(range(1, frame_len), desc="Saving trajectory"):
            if cfg.data_type == "real":
                simulator.set_controller_target(i, pure_inference=False)
            if simulator.object_collision_flag:
                simulator.update_collision_graph()

            if cfg.use_graph:
                wp.capture_launch(simulator.forward_graph)
            else:
                simulator.step()

            # Compute loss if requested
            if compute_loss:
                if cfg.data_type == "real":
                    simulator.calculate_loss()
                    chamfer_loss = wp.to_torch(simulator.chamfer_loss, requires_grad=False).item()
                    track_loss = wp.to_torch(simulator.track_loss, requires_grad=False).item()
                    chamfer_losses.append(chamfer_loss)
                    track_losses.append(track_loss)
                else:
                    simulator.calculate_simple_loss()
                    chamfer_losses.append(0.0)
                    track_losses.append(0.0)

                frame_loss = wp.to_torch(simulator.loss, 
                                         requires_grad=False).item()
                frame_losses.append(frame_loss)
                simulator.clear_loss()

            x = wp.to_torch(simulator.wp_states[-1].wp_x, requires_grad=False)
            v = wp.to_torch(simulator.wp_states[-1].wp_v, requires_grad=False)
            vertices.append(x.cpu())
            velocities.append(v.cpu())
            # Set the intial state for the next step
            simulator.set_init_state(
                simulator.wp_states[-1].wp_x,
                simulator.wp_states[-1].wp_v,
            )

        vertices = torch.stack(vertices, dim=0)
        velocities = torch.stack(velocities, dim=0)

        # 3. Save the trajectory to a file
        logger.info(f"Save the trajectory to {save_path}")
        vertices_to_save = vertices.cpu().numpy()
        velocities_to_save = velocities.cpu().numpy()

        # Prepare data to save
        save_data = {
            'vertices': vertices_to_save,
            'velocities': velocities_to_save,
        }

        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Trajectory saved successfully with shape {vertices_to_save.shape}")
        if compute_loss:
            return frame_losses, chamfer_losses, track_losses, save_data

    def _preproc_multi_infos(self, mdata, node_in_feature, edge_mech_in_feature):
        # process the multi-level mesh for batched data here
    
        # no contact, then share the graph between batches
        # only need to reshape input tensor
        m_ids = mdata.m_idx
        m_gs_list = mdata.m_g
        m_gs_parents_list = mdata.m_edge_parents
        
        m_gs = [torch.tensor(g, dtype=torch.long).to(cfg.device) for g in m_gs_list]
        m_gs_parents = [torch.tensor(g_parent, dtype=torch.long).to(cfg.device) for g_parent in m_gs_parents_list]
        
        # Load data to GPU
        node_in_feature = node_in_feature.to(cfg.device)
        edge_mech_in_feature = torch.cat([edge_mech_in_feature,edge_mech_in_feature], dim = 0).to(cfg.device)

        return m_ids, m_gs, m_gs_parents, node_in_feature, edge_mech_in_feature
    
    def generate_data_point_sequence(self, update_frame_num, enable_backward=False):
        simulator = self.model.module.simulator
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
        self.model.module.simulator.set_init_state(
            self.model.module.simulator.wp_init_vertices, 
            self.model.module.simulator.wp_init_velocities
        )
        
        vertices_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[0].wp_x).clone(), wp.to_torch(simulator.wp_states[0].wp_control_x).clone()]))
        velocities_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[0].wp_v).clone(), wp.to_torch(simulator.wp_states[0].wp_control_v).clone()]))

        valid_frames = 0
        
        for frame_idx in tqdm(range(1, update_frame_num)):
            self.model.module.simulator.set_controller_target(frame_idx)

            # Record data
            vertices_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[-1].wp_x).clone(), wp.to_torch(simulator.wp_states[-1].wp_control_x).clone()]))
            velocities_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[-1].wp_v).clone(), wp.to_torch(simulator.wp_states[-1].wp_control_v).clone()]))

            if self.model.module.simulator.object_collision_flag:
                self.model.module.simulator.update_collision_graph()

            # 计算 Loss
            if cfg.use_graph:
                wp.capture_launch(self.model.module.simulator.graph)
            else:
                if cfg.data_type == "real":
                    with self.model.module.simulator.tape:
                        self.model.module.simulator.step()
                        self.model.module.simulator.calculate_loss()
                    self.model.module.simulator.tape.backward(self.model.module.simulator.loss)
                else:
                    with self.model.module.simulator.tape:
                        self.model.module.simulator.step()
                        self.model.module.simulator.calculate_simple_loss()
                    self.model.module.simulator.tape.backward(self.model.module.simulator.loss)
            # self.model.module.simulator.export_forces_to_txt(frame_idx=frame_idx, filename="simulation_forces.txt")
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
        train_node_pos = torch.FloatTensor(self.mdata.mesh_pos)
        train_node_type = torch.FloatTensor(self.mdata.node_type[0])
        spring_rest_length = torch.FloatTensor(self.mdata.spring_rest_length)[:, None]
   
        # normalize train node pos into [-1,1]
        pos_min = train_node_pos.min(dim=0, keepdim=True)[0]
        pos_max = train_node_pos.max(dim=0, keepdim=True)[0]
        normalized_train_node_pos = 2 * (train_node_pos - pos_min) / (pos_max - pos_min) - 1

        # apply NeRF-style positional encoding to node positions
        pos_encoded = self.positional_encoding(normalized_train_node_pos, num_freq_bands=10)

        node_in_feature = torch.cat([
            normalized_train_node_pos, pos_encoded, train_node_type
        ], dim = 1)

        # normalize spring_rest_length into [-1,1]
        spring_min = spring_rest_length.min(dim=0, keepdim=True)[0]
        spring_max = spring_rest_length.max(dim=0, keepdim=True)[0]
        normalized_spring_rest_length = 2 * (spring_rest_length - spring_min) / (spring_max - spring_min) - 1

        # apply NeRF-style positional encoding to spring lengths
        spring_encoded = self.positional_encoding(normalized_spring_rest_length, num_freq_bands=10)

        edge_mech_in_feature = torch.cat(
            [spring_rest_length, spring_encoded], dim = 1
        )
        
        m_ids, m_gs, m_gs_parent, node_in_feature, edge_mech_in_feature =\
              self._preproc_multi_infos(self.mdata, node_in_feature, edge_mech_in_feature)

        if mode != 'train':
            self.model.eval()

        attn_mask = self.mdata.attn_mask
            
        # 运行仿真
        if mode == 'train':
            for i in range(10):
                # 获取平均梯度
                self.optimizer.zero_grad()

                # Update simulator with predicted mechanical properties
                # Note: You may need to adjust this based on actual mech_info structure
                st = time()

                new_spring_Y, drag_damping_out, dashpot_damping_out =\
                      self.model(m_ids, m_gs, m_gs_parent, node_in_feature, edge_mech_in_feature, attn_mask)
                log_new_spring_Y = torch.log(new_spring_Y) #+ 1e-8)  # 添加小值避免 log(0)

                # [关键] 映射到 Warp 时，必须设置 requires_grad=True
                # 这样 Warp 才会为这些数组分配梯度缓冲区，供 Tape 使用

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

                # # Convert predicted collision values to warp
                # wp_predicted_collision_elas = wp.from_torch(
                #     collision_elas_out.contiguous(), dtype=wp.float32, requires_grad=True
                # )
                # wp_predicted_collision_fric = wp.from_torch(
                #     collision_fric_out.contiguous(), dtype=wp.float32, requires_grad=True
                # )
                # wp_predicted_collision_object_elas = wp.from_torch(
                #     collision_object_elas_out.contiguous(), dtype=wp.float32, requires_grad=True
                # )
                # wp_predicted_collision_object_fric = wp.from_torch(
                #     collision_object_fric_out.contiguous(), dtype=wp.float32, requires_grad=True
                # )

                # Set predicted spring_Y and damping to simulator
                self.model.module.simulator.set_spring_Y(wp_predicted_spring_Y)
                self.model.module.simulator.set_drag_damping(wp_predicted_drag_damping)
                self.model.module.simulator.set_dashpot_damping(wp_predicted_dashpot_damping)
                # self.model.module.simulator.set_collision_elas(wp_predicted_collision_elas)
                # self.model.module.simulator.set_collision_fric(wp_predicted_collision_fric)
                # self.model.module.simulator.set_collision_object_elas(wp_predicted_collision_object_elas)
                # self.model.module.simulator.set_collision_object_fric(wp_predicted_collision_object_fric)

                print("Forward time is : {}".format(time() - st))
                st = time()
                
                pos, vel, _, _, _, _, _, loss_val, chamfer_loss_val, track_loss_val, grad_avg_dict, valid_frames = \
                    self.generate_data_point_sequence(
                    update_frame_num=cfg.train_frame,
                    enable_backward=True,
                )
                
                self.total_update += 1

                # Print spring_Y average
                print(f"\n{'='*60}")
                print(f"Spring Y iteration {i+1}/10")
                print(f"spring_Y grad Average: {grad_avg_dict['log_spring_Y'].mean().item()}")
                print(f"Spring_Y Average: {log_new_spring_Y.mean().item():.6f}")
                print(f"drag_damping grad Average: {grad_avg_dict['drag_damping'].mean().item()}")
                print(f"Drag_Damping: {drag_damping_out.item():.6f}")
                print(f"dashpot_damping grad Average: {grad_avg_dict['dashpot_damping'].mean().item()}")
                print(f"Dashpot_Damping: {dashpot_damping_out.item():.6f}")
                # print(f"collision_elas grad Average: {grad_avg_dict['collision_elas'].mean().item()}")
                # print(f"Collision_Elas: {collision_elas_out.item():.6f}")
                # print(f"collision_fric grad Average: {grad_avg_dict['collision_fric'].mean().item()}")
                # print(f"Collision_Fric: {collision_fric_out.item():.6f}")
                # print(f"collision_object_elas grad Average: {grad_avg_dict['collision_object_elas'].mean().item()}")
                # print(f"Collision_Object_Elas: {collision_object_elas_out.item():.6f}")
                # print(f"collision_object_fric grad Average: {grad_avg_dict['collision_object_fric'].mean().item()}")
                # print(f"Collision_Object_Fric: {collision_object_fric_out.item():.6f}")
                print(f"Loss sum: {np.sum(loss_val):.10f}")
                print(f"Chamfer loss sum: {np.sum(chamfer_loss_val):.10f}")
                print(f"Track loss sum: {np.sum(track_loss_val):.10f}")
                print(f"Valid frames: {valid_frames}")
                print(f"{'='*60}\n")

                # import pdb; pdb.set_trace()
                self.writer.add_scalar('Spring_Y_update/overall_Loss', np.sum(loss_val), self.total_update)
                self.writer.add_scalar('Spring_Y_update/chamfer_Loss', np.sum(chamfer_loss_val), self.total_update)
                self.writer.add_scalar('Spring_Y_update/track_Loss', np.sum(track_loss_val), self.total_update)
                self.writer.add_scalar('Spring_Y_update/Spring_Y_Average', log_new_spring_Y.mean().item(), self.total_update)
                self.writer.add_scalar('Spring_Y_update/Drag_Damping', drag_damping_out.item(), self.total_update)
                self.writer.add_scalar('Spring_Y_update/Dashpot_Damping', dashpot_damping_out.item(), self.total_update)
                # self.writer.add_scalar('Spring_Y_update/Collision_Elas', collision_elas_out.item(), self.total_update)
                # self.writer.add_scalar('Spring_Y_update/Collision_Fric', collision_fric_out.item(), self.total_update)
                # self.writer.add_scalar('Spring_Y_update/Collision_Object_Elas', collision_object_elas_out.item(), self.total_update)
                # self.writer.add_scalar('Spring_Y_update/Collision_Object_Fric', collision_object_fric_out.item(), self.total_update)

                # -------------------------------------------------------
                # [核心修复] 使用平均梯度更新模型
                # -------------------------------------------------------

                # 将平均梯度手动注入 PyTorch 计算图
                # 2. 新增：将 edge_reg_loss 和 damping 参数加入反向传播列表
                tensors_to_backward = [log_new_spring_Y,
                                       drag_damping_out,
                                       dashpot_damping_out,]
                                    #    collision_elas_out,
                                    #    collision_fric_out,
                                    #    collision_object_elas_out,
                                    #    collision_object_fric_out]
                # edge_reg_loss 是标量 Loss，它在这个层级的起始梯度就是 1.0
                grad_tensors_to_backward = [grad_avg_dict['log_spring_Y'],
                                            grad_avg_dict['drag_damping'],
                                            grad_avg_dict['dashpot_damping'],]
                                            # grad_avg_dict['collision_elas'],
                                            # grad_avg_dict['collision_fric'],
                                            # grad_avg_dict['collision_object_elas'],
                                            # grad_avg_dict['collision_object_fric']]

                # 一并执行反向传播
                torch.autograd.backward(
                    tensors=tensors_to_backward,
                    grad_tensors=grad_tensors_to_backward
                )

                # PyTorch 优化器更新神经网络参数
                self.optimizer.step()
                print(f"Model updated with average gradient per frame")

                # Check if this is the best iteration and save immediately
                current_loss = np.sum(loss_val)
                if  current_loss < self.global_best_loss:
                    prev_best_loss = self.global_best_loss
                    self.global_best_loss = current_loss
                    self.global_best_epoch = epoch
                    self.global_best_iteration = i

                    # Save best model checkpoint
                    best_ckpt_path = os.path.join(self.args.dump_dir, 'ckpts', f'best_iter_epoch{epoch}_iter{i}_{self.checkpt_name}')
                    torch.save(self.model.module.state_dict(), best_ckpt_path)

                    # Save best mech_info
                    best_mech_info = {
                        'log_spring_Y': log_new_spring_Y.detach(),
                        'drag_damping': drag_damping_out.detach().clone(),
                        'dashpot_damping': dashpot_damping_out.detach().clone(),
                        # 'collision_elas': collision_elas_out.detach().clone(),
                        # 'collision_fric': collision_fric_out.detach().clone(),
                        # 'collision_object_elas': collision_object_elas_out.detach().clone(),
                        # 'collision_object_fric': collision_object_fric_out.detach().clone()
                    }
                    self.global_best_mech_info = best_mech_info.copy()
                    best_mech_info_path = os.path.join(self.args.dump_dir, 'spring_mech_info', f'best_iter_epoch{epoch}_iter{i}_{self.checkpt_name}')
                    torch.save(best_mech_info, best_mech_info_path)

                    # Save best trajectory
                    best_traj_path = os.path.join(self.args.dump_dir, 'trajectories', 'best_trajectory.pkl')
                    logger.info(f"New best loss {current_loss:.10f} at epoch {epoch}, iteration {i+1}/10. Saving checkpoint and trajectory...")
                    self.save_traj(mech_info={k: v.clone() for k, v in best_mech_info.items()}, save_path=best_traj_path, compute_loss=True)

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

            # Update collide parameters first (10 iterations)
            if cfg.collision_learn:
                print(f"\n{'='*60}")
                print("Starting collide parameters optimization...")
                print(f"{'='*60}\n")

                for i in range(5):
                    loss_val = []

                    # Reset position and velocity
                    simulator = self.model.module.simulator
                    simulator.set_init_state(
                        simulator.wp_init_vertices,
                        simulator.wp_init_velocities
                    )

                    # Run simulation for all frames
                    for j in tqdm(range(1, cfg.train_frame), desc=f"Collide iter {i+1}/10"):
                        simulator.set_controller_target(j)

                        if simulator.object_collision_flag:
                            simulator.update_collision_graph()

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

                        # Update collide parameters
                        self.collide_optimizer.step()

                        # Accumulate loss
                        loss = wp.to_torch(simulator.loss, requires_grad=False)
                        loss_val.append(loss.item())

                        if cfg.use_graph:
                            simulator.tape.zero()
                        else:
                            simulator.tape.reset()

                        simulator.clear_loss()

                        # Set initial state for next step
                        simulator.set_init_state(
                            simulator.wp_states[-1].wp_x,
                            simulator.wp_states[-1].wp_v
                        )

                    # Print collide optimization results
                    print(f"\n{'='*60}")
                    print(f"Collide iteration {i+1}/10 - Average Loss: {np.mean(loss_val):.10f}")
                    print(f"Differentiable parameters:")
                    print(f"  collide_elas: {self.torch_collide_elas.mean().item():.6f}")
                    print(f"  collide_fric: {self.torch_collide_fric.mean().item():.6f}")
                    print(f"  collide_object_elas: {self.torch_collide_object_elas.mean().item():.6f}")
                    print(f"  collide_object_fric: {self.torch_collide_object_fric.mean().item():.6f}")
                    # if self.torch_collision_dist is not None:
                    #     print(f"  collision_dist: {self.torch_collision_dist.item():.6f}")
                    print(f"{'='*60}\n")

            # Update spring Y parameters (10 iterations)
            print(f"\n{'='*60}")
            print("Starting spring Y parameters optimization...")
            print(f"{'='*60}\n")

        else:
            # 验证模式不需要 tape
            _, _, _, _, _, _, _, loss, chamfer_loss, track_loss = self.generate_data_point_sequence(
                update_frame_num=self.mdata.train_frame,
                enable_backward=False,
                set_object_point=False
            )
            loss_val = loss if isinstance(loss, float) else loss # 处理验证集返回值

        print("Backward time : {}".format(time() - st))
        # stats
        mean_loss = np.sum(loss_val) if isinstance(loss_val, list) else loss_val


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

        final_mech_info = {
            'log_spring_Y': log_new_spring_Y.detach(),
            'drag_damping': drag_damping_out.detach(),
            'dashpot_damping': dashpot_damping_out.detach(),
            # 'collision_elas': collision_elas_out.detach(),
            # 'collision_fric': collision_fric_out.detach(),
            # 'collision_object_elas': collision_object_elas_out.detach(),
            # 'collision_object_fric': collision_object_fric_out.detach()
        }

        return mean_loss, final_mech_info

    
    def train(self):

        # Save initial trajectory before training
        initial_traj_save_path = os.path.join(self.args.dump_dir, 'trajectories', 'initial_trajectory.pkl')
        logger.info("Saving initial trajectory before training with loss computation")
        frame_losses, chamfer_losses, track_losses, save_data\
            = self.save_traj(mech_info=None, save_path=initial_traj_save_path, compute_loss=True)

        # Initialize global_best_loss with the initial trajectory loss if not set
        initial_loss = np.sum(frame_losses[1:cfg.train_frame])
        if self.global_best_loss == float('inf'):
            self.global_best_loss = initial_loss
            logger.info(f"Initialized global_best_loss with initial trajectory loss: {initial_loss:.10f}")

        for epoch in self.pbar:
            mean_loss_train, mech_info = self.run_epoch(epoch, 
                                                        'train')
            with torch.autograd.no_grad():
                # train loss record
                if dist.get_rank() == 0:
                    self.pbar.set_description(f"Epoch: {epoch}, Training RMSE: {math.sqrt(mean_loss_train)}")
                    self.writer.add_scalar('RMSE/train', math.sqrt(mean_loss_train), epoch)
                    self.writer.add_scalar('lr/train', self.optimizer.param_groups[0]['lr'], epoch)
                    
                    # record mean value of mech_info for monitoring
                    self.writer.add_scalar('mech_info/log_spring_Y', mech_info['log_spring_Y'].mean().item(), epoch)
                    self.writer.add_scalar('mech_info/drag_damping', mech_info['drag_damping'].item(), epoch)
                    self.writer.add_scalar('mech_info/dashpot_damping', mech_info['dashpot_damping'].item(), epoch)

                    # dump ckpt
                    ckpt_path = os.path.join(self.args.dump_dir, 'ckpts', str(epoch) + "_" + self.checkpt_name)
                    torch.save(self.model.module.state_dict(), ckpt_path)
                    
                    # dump spring emch info
                    mech_info_path = os.path.join(self.args.dump_dir, 'spring_mech_info', str(epoch) + "_" + self.checkpt_name)
                    torch.save({k: v.clone() for k, v in mech_info.items()}, mech_info_path)

                    # save trajectory with current mech_info
                    traj_save_path = os.path.join(self.args.dump_dir, 'trajectories', f'epoch_{epoch}_trajectory.pkl')
                    logger.info(f"Saving trajectory for epoch {epoch}")

                    self.save_traj(mech_info={k: v.clone() for k, v in mech_info.items()},
                                         save_path=traj_save_path)

        # dump best spring mech info estimation
        if self.global_best_epoch is not None:
            if self.global_best_iteration is not None:
                mech_info_path = os.path.join(self.args.dump_dir, 'spring_mech_info', f"final_best_epoch{self.global_best_epoch}_iter{self.global_best_iteration}.pth")
            else:
                mech_info_path = os.path.join(self.args.dump_dir, 'spring_mech_info', f"final_best_epoch{self.global_best_epoch}.pth")
            torch.save(self.global_best_mech_info, mech_info_path)
            logger.info(f"Training completed. Best loss: {self.global_best_loss:.10f} at epoch {self.global_best_epoch}" +
                       (f", iteration {self.global_best_iteration}" if self.global_best_iteration is not None else ""))

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
