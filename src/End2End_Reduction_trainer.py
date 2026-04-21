import torch
from qqtt.utils import logger, cfg
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import models as models
from End2End_EvoSpring_Reduction import End2EndReduction_EvoSpring
from End2End_Reduction_Dataset import End2EndReductionDataset
import math
import numpy as np
import os
from Reducer import FastAdaptiveNetworkReducer
import warp as wp
from time import time
import pickle
import logging
from datetime import datetime

# 禁用所有 warp 相关的 logger
logging.getLogger("warp").setLevel(logging.ERROR)


class E2EReductionTrainer:
    def __init__(self, args, device="cuda:0",):
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
                                          lr=self.args.lr * min(np.sqrt(cfg.train_frame), 2), 
                                          betas=(0.9, 0.99))

        self.epochs_per_stage = 20
        self.iter_per_epoch = 10
        
        # 分阶段训练相关变量
        self.num_stages = self.args.multi_mesh_layer  # 阶段数等于层数
        self.current_stage = 0  # 当前训练阶段
        
        # 根据程序运行时间创建保存文件夹
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_base_dir = os.path.join(self.args.dump_dir, self.run_timestamp)
        os.makedirs(self.save_base_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {self.save_base_dir}")
        
        # 在时间戳文件夹下创建所有需要的子目录
        for subdir in ['ckpts', 'log', 'test_RMSE', 'spring_mech_info', 'trajectories', 'exported_meshes', 'visualization']:
            dir = os.path.join(self.save_base_dir, subdir)
            os.makedirs(dir, exist_ok=True)
        
        # if dist.get_rank() == 0:
            # SummaryWriter 保存到时间戳文件夹下的 log 目录
        self.writer = SummaryWriter(os.path.join(self.save_base_dir, 'log'))

        self.total_update = 0
        self.reducer = FastAdaptiveNetworkReducer(num_modes=500)

        # 每个阶段独立跟踪最佳结果
        self.stage_best_losses = []  # 每个阶段的最佳 loss 列表
        self.stage_best_mech_info = []  # 每个阶段的最佳 mech_info 列表
        self.stage_best_epochs = []  # 每个阶段的最佳 epoch
        self.stage_best_iterations = []  # 每个阶段的最佳 iteration
        
        # 初始化阶段最佳结果
        for _ in range(self.num_stages):
            self.stage_best_losses.append(float('inf'))
            self.stage_best_mech_info.append(None)
            self.stage_best_epochs.append(None)
            self.stage_best_iterations.append(None)
        
        # 全局最佳结果（用于下采样时参考）
        self.global_best_loss = float('inf')
        self.global_best_mech_info = None
        self.global_best_epoch = None
        self.global_best_iteration = None

        self.mlvl_simulators = []
        self.mlvl_collide_optimizer = []

    def create_simulator(self, stage):
        # create simulator here
        vertice = self.m_vertices[stage]
        masses = self.m_masses[stage]
        
        spring_graph = self.m_gs[stage].int().T.contiguous()
        num_edge = len(spring_graph)
        spring_graph = spring_graph[:num_edge // 2]
        node_type = self.m_node_type[stage]
        
        rest_lengths = torch.norm((vertice[spring_graph[:, 0]]
                                   -vertice[spring_graph[:, 1]]), dim = 1)
  
        # downsample for gt points
        gt_object_points = self.m_gt_object_points[stage]
        gt_object_visibilities = self.m_gt_object_visibilities[stage]
        gt_object_motions_valid = self.m_gt_object_motions_valid[stage]

        # create simulator
        simulator = self.mdata.create_spring_mass_sim(vertice, 
                                                    spring_graph, 
                                                    rest_lengths, 
                                                    masses,
                                                    node_type,
                                                    gt_object_points, 
                                                    gt_object_visibilities, 
                                                    gt_object_motions_valid)

        # 初始化 Warp 优化器（仅在训练模式且 collision_learn 开启时）
        if cfg.collision_learn:
            # 确保 requires_grad=True
            # 碰撞参数 (Collision Parameters)
            torch_collide_elas = wp.to_torch(simulator.wp_collide_elas)
            torch_collide_fric = wp.to_torch(simulator.wp_collide_fric)
            torch_collide_object_elas = wp.to_torch(simulator.wp_collide_object_elas)
            torch_collide_object_fric = wp.to_torch(simulator.wp_collide_object_fric)

            # 将所有可微分参数添加到列表
            warp_params_list = [
                torch_collide_elas,
                torch_collide_fric,
                torch_collide_object_elas,
                torch_collide_object_fric,
            ]
        else:
            warp_params_list = []

        collide_optimizer = torch.optim.Adam(warp_params_list,
                                          lr=self.args.lr,
                                          betas=(0.9, 0.99))
        self.mlvl_simulators.append(simulator)
        self.mlvl_collide_optimizer.append(collide_optimizer)

    def export_multi_level_obj(self, save_dir="exported_meshes"):
        """
        将每一层降维后的图结构（顶点和边）导出为标准的 .obj 文件。
        顶点坐标直接保留代表节点的原始物理坐标（不进行质心平均）。
        
        修改说明：导出到时间戳文件夹下的 exported_meshes 目录
        """
        logger.info(f"Exporting multi-level topology to {save_dir}...")
        
        # 创建导出目录 - 保存到时间戳文件夹下
        export_path = os.path.join(self.save_base_dir, save_dir)
        os.makedirs(export_path, exist_ok=True)
        
        # ==========================================
        # 1. 准备全局原始坐标
        # ==========================================
        if isinstance(self.mdata.init_vertices, torch.Tensor):
            global_vertices = self.mdata.init_vertices.detach().cpu().numpy()
        else:
            global_vertices = self.mdata.init_vertices
            
        # ==========================================
        # 2. 导出第 0 层 (原始高分辨率结构)
        # ==========================================
        current_edges = self.m_gs[0].detach().cpu().numpy()  # [2, num_edges]
        
        self._write_obj(
            filename=os.path.join(export_path, f"layer_0_nodes_{global_vertices.shape[0]}.obj"),
            vertices=global_vertices,
            edges=current_edges
        )
        
        # ==========================================
        # 3. 逐层提取代表节点并导出
        # ==========================================
        # self.m_ids 中存储了每一层的节点在全局图中的原始索引
        # 注意：m_ids[0] 可能是全集，m_ids[1] 是第一次降采样的结果
        
        for layer_idx in range(1, len(self.m_ids)):
            # 获取当前层保留的全局节点索引 (例如: [3, 5, 9, 12, ...])
            current_node_ids = self.m_ids[layer_idx]
            
            # 【核心修改】：直接从全局坐标中，抽出这些保留节点的位置
            # 因为 new_ids 记录的就是被选为代表的那个原始节点的 ID
            current_points = global_vertices[current_node_ids]
            
            # 获取当前层的边拓扑关系 (这些边使用的是 0 到 N_new-1 的局部索引)
            current_edges = self.m_gs[layer_idx].detach().cpu().numpy()

            # 导出当前层
            filename = os.path.join(export_path, f"layer_{layer_idx}_nodes_{current_points.shape[0]}.obj")
            self._write_obj(filename, current_points, current_edges)
            
        logger.info(f"Successfully exported {len(self.m_ids)} layers of .obj files to {export_path}")

    def _write_obj(self, filename, vertices, edges):
        """
        内部辅助函数：将顶点和边写入 OBJ 文件格式
        """
        with open(filename, 'w') as f:
            f.write(f"# Auto-generated hierarchical point cloud\n")
            f.write(f"# Vertices: {vertices.shape[0]}, Edges: {edges.shape[1]}\n\n")
            
            # 1. 写入顶点 (Vertices)
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
            f.write("\n")
            
            # 2. 写入边/线段 (Lines)
            if edges.shape[1] > 0:
                edges_sorted = np.sort(edges, axis=0)
                edges_unique = np.unique(edges_sorted, axis=1)
                
                for col in range(edges_unique.shape[1]):
                    # OBJ 的索引从 1 开始
                    u = edges_unique[0, col] + 1  
                    v = edges_unique[1, col] + 1
                    f.write(f"l {u} {v}\n")
        
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

        self.model.to(self.args.local_rank)
        # self.model = nn.parallel.DistributedDataParallel(
        #     self.model.cuda(self.args.local_rank),
        #     device_ids=[self.args.local_rank],
        #     output_device=self.args.local_rank,
        #     find_unused_parameters=True  # [关键修复] 允许部分参数不参与计算
        # )

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
        self, mlvl_mech_info=None, save_path=None, compute_loss=False, gt_indices=None, stage_idx=None,
        mlvl_masses=None, mlvl_edges=None
    ):
        """
        保存轨迹数据，可选保存真值下采样索引（用于多阶段评估）
        
        Args:
            mlvl_mech_info: 力学参数列表
            save_path: 保存路径
            compute_loss: 是否计算 loss
            gt_indices: 真值下采样索引列表（每个阶段对应一个索引列表）
            stage_idx: 当前阶段索引（用于从 gt_indices 中提取对应阶段的索引）
            mlvl_masses: 每个 level 的质点质量列表（可选）
            mlvl_edges: 每个 level 的连接关系/边拓扑列表（可选）
        """
        # ====================================================================
        # [核心修改]: 保存所有 level 的轨迹结果
        # ====================================================================
        logger.info("Initializing all level simulators for trajectory rollout...")
        
        # 保存所有 level 的结果
        all_level_vertices = []
        all_level_velocities = []
        
        for level_idx, simulator in enumerate(self.mlvl_simulators):
            logger.info(f"Processing Level {level_idx}...")
            
            # ====================================================================
            # 2. 将预测的力学参数 (mech_info) 赋予模拟器
            # ====================================================================
            if mlvl_mech_info is not None and len(mlvl_mech_info) > level_idx:
                logger.info(f"Setting predicted mechanical properties to simulator (Level {level_idx})")
                level_mech_info = mlvl_mech_info[level_idx]

                if isinstance(level_mech_info, dict):
                    if 'log_spring_Y' in level_mech_info:
                        wp_predicted_spring_Y = wp.from_torch(level_mech_info['log_spring_Y'].contiguous(), dtype=wp.float32, requires_grad=False)
                        simulator.set_spring_Y(wp_predicted_spring_Y)

                    if 'drag_damping' in level_mech_info:
                        wp_predicted_drag_damping = wp.from_torch(level_mech_info['drag_damping'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                        simulator.set_drag_damping(wp_predicted_drag_damping)

                    if 'dashpot_damping' in level_mech_info:
                        wp_predicted_dashpot_damping = wp.from_torch(level_mech_info['dashpot_damping'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                        simulator.set_dashpot_damping(wp_predicted_dashpot_damping)

                    if 'collision_elas' in level_mech_info:
                        wp_predicted_collision_elas = wp.from_torch(level_mech_info['collision_elas'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                        simulator.set_collision_elas(wp_predicted_collision_elas)

                    if 'collision_fric' in level_mech_info:
                        wp_predicted_collision_fric = wp.from_torch(level_mech_info['collision_fric'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                        simulator.set_collision_fric(wp_predicted_collision_fric)

                    if 'collision_object_elas' in level_mech_info:
                        wp_predicted_collision_object_elas = wp.from_torch(level_mech_info['collision_object_elas'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                        simulator.set_collision_object_elas(wp_predicted_collision_object_elas)

                    if 'collision_object_fric' in level_mech_info:
                        wp_predicted_collision_object_fric = wp.from_torch(level_mech_info['collision_object_fric'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
                        simulator.set_collision_object_fric(wp_predicted_collision_object_fric)
                else:
                    logger.warning(f"level_mech_info for level {level_idx} has unexpected format, skipping property update")
            else:
                logger.warning(f"No mech_info available for level {level_idx}")

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

            for i in tqdm(range(1, frame_len), desc=f"Saving trajectory Level {level_idx}", leave=False):
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
            
            all_level_vertices.append(vertices.cpu().numpy())
            all_level_velocities.append(velocities.cpu().numpy())
        
        # ====================================================================
        # 4. 打包并保存结果 (.pkl)
        # ====================================================================
        logger.info(f"Save the trajectory to {save_path}")

        # 构建 save_data，包含所有 level 的轨迹和物理属性
        save_data = {
            'vertices': all_level_vertices,       # 所有 level 的轨迹列表
            'velocities': all_level_velocities,   # 所有 level 的速度列表
        }
        
        # 保存每个 level 的节点类型信息
        save_data['node_types'] = []
        for level_idx in range(len(self.mlvl_simulators)):
            if hasattr(self, 'm_node_type') and level_idx < len(self.m_node_type):
                level_node_type = self.m_node_type[level_idx]
                if isinstance(level_node_type, torch.Tensor):
                    save_data['node_types'].append(level_node_type.cpu().numpy())
                else:
                    save_data['node_types'].append(level_node_type)
            else:
                # 如果没有 node_type 信息，使用默认值（全为 0）
                num_nodes = all_level_vertices[level_idx].shape[1]
                save_data['node_types'].append(np.zeros(num_nodes, dtype=np.int32))
        logger.info(f"Saved node_types for {len(save_data['node_types'])} levels")
        
        # 保存每个 level 的质点质量信息
        if mlvl_masses is not None:
            save_data['masses'] = []
            for level_idx, level_masses in enumerate(mlvl_masses):
                if isinstance(level_masses, torch.Tensor):
                    save_data['masses'].append(level_masses.detach().cpu().numpy())
                else:
                    save_data['masses'].append(level_masses)
            logger.info(f"Saved masses for {len(save_data['masses'])} levels")
        
        # 保存每个 level 的连接关系（边拓扑）
        
        if mlvl_edges is not None:
            save_data['edges'] = []
            for level_idx, level_edges in enumerate(mlvl_edges):
                if isinstance(level_edges, torch.Tensor):
                    save_data['edges'].append(level_edges.cpu().numpy())
                else:
                    save_data['edges'].append(level_edges)
            logger.info(f"Saved edges for {len(save_data['edges'])} levels")
        
        # 如果提供了 gt_indices 和 stage_idx，保存对应阶段的真值索引
        if gt_indices is not None and stage_idx is not None:
            if stage_idx < len(gt_indices):
                save_data['gt_indices'] = gt_indices[stage_idx]
                logger.info(f"Saved GT indices for stage {stage_idx}: {len(save_data['gt_indices'])} nodes")
            else:
                logger.warning(f"stage_idx {stage_idx} out of range for gt_indices (len={len(gt_indices)})")
        
        # 也保存当前的 stage_idx 以便评估时识别
        if stage_idx is not None:
            save_data['stage_idx'] = stage_idx

        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Trajectory saved successfully with {len(all_level_vertices)} levels")
        if compute_loss:
            return frame_losses, chamfer_losses, track_losses, save_data

    def _preproc_multi_infos(self, mdata):
        # no contact, then share the graph between batches
        # only keep the first layer
        self.m_ids = [[i for i in range(len(mdata.init_vertices))]]
        m_gs_list = [np.concatenate([mdata.cells, mdata.cells[[1,0]]], axis = 1)]
        self.m_gs = [torch.tensor(g, dtype=torch.long).to(cfg.device) for g in m_gs_list]

        # 投影压缩矩阵
        self.m_proj = []
        self.m_vertices = [torch.tensor(mdata.init_vertices, dtype=torch.float32, device=cfg.device)]
        self.m_masses = [mdata.init_masses]
        self.m_node_type = [torch.tensor(mdata.node_type[0, :, 0], dtype=torch.long, device=cfg.device)]

        self.m_gt_object_points = [mdata.object_points.clone().to(cfg.device)]
        self.m_gt_object_visibilities = [mdata.object_visibilities.clone().to(cfg.device)]
        self.m_gt_object_motions_valid = [mdata.object_motions_valid.clone().to(cfg.device)]


    
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
        
        for frame_idx in tqdm(range(1, update_frame_num), leave=False):
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

        # 转换输出张量
        vertices_tensor = torch.stack(vertices_sequence, dim=0)
        velocities_tensor = torch.stack(velocities_sequence, dim=0)
        node_mass = torch.cat([wp.to_torch(simulator.wp_masses).clone(), torch.zeros(simulator.num_control_points, device=vertices_tensor.device)])
        spring_Y = wp.to_torch(simulator.wp_spring_Y).clone()

        spring_rest_length = wp.to_torch(simulator.wp_rest_lengths).clone()
        spring_dashpot_damping = simulator.wp_dashpot_damping
        drag_damping = simulator.wp_drag_damping

        if enable_backward:
            # 计算所有参数的平均梯度
            grad_avg_dict = {}

            for param_name, grad_list in grad_sequences.items():
                if len(grad_list) == 0:
                    continue
                else:
                    grad_avg_dict[param_name] = torch.sum(torch.stack(grad_list, dim=0), dim=0)

            # 返回平均梯度字典
            return vertices_tensor, velocities_tensor, node_mass, spring_Y, spring_rest_length, spring_dashpot_damping, drag_damping, losses, chamfer_losses, track_losses, grad_avg_dict, valid_frames
        else:
            return vertices_tensor, velocities_tensor, node_mass, spring_Y, spring_rest_length, spring_dashpot_damping, drag_damping, losses, chamfer_losses, track_losses


    # ====================================================================
    # Train 函数：分阶段训练
    # 每个阶段只监督一层的 simulator：
    # - 第 0 阶段：只监督最上层（未下采样）的输出
    # - 第 1 阶段：只监督下采样一次后的结果
    # - 以此类推
    # 每个阶段训练 epochs_per_stage 轮
    # 每层的最佳结果独立保存，不受后续阶段影响
    # ====================================================================
    def train(self):
        # Initialize initial graph structure info before loop
        self._preproc_multi_infos(self.mdata)
        
        # 创建总体进度条
        total_epochs = self.num_stages * self.epochs_per_stage
        self.pbar = tqdm(total=total_epochs, unit="epoch")
        
        logger.info(f"Starting multi-stage training with {self.num_stages} stages, {self.epochs_per_stage} epochs per stage")
        
        # 分阶段训练
        for stage in range(self.num_stages):
            # DEBUG(CXY)
            if stage >= 1:
                break

            self.current_stage = stage
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting Stage {stage} (supervising level {stage})")
            logger.info(f"{'='*60}\n")
            
            # 注意：downsample 是在模型 forward 中通过 pooling 操作实现的
            # 每个 stage 对应模型输出的一层，不需要手动执行 downsample
            # 模型 forward 会返回所有层的输出和 pooling_losses
            
            # 训练当前阶段的所有 epoch
            for epoch in range(self.epochs_per_stage):
                global_epoch = stage * self.epochs_per_stage + epoch
                logger.info(f"--- Stage {stage}, Epoch {epoch+1}/{self.epochs_per_stage} (Global Epoch {global_epoch+1}/{total_epochs}) ---")
                
                # 运行 epoch，只监督当前层
                self.run_epoch(epoch, mode='train', stage=stage)
                
                # 更新进度条
                self.pbar.update(1)
            
            # # 保存当前阶段的最佳结果
            self._save_stage_result(stage)
            # 合并所有阶段的最佳结果为 global_best
            self._merge_stage_results()
            logger.info(f"Stage {stage} completed. Best loss: {self.stage_best_losses[stage]:.6f}")
        
        # 所有阶段训练完成后，导出和可视化
        try:
            self.export_multi_level_obj()
        except Exception as e:
            logger.error(f"Failed to export OBJ models: {e}")
        
        self.pbar.close()
        logger.info("Multi-stage training completed!")
    
    def _create_stage_simulator(self, stage):
        """
        为当前阶段创建 simulator（累积创建所有层，不清空之前的 simulator）
        使用逐层下采样的方式获取 GT 数据
        """
        logger.info(f"Creating simulator for stage {stage}...")
        
        # [修复] 不再清空之前的 simulator，而是累积保存所有层的 simulator
        # 只在 stage 0 时清空，确保每个 stage 的 simulator 都被保留
        if stage == 0:
            self.mlvl_simulators = []
            self.mlvl_collide_optimizer = []
        else:
            # 对于 stage > 0，只清空当前 stage 及之后的部分（如果有的话）
            # 保留之前 stage 的 simulator
            if len(self.mlvl_simulators) > stage:
                self.mlvl_simulators = self.mlvl_simulators[:stage]
                self.mlvl_collide_optimizer = self.mlvl_collide_optimizer[:stage]
        
        # 保存原始的 GT 数据（第 0 层）
        original_gt_points = self.mdata.object_points  # [T, N_original, ...]
        original_gt_vis = self.mdata.object_visibilities
        original_gt_valid = self.mdata.object_motions_valid
        
        # 逐层下采样 GT 数据
        # 注意：GT 数据只对应 object points (node_type=0)
        # 所以需要从 current_ids 中筛选出 object points 的索引
        gt_object_points_list = [original_gt_points]
        gt_object_visibilities_list = [original_gt_vis]
        gt_object_motions_valid_list = [original_gt_valid]
        
        # 根据 down_ids 逐层下采样 GT 数据
        for lvl in range(1, stage + 1):
            # down_ids[lvl] 存储的是当前层节点在上一层的索引
            prev_gt_points = gt_object_points_list[-1]
            prev_gt_vis = gt_object_visibilities_list[-1]
            prev_gt_valid = gt_object_motions_valid_list[-1]
            
            # 获取当前层的 node_type，筛选出 object points (node_type=0)
            current_node_type = self.m_node_type[lvl]
            object_mask = (current_node_type == 0)
            object_indices = torch.where(object_mask)[0]
            
            # 从 current_ids 中只提取 object points 对应的索引
            # current_ids 是 list，需要转换为 tensor 才能进行索引
            current_ids = torch.tensor(self.m_ids[lvl], dtype=torch.long, device=cfg.device)
            object_ids = current_ids[object_indices]
            
            # 从上一层的 GT 数据中提取当前层的 GT 数据（只针对 object points）
            current_gt_points = prev_gt_points[:, object_ids]
            current_gt_vis = prev_gt_vis[:, object_ids]
            current_gt_valid = prev_gt_valid[:, object_ids]
            
            gt_object_points_list.append(current_gt_points)
            gt_object_visibilities_list.append(current_gt_vis)
            gt_object_motions_valid_list.append(current_gt_valid)
            logger.info(f"Downsampled GT data for level {lvl}: {prev_gt_points.shape} -> {current_gt_points.shape}")
        
        # [关键修改] 将逐层下采样的 GT 数据保存回实例变量列表
        # 这样在 run_epoch 中保存最佳结果时可以正确获取下采样后的 GT 数据
        self.m_gt_object_points = gt_object_points_list
        self.m_gt_object_visibilities = gt_object_visibilities_list
        self.m_gt_object_motions_valid = gt_object_motions_valid_list
        
        # 使用当前阶段的顶点、质量等参数
        vertice = self.m_vertices[stage]
        masses = self.m_masses[stage]
        
        spring_graph = self.m_gs[stage].int().T.contiguous()
        num_edge = len(spring_graph)
        spring_graph = spring_graph[:num_edge // 2]

        node_type = self.m_node_type[stage]
        
        rest_lengths = torch.norm((vertice[spring_graph[:, 0]]
                                   -vertice[spring_graph[:, 1]]), dim=1)
        
        # 使用逐层下采样的 GT 数据
        gt_object_points = gt_object_points_list[stage]
        gt_object_visibilities = gt_object_visibilities_list[stage]
        gt_object_motions_valid = gt_object_motions_valid_list[stage]
        
        simulator = self.mdata.create_spring_mass_sim(vertice, 
                                                    spring_graph, 
                                                    rest_lengths, 
                                                    masses,
                                                    node_type,
                                                    gt_object_points, 
                                                    gt_object_visibilities, 
                                                    gt_object_motions_valid)
        
        if cfg.collision_learn:
            torch_collide_elas = wp.to_torch(simulator.wp_collide_elas)
            torch_collide_fric = wp.to_torch(simulator.wp_collide_fric)
            torch_collide_object_elas = wp.to_torch(simulator.wp_collide_object_elas)
            torch_collide_object_fric = wp.to_torch(simulator.wp_collide_object_fric)
            print(f"collide_elas {torch_collide_elas}")
            warp_params_list = [
                torch_collide_elas,
                torch_collide_fric,
                torch_collide_object_elas,
                torch_collide_object_fric,
            ]
        else:
            warp_params_list = []
        
        collide_optimizer = torch.optim.Adam(warp_params_list,
                                          lr=self.args.lr,
                                          betas=(0.9, 0.99))
        
        self.mlvl_simulators.append(simulator)
        self.mlvl_collide_optimizer.append(collide_optimizer)
        logger.info(f"Simulator created for stage {stage} with {vertice.shape[0]} nodes")
    
    def _save_stage_result(self, stage):
        """
        保存当前阶段的最佳结果
        保留之前阶段的设计，累积保存所有阶段的最佳结果
        现在每个阶段包含完整的下采样信息
        
        修改说明：
        1. 结果保存到按时间戳创建的 save 文件夹下
        2. 每个 stage 保存时，累积保存从 stage 0 到当前 stage 的所有结果
        3. 调用 _save_accumulated_results 保存累积的所有阶段结果
        """
        if self.stage_best_mech_info[stage] is None:
            logger.warning(f"No best mech info for stage {stage}, skipping save")
            return
        
        # 保存当前阶段的最佳 mech_info（已包含完整信息）
        stage_mech_info = self.stage_best_mech_info[stage]
        
        # 累积之前阶段的设计（如果存在）
        accumulated_mech_info = []
        for prev_stage in range(stage + 1):
            if self.stage_best_mech_info[prev_stage] is not None:
                accumulated_mech_info.append(self.stage_best_mech_info[prev_stage])
            else:
                logger.warning(f"Stage {prev_stage} has no best mech info, skipping")
        
        # 使用 stage_epoch_iter 格式保存累积的 mech_info
        best_epoch = self.stage_best_epochs[stage]
        best_iter = self.stage_best_iterations[stage]
        
        # 保存到时间戳文件夹下
        best_mech_filename = f'stage{stage}_epoch{best_epoch}_iter{best_iter}_accumulated_best_mech_info.pth'
        best_mech_info_path = os.path.join(self.save_base_dir, 'spring_mech_info', best_mech_filename)
        
        # 保存累积的 mech_info，每个 stage 现在包含完整信息：
        # - log_spring_Y, drag_damping, dashpot_damping (力学参数，杨氏模量以 log 保存)
        # - collision_elas, collision_fric, collision_object_elas, collision_object_fric (碰撞参数)
        # - vertices (下采样后的节点位置：object + interior + surface + controller points)
        # - edges (下采样后的连接关系)
        # - node_type (下采样后的节点类型)
        # - masses (下采样后的节点质量)
        # - gt_vertices, gt_visibility, gt_motions_valid (下采样后的 GT 数据)
        # - node_ids (节点 ID 映射)
        torch.save({'mech': accumulated_mech_info}, best_mech_info_path)
        logger.info(f"Saved accumulated mech info with complete stage data to {best_mech_info_path}")
        
        # 保存当前阶段的最佳轨迹（使用累积的设计，文件名包含 stage_epoch_iter）
        # 同时保存累积的所有 stage 的轨迹结果
        best_traj_filename = f'stage{stage}_epoch{best_epoch}_iter{best_iter}_best_trajectory.pkl'
        best_traj_path = os.path.join(self.save_base_dir, 'trajectories', best_traj_filename)
        
        # 为 save_traj 提取力学参数部分（save_traj 只需要力学参数用于仿真）
        mlvl_mech_info_for_traj = []
        for mech_info in accumulated_mech_info:
            traj_mech = {
                'log_spring_Y': mech_info['log_spring_Y'],
                'drag_damping': mech_info['drag_damping'],
                'dashpot_damping': mech_info['dashpot_damping'],
                'collision_elas': mech_info['collision_elas'],
                'collision_fric': mech_info['collision_fric'],
                'collision_object_elas': mech_info['collision_object_elas'],
                'collision_object_fric': mech_info['collision_object_fric'],
            }
            mlvl_mech_info_for_traj.append(traj_mech)
        
        self.save_traj(mlvl_mech_info=mlvl_mech_info_for_traj, 
                       save_path=best_traj_path, 
                       compute_loss=True,
                       gt_indices=self.m_ids,
                       stage_idx=stage,
                       mlvl_masses=self.m_masses,
                       mlvl_edges=self.m_gs)
        
        # 调用 _save_accumulated_results 保存累积的所有阶段结果（包含从 stage 0 到当前 stage 的所有轨迹）
        self._save_accumulated_results(stage, accumulated_mech_info)
        
        logger.info(f"Stage {stage} best result saved with {len(accumulated_mech_info)} levels (stages 0-{stage})")
        logger.info(f"  Mech info: {best_mech_info_path}")
        logger.info(f"  Trajectory: {best_traj_path}")
    
    def _save_accumulated_results(self, current_stage, accumulated_mech_info):
        """
        保存累积的所有阶段的结果（从 stage 0 到 current_stage）
        
        这个函数会在每个阶段结束时被调用，保存当前阶段和之前所有阶段的累积结果
        
        Args:
            current_stage: 当前阶段索引
            accumulated_mech_info: 累积的所有阶段力学参数列表
        """
        logger.info(f"Saving accumulated results for stages 0-{current_stage}...")
        
        # 保存累积的所有 stage 的结果（包含从 stage 0 到当前 stage 的所有轨迹）
        accumulated_traj_filename = f'stage{current_stage}_accumulated_all_stages_trajectory.pkl'
        accumulated_traj_path = os.path.join(self.save_base_dir, 'trajectories', accumulated_traj_filename)
        
        # 为每个已完成的 stage 生成轨迹
        all_stages_traj_data = []
        for completed_stage in range(current_stage + 1):
            if self.stage_best_mech_info[completed_stage] is not None:
                stage_traj_mech = {
                    'log_spring_Y': self.stage_best_mech_info[completed_stage]['log_spring_Y'],
                    'drag_damping': self.stage_best_mech_info[completed_stage]['drag_damping'],
                    'dashpot_damping': self.stage_best_mech_info[completed_stage]['dashpot_damping'],
                    'collision_elas': self.stage_best_mech_info[completed_stage]['collision_elas'],
                    'collision_fric': self.stage_best_mech_info[completed_stage]['collision_fric'],
                    'collision_object_elas': self.stage_best_mech_info[completed_stage]['collision_object_elas'],
                    'collision_object_fric': self.stage_best_mech_info[completed_stage]['collision_object_fric'],
                }
                all_stages_traj_data.append({
                    'stage': completed_stage,
                    'mech_info': stage_traj_mech,
                    'loss': self.stage_best_losses[completed_stage],
                    'epoch': self.stage_best_epochs[completed_stage],
                    'iter': self.stage_best_iterations[completed_stage],
                })
        
        # 保存累积的所有 stage 的信息
        accumulated_save_data = {
            'all_stages_traj_data': all_stages_traj_data,
            'accumulated_mech_info': accumulated_mech_info,
            'current_stage': current_stage,
            'timestamp': self.run_timestamp,
        }
        
        with open(accumulated_traj_path, "wb") as f:
            pickle.dump(accumulated_save_data, f)
        
        logger.info(f"Saved accumulated all stages trajectory to {accumulated_traj_path}")
    
    def _merge_stage_results(self):
        """
        合并所有阶段的最佳结果为 global_best
        现在每个阶段包含完整的下采样数据
        
        修改说明：
        1. 结果保存到按时间戳创建的 save 文件夹下
        2. 同时保存一份到原来的 spring_mech_info 目录以便兼容
        """
        logger.info("Merging stage best results into global best...")
        
        self.global_best_mech_info = []
        for stage in range(self.num_stages):
            if self.stage_best_mech_info[stage] is not None:
                self.global_best_mech_info.append(self.stage_best_mech_info[stage])
                logger.info(f"  Stage {stage}: loss={self.stage_best_losses[stage]:.6f}, "
                           f"epoch={self.stage_best_epochs[stage]}, iter={self.stage_best_iterations[stage]}")
            else:
                logger.warning(f"  Stage {stage}: no best result available")
        
        # 保存合并后的全局最佳结果（包含所有阶段的完整信息）
        if len(self.global_best_mech_info) > 0:
            # 时间戳文件夹路径
            timestamp_mech_info_path = os.path.join(self.save_base_dir, 'spring_mech_info', 'global_best_mech_info.pth')
            
            # global_best_mech_info 现在包含每个阶段的完整信息：
            # 对于每个 stage:
            #   - log_spring_Y, drag_damping, dashpot_damping (力学参数，杨氏模量以 log 保存)
            #   - collision_elas, collision_fric, collision_object_elas, collision_object_fric (碰撞参数)
            #   - vertices (下采样后的节点位置：object + interior + surface + controller points)
            #   - edges (下采样后的连接关系)
            #   - node_type (下采样后的节点类型)
            #   - masses (下采样后的节点质量)
            #   - gt_vertices, gt_visibility, gt_motions_valid (下采样后的 GT 数据)
            #   - node_ids (节点 ID 映射)
            save_data = {'mech': self.global_best_mech_info}
            torch.save(save_data, timestamp_mech_info_path)
            logger.info(f"Saved global_best_mech_info with complete stage data to {timestamp_mech_info_path}")
            
            # 保存全局最佳轨迹（提取力学参数部分用于仿真）
            # 时间戳文件夹路径
            timestamp_traj_path = os.path.join(self.save_base_dir, 'trajectories', 'global_best_trajectory.pkl')
            
            # 为 save_traj 提取力学参数
            mlvl_mech_info_for_traj = []
            for mech_info in self.global_best_mech_info:
                traj_mech = {
                    'log_spring_Y': mech_info['log_spring_Y'],
                    'drag_damping': mech_info['drag_damping'],
                    'dashpot_damping': mech_info['dashpot_damping'],
                    'collision_elas': mech_info['collision_elas'],
                    'collision_fric': mech_info['collision_fric'],
                    'collision_object_elas': mech_info['collision_object_elas'],
                    'collision_object_fric': mech_info['collision_object_fric'],
                }
                mlvl_mech_info_for_traj.append(traj_mech)
            
            # 保存轨迹到时间戳文件夹
            self.save_traj(mlvl_mech_info=mlvl_mech_info_for_traj, 
                           save_path=timestamp_traj_path, 
                           compute_loss=False,
                           gt_indices=self.m_ids,
                           mlvl_masses=self.m_masses,
                           mlvl_edges=self.m_gs)
            
            logger.info(f"Global best trajectory saved to {timestamp_traj_path}")
            
            # 保存一个汇总文件，记录所有 stage 的累积结果
            all_stages_summary_path = os.path.join(self.save_base_dir, 'trajectories', 'all_stages_summary.pkl')
            
            # 构建所有 stage 的汇总数据
            all_stages_summary = {
                'timestamp': self.run_timestamp,
                'num_stages': self.num_stages,
                'stage_results': []
            }
            
            for stage in range(self.num_stages):
                if self.stage_best_mech_info[stage] is not None:
                    stage_summary = {
                        'stage': stage,
                        'loss': self.stage_best_losses[stage],
                        'epoch': self.stage_best_epochs[stage],
                        'iter': self.stage_best_iterations[stage],
                        'mech_info': {
                            'log_spring_Y': self.stage_best_mech_info[stage]['log_spring_Y'],
                            'drag_damping': self.stage_best_mech_info[stage]['drag_damping'],
                            'dashpot_damping': self.stage_best_mech_info[stage]['dashpot_damping'],
                            'collision_elas': self.stage_best_mech_info[stage]['collision_elas'],
                            'collision_fric': self.stage_best_mech_info[stage]['collision_fric'],
                            'collision_object_elas': self.stage_best_mech_info[stage]['collision_object_elas'],
                            'collision_object_fric': self.stage_best_mech_info[stage]['collision_object_fric'],
                        },
                        'vertices_shape': self.stage_best_mech_info[stage]['vertices'].shape,
                        'edges_shape': self.stage_best_mech_info[stage]['edges'].shape,
                    }
                    all_stages_summary['stage_results'].append(stage_summary)
            
            with open(all_stages_summary_path, "wb") as f:
                pickle.dump(all_stages_summary, f)
            
            logger.info(f"Saved all stages summary to {all_stages_summary_path}")
            
            # 打印每个阶段保存的数据摘要
            for stage, mech_info in enumerate(self.global_best_mech_info):
                logger.info(f"\n  Stage {stage} saved data summary:")
                logger.info(f"    - Vertices shape: {mech_info['vertices'].shape}")
                logger.info(f"    - Edges shape: {mech_info['edges'].shape}")
                logger.info(f"    - Node types: {mech_info['node_type'].shape}")
                logger.info(f"    - Masses shape: {mech_info['masses'].shape if isinstance(mech_info['masses'], torch.Tensor) else len(mech_info['masses'])}")
                logger.info(f"    - GT vertices shape: {mech_info['gt_vertices'].shape}")
                logger.info(f"    - Spring Y (log) shape: {mech_info['log_spring_Y'].shape}")
                logger.info(f"    - Drag damping: {mech_info['drag_damping'].item():.6f}")
                logger.info(f"    - Dashpot damping: {mech_info['dashpot_damping'].item():.6f}")
    
    def run_epoch(self, epoch, mode='train', stage=0):
        """
        运行一个 epoch，只监督指定阶段的层
        
        Args:
            epoch: 当前 epoch 索引
            mode: 'train' 或 'val'
            stage: 当前训练阶段（只监督该阶段的层）
        """
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
        ], dim=1)

        if mode != 'train':
            self.model.eval()

        # 运行仿真
        if mode == 'train':
            for i in range(self.iter_per_epoch):
                self.optimizer.zero_grad()
                
                st = time()

                # 1. 执行模型前向传播获取 downsample 结果和力学参数
                # 从配置中获取 object_radius 参数（用于控制 merge 的距离约束）
                object_radius = getattr(cfg, 'object_radius', None)
                
                mlvl_s_out, mlvl_drag_damping_out, mlvl_dashpot_damping_out, downsample_results, pooling_losses = self.model(
                    self.m_ids, 
                    self.m_gs, 
                    self.m_proj,
                    self.m_vertices,
                    self.m_masses,
                    self.m_node_type,
                    node_in_feature,
                    object_radius
                )
                
                # 在分阶段训练中，只使用当前阶段的 pooling loss
                # pooling_losses 包含每个 downpooling layer 的 loss，我们只取当前阶段对应的 loss
                if stage < len(pooling_losses):
                    stage_pooling_loss = pooling_losses[stage]
                else:
                    stage_pooling_loss = torch.tensor(0.0, device=cfg.device)
                
                # 2. 使用 downsample 结果更新拓扑结构（只在第一次迭代时）
                logger.info(f"Updating topology structure at epoch {epoch}, iter {i+1}")
                self.m_ids = downsample_results['down_ids'][:stage+1]
                self.m_proj = downsample_results['down_proj'][:stage] # downsample proj number is level - 1
                self.m_vertices = downsample_results['down_ps'][:stage+1]
                self.m_masses = downsample_results['down_mass'][:stage+1]
                self.m_node_type = downsample_results['down_type'][:stage+1]
                self.m_gs = downsample_results['down_gs'][:stage+1]

                if epoch == 0 and i==0:
                    # 重新创建当前阶段的 simulator
                    self._create_stage_simulator(stage)

                # 3. 只处理当前阶段的层
                level_info = {
                    'log_spring_Y': mlvl_s_out[stage],
                    'drag_damping': mlvl_drag_damping_out[stage],
                    'dashpot_damping': mlvl_dashpot_damping_out[stage],
                }
                
                # [修复] 使用当前 stage 对应的 simulator
                current_simulator = self.mlvl_simulators[stage]
                
                # 设置 simulator 参数并运行仿真
                log_new_spring_Y = torch.log(level_info['log_spring_Y'])
                wp_predicted_spring_Y = wp.from_torch(
                    log_new_spring_Y.contiguous(), dtype=wp.float32, requires_grad=True
                )
                wp_predicted_drag_damping = wp.from_torch(
                    level_info['drag_damping'], dtype=wp.float32, requires_grad=True
                )
                wp_predicted_dashpot_damping = wp.from_torch(
                    level_info['dashpot_damping'], dtype=wp.float32, requires_grad=True
                )

                current_simulator.set_spring_Y(wp_predicted_spring_Y)
                current_simulator.set_drag_damping(wp_predicted_drag_damping)
                current_simulator.set_dashpot_damping(wp_predicted_dashpot_damping)
                
                # 运行仿真序列
                pos, _, _, _, _, _, _, loss_val, chamfer_loss_val, track_loss_val, grad_avg_dict, valid_frames = \
                    self.generate_data_point_sequence(
                        current_simulator,
                        update_frame_num=cfg.train_frame,
                        enable_backward=True,
                    )

                # 获取碰撞参数
                wp_collide_elas = wp.to_torch(current_simulator.wp_collide_elas).clone()
                wp_collide_fric = wp.to_torch(current_simulator.wp_collide_fric).clone()
                wp_collide_object_elas = wp.to_torch(current_simulator.wp_collide_object_elas).clone()
                wp_collide_object_fric = wp.to_torch(current_simulator.wp_collide_object_fric).clone()
                
                # 构建当前阶段的 mech_info
                current_mech_info = {
                    'log_spring_Y': log_new_spring_Y.detach().clone(),
                    'drag_damping': level_info['drag_damping'].detach().clone(),
                    'dashpot_damping': level_info['dashpot_damping'].detach().clone(),
                    'collision_elas': wp_collide_elas.detach().clone(),
                    'collision_fric': wp_collide_fric.detach().clone(),
                    'collision_object_elas': wp_collide_object_elas.detach().clone(),
                    'collision_object_fric': wp_collide_object_fric.detach().clone(),
                }
                
                # 计算损失
                simulation_loss = np.sum(loss_val)
                pooling_loss_value = stage_pooling_loss.item() * 0.000 
                total_loss_val = simulation_loss # + pooling_loss_value

                print("Forward time is : {}".format(time() - st))
                st = time()
                
                self.total_update += 1

                # 打印信息
                # 获取 spring rest_length 统计信息
                spring_rest_lengths = wp.to_torch(current_simulator.wp_rest_lengths, requires_grad=False)
                rest_length_min = spring_rest_lengths.min().item()
                rest_length_max = spring_rest_lengths.max().item()
                rest_length_mean = spring_rest_lengths.mean().item()
                
                print(f"\n{'='*60}")
                print(f"Stage {self.current_stage}, Epoch {epoch}")
                print(f"Spring Y iteration {i+1}/10")
                print(f"spring_Y grad Average: {grad_avg_dict['log_spring_Y'].mean().item()}")
                print(f"Spring_Y Average: {log_new_spring_Y.mean().item():.6f}")
                print(f"drag_damping grad Average: {grad_avg_dict['drag_damping'].mean().item()}")
                print(f"Drag_Damping: {level_info['drag_damping'].item():.6f}")
                print(f"dashpot_damping grad Average: {grad_avg_dict['dashpot_damping'].mean().item()}")
                print(f"Dashpot_Damping: {level_info['dashpot_damping'].item():.6f}")
                print(f"Simulation Loss: {simulation_loss:.10f}")
                print(f"Pooling Loss: {pooling_loss_value:.10f}")
                print(f"Total Loss: {total_loss_val:.10f}")
                print(f"Valid frames: {valid_frames}")
                print(f"Spring Rest Length - Min: {rest_length_min:.6f}, Max: {rest_length_max:.6f}, Mean: {rest_length_mean:.6f}")
                print(f"{'='*60}\n")

                self.writer.add_scalar(f'Stage_{self.current_stage}/overall_Loss', total_loss_val, self.total_update)
                self.writer.add_scalar(f'Stage_{self.current_stage}/pooling_Loss', pooling_loss_value, self.total_update)
                self.writer.add_scalar(f'Stage_{self.current_stage}/simulation_Loss', simulation_loss, self.total_update)
                self.writer.add_scalar(f'Stage_{self.current_stage}/Spring_Y_Average', log_new_spring_Y.mean().item(), self.total_update)
                self.writer.add_scalar(f'Stage_{self.current_stage}/Drag_Damping', level_info['drag_damping'].item(), self.total_update)
                self.writer.add_scalar(f'Stage_{self.current_stage}/Dashpot_Damping', level_info['dashpot_damping'].item(), self.total_update)
 
                # 反向传播
                tensors_to_backward = [log_new_spring_Y, level_info['drag_damping'], level_info['dashpot_damping']]
                grads_to_backward = [
                    grad_avg_dict['log_spring_Y'],
                    grad_avg_dict['drag_damping'],
                    grad_avg_dict['dashpot_damping'],
                ]
                
                # 添加 pooling loss
                tensors_to_backward.append(stage_pooling_loss)
                grads_to_backward.append(torch.ones_like(stage_pooling_loss)*0.000)
                
                torch.autograd.backward(
                    tensors=tensors_to_backward,
                    grad_tensors=grads_to_backward
                )

                self.optimizer.step()

                # 检查并保存当前阶段的最佳结果
                if total_loss_val < self.stage_best_losses[self.current_stage]:
                    prev_best = self.stage_best_losses[self.current_stage]
                    self.stage_best_losses[self.current_stage] = total_loss_val

                    num_edge = self.m_gs[self.current_stage].shape[1]
                    
                    # 构建完整的 stage 信息，包含所有下采样后的数据
                    complete_stage_info = {
                        # 1. 力学参数 (以log保存弹簧杨氏模量)
                        'log_spring_Y': log_new_spring_Y.detach().clone(),
                        'drag_damping': level_info['drag_damping'].detach().clone(),
                        'dashpot_damping': level_info['dashpot_damping'].detach().clone(),
                        
                        # 2. 碰撞参数
                        'collision_elas': wp_collide_elas.detach().clone(),
                        'collision_fric': wp_collide_fric.detach().clone(),
                        'collision_object_elas': wp_collide_object_elas.detach().clone(),
                        'collision_object_fric': wp_collide_object_fric.detach().clone(),
                        
                        # 3. 下采样后的节点位置（包含 object_point, interior point, surface point, controller point）
                        'vertices': self.m_vertices[self.current_stage].detach().clone(),
                        
                        # 4. 下采样后的连接关系（边拓扑）
                        'edges': self.m_gs[self.current_stage].detach().clone()[:, :num_edge//2],
                        
                        # 5. 下采样后的节点类型
                        'node_type': self.m_node_type[self.current_stage].detach().clone(),
                        
                        # 6. 下采样后的节点质量
                        'masses': self.m_masses[self.current_stage].clone() if isinstance(self.m_masses[self.current_stage], torch.Tensor) else torch.tensor(self.m_masses[self.current_stage], device=cfg.device),
                        
                        # 7. 下采样后的 GT 数据（gt_vertices 和 visibility）
                        'gt_vertices': self.m_gt_object_points[self.current_stage].detach().clone(),
                        'gt_visibility': self.m_gt_object_visibilities[self.current_stage].detach().clone(),
                        'gt_motions_valid': self.m_gt_object_motions_valid[self.current_stage].detach().clone(),
                        
                        # 8. 节点 ID 映射（下采样索引）
                        'node_ids': self.m_ids[self.current_stage],
                    }
                    
                    self.stage_best_mech_info[self.current_stage] = complete_stage_info
                    self.stage_best_epochs[self.current_stage] = epoch
                    self.stage_best_iterations[self.current_stage] = i
                    
                    print(f"\n{'='*60}")
                    print(f"NEW STAGE {self.current_stage} BEST!")
                    print(f"Epoch: {epoch}, Iteration: {i+1}/10")
                    print(f"Loss: {total_loss_val:.10f}")
                    print(f"Previous best: {prev_best:.10f}")
                    print(f"{'='*60}\n")
                    
                    # 立即保存当前最优的 mech_info，文件名包含 stage_epoch_iter 标签
                    # 保存到时间戳文件夹下
                    best_mech_filename = f'stage{self.current_stage}_epoch{epoch}_iter{i}_best_mech_info.pth'
                    best_mech_info_path = os.path.join(self.save_base_dir, 'spring_mech_info', best_mech_filename)
                    torch.save({'mech': complete_stage_info}, best_mech_info_path)
                    logger.info(f"Saved best mech info for stage {self.current_stage} to {best_mech_info_path}")

            # 碰撞参数优化
            if cfg.collision_learn:
                print(f"\n{'='*60}")
                print("Starting collide parameters optimization...")
                print(f"{'='*60}\n")
                
                # [修复] 为当前 stage 的 simulator 优化碰撞参数
                # 使用 self.current_stage 索引来获取对应的 simulator
                simulator = self.mlvl_simulators[self.current_stage]
                collide_optimizer = self.mlvl_collide_optimizer[self.current_stage]
                
                for opt_iter in range(5):
                    iter_losses = []
                    
                    simulator.set_init_state(
                        simulator.wp_init_vertices,
                        simulator.wp_init_velocities
                    )
                    
                    for j in range(1, cfg.train_frame):
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
                        
                        collide_optimizer.step()
                        
                        loss = wp.to_torch(simulator.loss, requires_grad=False)
                        iter_losses.append(loss.item())
                        
                        if cfg.use_graph:
                            simulator.tape.zero()
                        else:
                            simulator.tape.reset()
                        
                        simulator.clear_loss()
                        simulator.set_init_state(
                            simulator.wp_states[-1].wp_x,
                            simulator.wp_states[-1].wp_v
                        )
                    
                    avg_loss = np.mean(iter_losses)
                    print(f"Collide iter {opt_iter+1}/5 - Avg Loss: {avg_loss:.10f}")
        else:
            # 验证模式
            _, _, _, _, _, _, _, loss, chamfer_loss, track_loss = self.generate_data_point_sequence(
                update_frame_num=self.mdata.train_frame,
                enable_backward=False,
                set_object_point=False
            )
            total_loss_val = loss

        print("Backward/Overhead time : {}".format(time() - st))
        mean_loss = np.sum(total_loss_val) if isinstance(total_loss_val, list) else total_loss_val

        if mode == 'train':
            return mean_loss, [current_mech_info] if 'current_mech_info' in locals() else None
        else:
            return mean_loss, None, None