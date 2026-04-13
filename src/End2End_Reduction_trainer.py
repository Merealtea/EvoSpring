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
from Reducer import FastAdaptiveNetworkReducer
import warp as wp
from time import time
import pickle
import logging
from visualization_utils import visualize_level_results, LevelVisualizer
import threading
from concurrent.futures import ThreadPoolExecutor

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

        self.epochs_per_stage = 10
        self.iter_per_epoch = 10
        self.total_epochs = self.epochs_per_stage
        
        self.pbar = tqdm(total=self.total_epochs, unit="iters")

        os.makedirs(self.args.dump_dir, exist_ok=True)
        for subdir in ['ckpts', 'log', 'test_RMSE', 'spring_mech_info', 'trajectories']:
            dir = os.path.join(self.args.dump_dir, subdir)
            os.makedirs(dir, exist_ok=True)

        self.total_update = 0
        self.reducer = FastAdaptiveNetworkReducer(num_modes=500)

        # Track global best results
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
        """
        logger.info(f"Exporting multi-level topology to {save_dir}...")
        
        # 创建导出目录
        export_path = os.path.join(self.args.dump_dir, save_dir)
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
        
        # 保存每个 level 的质点质量信息
        if mlvl_masses is not None:
            save_data['masses'] = []
            for level_idx, level_masses in enumerate(mlvl_masses):
                if isinstance(level_masses, torch.Tensor):
                    save_data['masses'].append(level_masses.cpu().numpy())
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
        self.m_proj = None
        self.m_vertices = [torch.tensor(mdata.init_vertices, dtype=torch.float32, device=cfg.device)]
        self.m_masses = [mdata.init_masses]
        self.m_node_type = [torch.tensor(mdata.node_type[0, :, 0], dtype=torch.long, device=cfg.device)]

        self.m_gt_object_points = [mdata.object_points.clone().to(cfg.device)]
        self.m_gt_object_visibilities = [mdata.object_visibilities.clone().to(cfg.device)]
        self.m_gt_object_motions_valid = [mdata.object_motions_valid.clone().to(cfg.device)]
    
    def downsample(self, stage_idx):
        # 1. 启发式阈值设定：阶段越往后，允许合并的差异度越大
        
        logger.info(f"Applying adaptive downsampling for Stage {stage_idx}...")
        
        node_mass = self.m_masses[-1].cpu().numpy()
        gs = self.m_gs[-1].cpu().numpy()

        gs = gs[:, :len(gs[0])//2]
        
        # 获取全局最优的力学参数
        if self.global_best_mech_info is None:
            logger.warning("No best mech info available, using default values")
            # 使用默认值进行下采样
            spring_Y = np.ones(len(node_mass)) * 1000.0  # 默认 spring Y
            drag_damping = 0.1
            dashpot_damping = 0.1
        else:
            best_mech = self.global_best_mech_info[-1]
            spring_Y = np.exp(best_mech['log_spring_Y'].cpu().numpy())
            drag_damping = best_mech['drag_damping'].item()
            dashpot_damping = best_mech['dashpot_damping'].item()
        
        node_type = self.m_node_type[-1]

        # 2. 调用自适应降维器进行聚类
        # reducer 内部会将物理参数转换为 M, D, L 矩阵并进行格拉姆矩阵聚类
        P_np, M_hat_np, D_hat_np, L_hat_np = self.reducer.reduce(
            node_mass, gs, spring_Y, dashpot_damping, drag_damping, node_type
        )
        
        new_node_count = P_np.shape[1]
        logger.info(f"Stage {stage_idx} reduced to {new_node_count} nodes.")
        
        # ====================================================================
        # 3. 生成新的拓扑结构 (new_ids 和 new_gs)
        # ====================================================================
        
        # 【提取 new_ids】：只保留每个簇的第一个节点作为代表 idx
        # argmax(axis=0) 顺着每一列找到第一个 1 的位置
        new_ids = np.argmax(P_np, axis=0).tolist()
        self.m_ids.append(new_ids)

        # assignment[i] 表示旧节点 i 映射到的新节点本地索引 (0 到 N_new - 1)
        assignment = np.argmax(P_np, axis=1)
    
        # 确保 gs 是 numpy 数组格式
        if isinstance(gs, torch.Tensor):
            gs_np = gs.detach().cpu().numpy()
        else:
            gs_np = gs
            
        # 将旧边映射到 0 到 N_new-1 的新节点局部索引上
        new_u = assignment[gs_np[0]]
        new_v = assignment[gs_np[1]]
        
        # 过滤掉自环 (被合并到同一个簇内的节点之间的边)
        valid_mask = new_u != new_v
        new_edges = np.stack([new_u[valid_mask], new_v[valid_mask]], axis=0)
        
        # 去重并生成双向边
        edges_sorted = np.sort(new_edges, axis=0)
        new_edges_unique = np.unique(edges_sorted, axis=1)
        new_edges_bidir = np.concatenate([new_edges_unique, new_edges_unique[::-1]], axis=1)

        
        # 保存投影矩阵 P，供神经网络 Forward 过程聚合特征使用
        P_tensor = torch.tensor(P_np, dtype=torch.float32, device=cfg.device)
        self.m_proj.append(P_tensor)

        new_gs = torch.tensor(new_edges_bidir, dtype=torch.long, device=cfg.device)
        self.m_gs.append(new_gs)

        # ====================================================================
        # 4. [新增] 传播物理属性：Mass, Node_Type, Vertices
        # ====================================================================
        # 质量 (Mass): 簇内质量总和。M_hat_np 是对角阵，对角线就是各簇的新质量
        new_masses = torch.tensor(np.diag(M_hat_np), dtype=torch.float32, device=cfg.device)
        self.m_masses.append(new_masses)
        
        # 节点类型 (Node Type): 直接继承代表节点的类型
        # 注意这里使用的是相对上一层的局部索引 new_ids，所以我们从上一层的 node_type 中取
        current_node_type = self.m_node_type[-1]  
        new_node_type = current_node_type[new_ids]
        self.m_node_type.append(new_node_type)

        # 物理坐标 (Vertices): 直接继承代表节点的坐标
        prev_vertices = self.m_vertices[-1]
        self.m_vertices.append(prev_vertices[new_ids])

        # ====================================================================
        # [新增] 5. GT Points 下采样逻辑
        # ====================================================================
        # GT 数据是针对全局原始图的 (即 mdata 中的维度)。
        # 因为 new_ids 是相对上一层的局部索引，而上一层可能已经筛掉了一些点。
        # 所以我们需要用 m_ids (包含映射到最初第 0 层全局索引的映射表) 来提取 GT 数据。
        
        # 当前层在原始全集中的绝对索引
        current_global_ids = self.m_ids[-1] 
        
        # 找出当前层中，哪些节点的 node_type 是 0 (即可观测的 Object Points)
        # 注意：这里 np.array(range(len(current_global_ids))) 生成的是 0 到 N_new-1 的局部索引
        # 我们用它来筛选出满足条件的节点的局部位置，然后再反查 global_ids
        local_object_idx = np.array(range(len(current_global_ids)))[new_node_type == 0].tolist()
        
        # 把这些局部位置映射回第 0 层的全局位置
        global_object_idx = [current_global_ids[idx] for idx in local_object_idx]

        # 从原始的 dataset 中直接抽取这些保留下来的全局节点所对应的 GT 轨迹
        new_gt_points = self.mdata.object_points[:, global_object_idx].clone().to(cfg.device)
        new_gt_vis = self.mdata.object_visibilities[:, global_object_idx].clone().to(cfg.device)
        new_gt_valid = self.mdata.object_motions_valid[:, global_object_idx].clone().to(cfg.device)

        self.m_gt_object_points.append(new_gt_points)
        self.m_gt_object_visibilities.append(new_gt_vis)
        self.m_gt_object_motions_valid.append(new_gt_valid)

    
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


    def run_epoch(self, epoch, mode='train'):
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
            # ====================================================================
            # [修改] 每次 run_epoch 的训练迭代中：
            # 1. 先执行模型前向传播获取 downsample 结果和力学参数
            # 2. 根据 downsample 结果创建/更新 simulator
            # 3. 使用 simulator 进行后续训练
            # ====================================================================
            
            # ====================================================================
            # 开始正常的训练迭代
            # ====================================================================
            self.m_proj = None
            for i in range(self.iter_per_epoch):
                # 获取平均梯度
                self.optimizer.zero_grad()
                
                st = time()

                # 1. 执行模型前向传播获取 downsample 结果和力学参数
                mlvl_s_out, mlvl_drag_damping_out, mlvl_dashpot_damping_out, downsample_results, pooling_loss = self.model(
                    self.m_ids, 
                    self.m_gs, 
                    self.m_proj,
                    self.m_vertices,
                    self.m_masses,
                    self.m_node_type,
                    node_in_feature
                )
                
                # 2. 使用 downsample 结果更新拓扑结构（如果有）
                if i == 0:
                    # 只在第一次迭代时更新拓扑结构
                    logger.info(f"Updating topology structure at epoch {epoch}, iter {i+1}")
                    self.m_ids = downsample_results['down_ids']
                    self.m_gs = downsample_results['down_gs']
                    self.m_proj = downsample_results['down_proj']
                    self.m_vertices = downsample_results['down_ps']
                    self.m_masses = downsample_results['down_mass']
                    self.m_node_type = downsample_results['down_type']
                    
                    # 保存旧的 collision 参数以便继承
                    old_collide_params = []
                    if len(self.mlvl_simulators) > 0:
                        for old_sim in self.mlvl_simulators:
                            old_collide_params.append({
                                'collide_elas': wp.to_torch(old_sim.wp_collide_elas).clone(),
                                'collide_fric': wp.to_torch(old_sim.wp_collide_fric).clone(),
                                'collide_object_elas': wp.to_torch(old_sim.wp_collide_object_elas).clone(),
                                'collide_object_fric': wp.to_torch(old_sim.wp_collide_object_fric).clone(),
                            })
                    
                    # 3. 根据新的拓扑结构创建 simulator
                    logger.info("Creating simulators based on model forward output...")
                    new_simulators = []
                    new_collide_optimizers = []
                    
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
                    for lvl in range(1, self.args.multi_mesh_layer):
                        # down_ids[stage] 存储的是当前层节点在上一层的索引
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
                    
                    for lvl in range(self.args.multi_mesh_layer):
                        vertice = self.m_vertices[lvl]  # [1, N, 3]
                        masses = self.m_masses[lvl]
                        
                        spring_graph = self.m_gs[lvl].int().T.contiguous()
                        num_edge = len(spring_graph)
                        spring_graph = spring_graph[:num_edge // 2]
                        node_type = self.m_node_type[lvl]
                        
                        rest_lengths = torch.norm((vertice[spring_graph[:, 0]]
                                                   -vertice[spring_graph[:, 1]]), dim = 1)
                          
                        # 使用逐层下采样的 GT 数据
                        gt_object_points = gt_object_points_list[lvl]
                        gt_object_visibilities = gt_object_visibilities_list[lvl]
                        gt_object_motions_valid = gt_object_motions_valid_list[lvl]
                        simulator = self.mdata.create_spring_mass_sim(vertice, 
                                                                    spring_graph, 
                                                                    rest_lengths, 
                                                                    masses,
                                                                    node_type,
                                                                    gt_object_points, 
                                                                    gt_object_visibilities, 
                                                                    gt_object_motions_valid)

                        # 继承旧的 collision 参数数值
                        # 首先检查 old_collide_params 是否存在（即是否是第一次创建 simulator）
                        if len(old_collide_params) > 0 and lvl < len(old_collide_params):
                            old_params = old_collide_params[lvl]
                            wp_predicted_collision_elas = wp.from_torch(
                                old_params['collide_elas'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False
                            )
                            simulator.set_collision_elas(wp_predicted_collision_elas)
                            wp_predicted_collision_fric = wp.from_torch(
                                old_params['collide_fric'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False
                            )
                            simulator.set_collision_fric(wp_predicted_collision_fric)
                            wp_predicted_collision_object_elas = wp.from_torch(
                                old_params['collide_object_elas'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False
                            )
                            simulator.set_collision_object_elas(wp_predicted_collision_object_elas)
                            wp_predicted_collision_object_fric = wp.from_torch(
                                old_params['collide_object_fric'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False
                            )
                            simulator.set_collision_object_fric(wp_predicted_collision_object_fric)
                            logger.info(f"Level {lvl}: Inherited collision parameters from previous simulator")
                        else:
                            logger.info(f"Level {lvl}: No previous collision parameters to inherit (first time or new level)")

                        if cfg.collision_learn:
                            torch_collide_elas = wp.to_torch(simulator.wp_collide_elas)
                            torch_collide_fric = wp.to_torch(simulator.wp_collide_fric)
                            torch_collide_object_elas = wp.to_torch(simulator.wp_collide_object_elas)
                            torch_collide_object_fric = wp.to_torch(simulator.wp_collide_object_fric)

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
                        new_simulators.append(simulator)
                        new_collide_optimizers.append(collide_optimizer)
                    
                    self.mlvl_simulators = new_simulators
                    self.mlvl_collide_optimizer = new_collide_optimizers
                    logger.info(f"Created {len(self.mlvl_simulators)} simulators for all levels")
                
                # 4. 将力学参数打包成 mlvl_info 格式，包含碰撞参数
                # 注意：这里不能 detach，因为后续需要通过这些 tensor 进行梯度反传
                mlvl_info = []
                for lvl in range(self.args.multi_mesh_layer):
                    # if lvl < len(self.mlvl_simulators):
                    #     wp_collide_elas = wp.to_torch(self.mlvl_simulators[lvl].wp_collide_elas).clone()
                    #     wp_collide_fric = wp.to_torch(self.mlvl_simulators[lvl].wp_collide_fric).clone()
                    #     wp_collide_object_elas = wp.to_torch(self.mlvl_simulators[lvl].wp_collide_object_elas).clone()
                    #     wp_collide_object_fric = wp.to_torch(self.mlvl_simulators[lvl].wp_collide_object_fric).clone()
                    # else:
                    #     wp_collide_elas = torch.tensor([cfg.collide_elas], device=cfg.device)
                    #     wp_collide_fric = torch.tensor([cfg.collide_fric], device=cfg.device)
                    #     wp_collide_object_elas = torch.tensor([cfg.collide_object_elas], device=cfg.device)
                    #     wp_collide_object_fric = torch.tensor([cfg.collide_object_fric], device=cfg.device)
                    
                    mlvl_info.append({
                        'log_spring_Y': mlvl_s_out[lvl],  # 不 detach，保留梯度
                        'drag_damping': mlvl_drag_damping_out[lvl],  # 不 detach，保留梯度
                        'dashpot_damping': mlvl_dashpot_damping_out[lvl],  # 不 detach，保留梯度
                        # 'collision_elas': wp_collide_elas.detach().clone(),
                        # 'collision_fric': wp_collide_fric.detach().clone(),
                        # 'collision_object_elas': wp_collide_object_elas.detach().clone(),
                        # 'collision_object_fric': wp_collide_object_fric.detach().clone(),
                    })

                # [关键] 映射到 Warp 时，必须设置 requires_grad=True
                # 这样 Warp 才会为这些数组分配梯度缓冲区，供 Tape 使用

                gradients, losses = [], []

                # -------------------------------------------------------
                # [修改] 并行化：使用 ThreadPoolExecutor 并行处理所有层级的 forward 和 gradient 计算
                # 使用有序字典和索引确保 tensors 和 gradients 的对应关系
                # -------------------------------------------------------
                all_tensors_to_backward = []
                all_grad_tensors_to_backward = []
                mlvl_current_mech_info = [] # 保存当前迭代的所有层级力学参数
                mlvl_losses = []  # 保存所有层级的 loss
                mlvl_gradients = []  # 保存所有层级的 gradients

                num_levels = len(mlvl_info)  # 层级数量
                
                # [修改] 定义单层级的处理函数，返回结果包含索引以确保对应关系
                def process_single_level(level_idx):
                    """
                    处理单个层级的 forward、simulation 和 gradient 计算
                    返回包含索引的结果字典，确保 tensors 和 gradients 的对应关系
                    """
                    # mlvl_info 是字典列表，每个字典包含一个 level 的所有参数
                    # mlvl_info[level_idx] 返回该 level 的完整参数字典
                    level_info = mlvl_info[level_idx]
                    new_spring_Y = level_info['log_spring_Y']
                    drag_damping_out = level_info['drag_damping']
                    dashpot_damping_out = level_info['dashpot_damping']
                    
                    # set up spring mass parameters
                    log_new_spring_Y = torch.log(new_spring_Y)
                    wp_predicted_spring_Y = wp.from_torch(
                        log_new_spring_Y.contiguous(), dtype=wp.float32, requires_grad=True
                    )

                    # Convert predicted damping values to warp
                    wp_predicted_drag_damping = wp.from_torch(
                        drag_damping_out, dtype=wp.float32, requires_grad=True
                    )
                    wp_predicted_dashpot_damping = wp.from_torch(
                        dashpot_damping_out, dtype=wp.float32, requires_grad=True
                    )

                    self.mlvl_simulators[level_idx].set_spring_Y(wp_predicted_spring_Y)
                    self.mlvl_simulators[level_idx].set_drag_damping(wp_predicted_drag_damping)
                    self.mlvl_simulators[level_idx].set_dashpot_damping(wp_predicted_dashpot_damping)
                    
                    pos, vel, _, _, _, _, _, loss_val, chamfer_loss_val, track_loss_val, grad_avg_dict, valid_frames = \
                        self.generate_data_point_sequence(
                            self.mlvl_simulators[level_idx],
                            update_frame_num=cfg.train_frame,
                            enable_backward=True,
                        )
                    
                    # Get collision parameters from simulator
                    wp_collide_elas = wp.to_torch(self.mlvl_simulators[level_idx].wp_collide_elas).clone()
                    wp_collide_fric = wp.to_torch(self.mlvl_simulators[level_idx].wp_collide_fric).clone()
                    wp_collide_object_elas = wp.to_torch(self.mlvl_simulators[level_idx].wp_collide_object_elas).clone()
                    wp_collide_object_fric = wp.to_torch(self.mlvl_simulators[level_idx].wp_collide_object_fric).clone()


                    # 构建结果字典，包含索引以确保对应关系
                    return {
                        'level_idx': level_idx,
                        'tensors': [log_new_spring_Y, drag_damping_out, dashpot_damping_out],
                        'grads': [
                            grad_avg_dict['log_spring_Y'],
                            grad_avg_dict['drag_damping'],
                            grad_avg_dict['dashpot_damping'],
                        ],
                        'mech_info': {
                            'log_spring_Y': log_new_spring_Y.detach().clone(),
                            'drag_damping': drag_damping_out.detach().clone(),
                            'dashpot_damping': dashpot_damping_out.detach().clone(),
                            # Collision parameters
                            'collision_elas': wp_collide_elas.detach().clone(),
                            'collision_fric': wp_collide_fric.detach().clone(),
                            'collision_object_elas': wp_collide_object_elas.detach().clone(),
                            'collision_object_fric': wp_collide_object_fric.detach().clone(),
                        },
                        'loss_val': loss_val,
                        'grad_avg_dict': grad_avg_dict,
                        'valid_frames': valid_frames,
                    }
                
                # [修改] 使用 ThreadPoolExecutor 并行处理所有层级
                # 注意：由于 PyTorch 和 Warp 的线程安全性问题，这里使用顺序执行但保持并行接口
                # 未来可以迁移到多进程或分布式训练以实现真正的并行
                print(f"Processing {num_levels} levels...")
                
                
                # 由于 Warp tape 不是线程安全的，使用顺序执行
                for level_idx in range(num_levels):
                    result = process_single_level(level_idx)                    
                    
                    # 按顺序追加 tensors 和 gradients（一一对应）
                    all_tensors_to_backward.extend(result['tensors'])
                    all_grad_tensors_to_backward.extend(result['grads'])
                    
                    mlvl_current_mech_info.append(result['mech_info'])
                    mlvl_losses.append(result['loss_val'])
                    mlvl_gradients.append(result['grad_avg_dict'])
                
                # 更新原有的 gradients 和 losses 列表
                gradients = mlvl_gradients
                losses = mlvl_losses
                log_new_spring_Y = result['tensors'][0] 
                drag_damping_out = result['tensors'][1] 
                dashpot_damping_out = result['tensors'][2] 
                loss_val = result['loss_val'] 
                chamfer_loss_val = result.get('chamfer_loss_val', [0]) 
                track_loss_val = result.get('track_loss_val', [0]) 
                valid_frames = result['valid_frames'] 

                print("Forward time is : {}".format(time() - st))
                st = time()
                
                self.total_update += 1

                # Print params (using the last level evaluated for printing)
                print(f"\n{'='*60}")
                print(f"Global Epoch {epoch}")
                print(f"Spring Y iteration {i+1}/10")
                print(f"spring_Y grad Average: {gradients[0]['log_spring_Y'].mean().item()}")
                print(f"Spring_Y Average: {log_new_spring_Y.mean().item():.6f}")
                print(f"drag_damping grad Average: {gradients[0]['drag_damping'].mean().item()}")
                print(f"Drag_Damping: {drag_damping_out.item():.6f}")
                print(f"dashpot_damping grad Average: {gradients[0]['dashpot_damping'].mean().item()}")
                print(f"Dashpot_Damping: {dashpot_damping_out.item():.6f}")
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

                # Check if this is the global best result and save immediately
                current_loss = np.sum(total_loss_val)
                if current_loss < self.global_best_loss:
                    prev_best_loss = self.global_best_loss
                    self.global_best_loss = current_loss
                    self.global_best_epoch = epoch
                    self.global_best_iteration = i

                    # Save global best model checkpoint
                    best_ckpt_path = os.path.join(self.args.dump_dir, 'ckpts', f'global_best_epoch{epoch}_iter{i}')
                    torch.save(self.model.module.state_dict(), best_ckpt_path)

                    # 保存全局最优的包含所有层级的 mech_info 列表
                    self.global_best_mech_info = [info.copy() for info in mlvl_current_mech_info]
                    
                    best_mech_info_path = os.path.join(self.args.dump_dir, 'spring_mech_info', f'global_best_epoch{epoch}_iter{i}')
                    torch.save({'mech': self.global_best_mech_info}, best_mech_info_path)

                    # Save global best trajectory
                    best_traj_path = os.path.join(self.args.dump_dir, 'trajectories', f'global_best_trajectory.pkl')
                    logger.info(f"New global best loss {total_loss_val:.10f}. Saving checkpoint and trajectory...")
                    
                    # 传递所有层的下采样索引列表，以及 mlvl_masses 和 mlvl_edges
                    self.save_traj(mlvl_mech_info=self.global_best_mech_info, 
                                   save_path=best_traj_path, 
                                   compute_loss=True,
                                   gt_indices=self.m_ids,  # 传递所有层的下采样索引列表
                                   mlvl_masses=self.m_masses,  # 传递所有 level 的质点质量
                                   mlvl_edges=self.m_gs)       # 传递所有 level 的边拓扑

                    # Log to tensorboard
                    self.writer.add_scalar('Global_Best/loss', current_loss, self.total_update)
                    self.writer.add_scalar('Global_Best/epoch', epoch, self.total_update)
                    self.writer.add_scalar('Global_Best/iteration', i, self.total_update)

                    print(f"\n{'='*60}")
                    print(f"NEW GLOBAL BEST RESULT SAVED!")
                    print(f"Epoch: {epoch}, Iteration: {i+1}/10")
                    print(f"Loss: {current_loss:.10f}")
                    print(f"Previous best: {prev_best_loss:.10f}")
                    print(f"{'='*60}\n")

            # Update collide parameters - Multi-threaded parallel optimization
            # [修改] 使用多线程并行优化所有 stage 的 collision 参数
            if cfg.collision_learn:
                print(f"\n{'='*60}")
                print("Starting multi-threaded parallel collide parameters optimization...")
                print(f"{'='*60}\n")

                # [修改] 定义单 stage 的 collision 参数优化函数
                def optimize_single_stage(cur_stage, num_iters=5):
                    """
                    在单个 stage 上执行 collision 参数优化
                    返回优化过程中的平均损失
                    """
                    simulator = self.mlvl_simulators[cur_stage]
                    collide_optimizer = self.mlvl_collide_optimizer[cur_stage]
                    all_iter_losses = []
                    
                    for i in range(num_iters):
                        iter_losses = []
                        
                        # Reset position and velocity
                        simulator.set_init_state(
                            simulator.wp_init_vertices,
                            simulator.wp_init_velocities
                        )
                        
                        # Run simulation for all frames
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
                            
                            # Update collide parameters
                            collide_optimizer.step()
                            
                            # Accumulate loss
                            loss = wp.to_torch(simulator.loss, requires_grad=False)
                            iter_losses.append(loss.item())
                            
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
                        
                        avg_loss = np.mean(iter_losses)
                        all_iter_losses.append(avg_loss)
                        print(f"[Stage {cur_stage+1}] Collide iter {i+1}/{num_iters} - Avg Loss: {avg_loss:.10f}")
                    
                    return all_iter_losses
                
                # 并行执行所有 level 的 collision 优化
                num_threads = len(self.mlvl_simulators)
                level_results = {}
                
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    # 提交所有 level 的优化任务
                    future_to_level = {
                        executor.submit(optimize_single_stage, level, 5): level 
                        for level in range(len(self.mlvl_simulators))
                    }
                    
                    # 收集结果
                    for future in future_to_level:
                        level = future_to_level[future]
                        try:
                            losses = future.result()
                            level_results[level] = losses
                            final_avg_loss = np.mean(losses)
                            print(f"\n{'='*60}")
                            print(f"Level {level+1} completed - Final Avg Loss: {final_avg_loss:.10f}")
                            print(f"{'='*60}\n")
                        except Exception as e:
                            print(f"Level {level+1} optimization failed: {e}")
                            level_results[level] = None
                
                # 打印所有 level 的最终结果
                print(f"\n{'='*60}")
                print("Multi-threaded Collision Parameter Optimization Summary:")
                for level, losses in level_results.items():
                    if losses is not None:
                        print(f"  Level {level+1}: Final Avg Loss = {np.mean(losses):.10f}")
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
        # if mode == 'train':
        #     if epoch < self.args.warmup_epochs:
        #         self.warmup_scheduler.step()
        #     else:
        #         if self.optimizer.param_groups[0]['lr'] > 1e-6:
        #             self.scheduler.step()
        #             if self.optimizer.param_groups[0]['lr'] < 1e-6:
        #                 self.optimizer.param_groups[0]['lr'] = 1e-6
        # else:
        #     self.model.eval()

        # [修复]: 返回总 loss，以及包含所有层的拓扑结构和力学参数列表
        if mode == 'train':
            return mean_loss, mlvl_current_mech_info
        else:
            return mean_loss, None, None

    # ====================================================================
    # Train 函数：从头开始训练，每次都输出多层级下采样结果
    # 拓扑结构低频更新：每 k 步更新一次，中间保持结构不变
    # 通过向模型输入 prev_P 来保持下采样一致性
    # ====================================================================
    def train(self):
        # Initialize initial graph structure info before loop
        self._preproc_multi_infos(self.mdata)
        
        # 训练所有 epoch
        for epoch in range(self.epochs_per_stage):
            logger.info(f"--- Starting Epoch {epoch+1}/{self.epochs_per_stage} ---")
            
            # 运行 epoch（不再需要 stage_idx）
            self.run_epoch(epoch, mode='train')
            
            # 手动更新外部的总体进度条
            self.pbar.update(1)

        # [新增]: 导出所有层级的 OBJ 模型
        try:
            self.export_multi_level_obj()
        except Exception as e:
            logger.error(f"Failed to export OBJ models: {e}")
        
        # [新增]: 可视化每个 level 的真值结果和渲染结果
        try:
            self.visualize_all_levels()
        except Exception as e:
            logger.error(f"Failed to visualize levels: {e}")
        
        self.pbar.close()
    
    def visualize_all_levels(self, mlvl_mech_info=None, num_frames=None):
        """
        Visualize ground truth and rendered results for all levels as MP4 videos.
        
        Args:
            mlvl_mech_info: List of mechanical info for each level (optional, uses global_best_mech_info if not provided)
            num_frames: Number of frames to render (default: cfg.train_frame + cfg.test_frame)
        """
        logger.info("Starting visualization of all levels...")
        
        # Use global best mech info if not provided
        if mlvl_mech_info is None:
            mlvl_mech_info = self.global_best_mech_info
        
        # Call the visualization utility
        output_videos = visualize_level_results(
            trainer=self,
            mlvl_mech_info=mlvl_mech_info,
            output_dir=os.path.join(self.args.dump_dir, 'visualization'),
            num_frames=num_frames
        )
        
        logger.info(f"Visualization completed! Generated {len(output_videos)} videos:")
        for video_path in output_videos:
            logger.info(f"  - {video_path}")
        
        return output_videos
