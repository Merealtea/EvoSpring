from core_model import MLP, gumbel_softmax
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, LayerNorm, Softplus
import torch
from torch import nn
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_geometric.utils import (
    remove_self_loops,
    to_edge_index,
    to_torch_csr_tensor,
    coalesce,
)
from core_model import EvoMesh, WeightedEdgeConv, Unpool
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
    
class PhysicsAwareAttentionPooling(nn.Module):
    def __init__(self, in_features: int, num_heads: int = 1, k_eigenvectors: int = 8, 
                 lambda_reg: float = 0.1, sigma: float = 1.0):
        """
        in_features: 节点输入特征维度
        num_heads: 注意力头数
        k_eigenvectors: 取拉普拉斯矩阵的前 k 个非零特征向量作为位置编码
        lambda_reg: 聚类数量正则化权重 (修复: 新增初始化)
        sigma: 高斯核带宽，用于从空间坐标构建图拓扑 (修复: 新增初始化)
        """
        super().__init__()
        self.k_eigenvectors = k_eigenvectors
        self.lambda_reg = lambda_reg 
        self.sigma = sigma
        
        # 1. 位置编码的线性投影层
        self.pe_proj = nn.Linear(k_eigenvectors, in_features)
        
        # 2. PyTorch 原生 MultiheadAttention
        self.mha = nn.MultiheadAttention(
            embed_dim=in_features, 
            num_heads=num_heads, 
            batch_first=True
        )

    def compute_laplacian_from_pos(self, node_pos: torch.Tensor, adj_mat: torch.Tensor = None) -> torch.Tensor:
        """
        根据节点坐标动态计算物理拉普拉斯矩阵。
        支持传入 adj_mat 进行拓扑级别的强制稀疏化。
        """
        N = node_pos.shape[-2]
        D_pos = node_pos.shape[-1]
        device, dtype = node_pos.device, node_pos.dtype
        
        # 1. 确定性微扰，防止重合点导致特征值求解崩溃
        idx = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
        dim_idx = torch.arange(1, D_pos + 1, device=device, dtype=dtype).unsqueeze(0)
        deterministic_noise = torch.sin(idx * dim_idx) * 1e-5
        stable_pos = node_pos + deterministic_noise
        
        # 2. 计算空间距离并应用高斯核
        dist_matrix = torch.cdist(stable_pos, stable_pos, p=2.0)
        A = torch.exp(-(dist_matrix ** 2) / (2 * self.sigma ** 2))
        A = A * adj_mat.float()
        
        # 3. 对角线清零 (消除自环)
        A.diagonal(dim1=-2, dim2=-1).zero_()
        
        # 4. 计算度矩阵与拉普拉斯矩阵
        degrees = torch.sum(A, dim=-1)
        D = torch.diag_embed(degrees)
        L = D - A
        
        return L

    def compute_pe(self, L: torch.Tensor) -> torch.Tensor:
        """计算拉普拉斯位置编码 (完全兼容 Batch，显式提取 Top-K 最小非零特征向量)"""
        # 1. 求解特征值和特征向量
        # eigh 默认升序，eigenvalues: (..., N), eigenvectors: (..., N, N)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        
        # 2. 构造惩罚矩阵：将无效特征值 (<= eps) 替换为无穷大
        # 这样在寻找“最小”特征值时，它们会被自动排挤到最后，不会被取到
        eps = 1e-5
        safe_eigenvalues = torch.where(
            eigenvalues > eps, 
            eigenvalues, 
            torch.tensor(float('inf'), device=L.device, dtype=eigenvalues.dtype)
        )
        
        # 3. 显式获取前 k 个最小特征值的索引 (Amplitude 排序)
        k = self.k_eigenvectors
        N = L.shape[-1]
        actual_k = min(k, N)
        
        # largest=False 表示从小到大取，sorted=True 保证顺序绝对正确
        # topk_indices 形状: (..., actual_k)
        _, topk_indices = torch.topk(safe_eigenvalues, k=actual_k, dim=-1, largest=False, sorted=True)
        
        # 4. 维度自适应扩展与特征向量 Gather
        if L.dim() == 3:  # Batch 模式: L 的形状是 (B, N, N)
            # 将索引扩展为 (B, N, actual_k) 以匹配 eigenvectors (B, N, N)
            gather_indices = topk_indices.unsqueeze(1).expand(-1, N, -1)
            pe = torch.gather(eigenvectors, dim=-1, index=gather_indices)
            
            # 清理：如果某个图里有效特征值连 k 个都不到，把取到 inf 对应的特征向量强制清零
            valid_mask = torch.gather(safe_eigenvalues, dim=-1, index=topk_indices) < float('inf')
            pe = pe * valid_mask.unsqueeze(1).float() # valid_mask 扩展为 (B, 1, actual_k)
            
        else:  # 单图模式: L 的形状是 (N, N)
            gather_indices = topk_indices.unsqueeze(0).expand(N, -1)
            pe = torch.gather(eigenvectors, dim=-1, index=gather_indices)
            
            valid_mask = torch.gather(safe_eigenvalues, dim=-1, index=topk_indices) < float('inf')
            pe = pe * valid_mask.unsqueeze(0).float()
            
        # 5. 如果节点数 N < k，在最后一维 Padding 补齐到 k 维
        if actual_k < k:
            pad_size = k - actual_k
            pe = F.pad(pe, (0, pad_size), mode='constant', value=0.0)
            
        return pe
    
    def forward(self, x: torch.Tensor, node_pos: torch.Tensor, node_mass: torch.Tensor, node_type: torch.Tensor,
                gs: torch.Tensor, tau: float = 1.0, prev_P = None):
        """
        输入:
        x: (N, in_features) 节点特征
        node_pos: (N, D_pos) 节点空间坐标
        node_mass: (N,) 或 (N, 1) 节点质量
        node_type: (N,) 节点类型，里面的类型包含 4 类，其中第 4 类为控制点，我们现在不对控制点进行合并，只对其他点进行合并
        adj_mask: (N, N) 邻接掩码，只有 adj_mask[i,j]=1 时 i 和 j 才是邻居点，才可以合并
        tau: Gumbel-softmax 温度参数
        prev_P: 预先给定的投影矩阵，如果提供则直接使用，否则使用 gumbel softmax 计算
        """
        N = x.shape[1]
        control_node_type = 3

        # 从边索引 gs 构建邻接矩阵
        # gs shape: (2, num_edges), 其中 gs[0] 是源节点，gs[1] 是目标节点
        num_nodes = node_pos.shape[-2]
        adj_mask = torch.zeros(num_nodes, num_nodes, device=node_pos.device, dtype=torch.bool)
        if gs.shape[1] > 0:
            adj_mask[gs[0], gs[1]] = True
            # 使邻接矩阵对称 (无向图)
            adj_mask = adj_mask | adj_mask.T
        
        # 将对角线设为 False (移除自环)
        adj_mask.fill_diagonal_(False)

        # ==========================================
        # ★ Step 0: 动态构建物理矩阵 (修复) ★
        # ==========================================
        L = self.compute_laplacian_from_pos(node_pos, adj_mat=adj_mask)

        # --- Step 1 & 2: 提取特征与归一化 ---
        pe = self.compute_pe(L)
        x_pe = x + self.pe_proj(pe)
        
        # TODO(add attn mask here)
        attn_output, _ = self.mha(query=x_pe, key=x_pe, value=x_pe, need_weights=False)
        z = attn_output.squeeze(0)  
        z_norm = F.normalize(z, p=2, dim=-1)

        # --- Step 3: 距离矩阵计算 ---
        P_gramian = z_norm @ z_norm.transpose(-1, -2)
        diag_P = torch.diag(P_gramian).unsqueeze(1)
        D_sq = diag_P + diag_P.transpose(0, 1) - 2 * P_gramian
        D_mat_base = torch.sqrt(torch.clamp(D_sq, min=0.0) + 1e-8)

        # calculate adj matrix based on gs
        

        penalty_value = D_mat_base.max().detach() * 10.0
        D_mat = D_mat_base.masked_fill(~adj_mask, penalty_value)

        # ==========================================
        # ★ Step 4: Gumbel-Softmax 动态分配 + 直通估计器 ★
        # ==========================================
        # 无论是否提供 prev_P，都先计算 logits 和 soft version 用于梯度反向传播
        logits = -D_mat 
        
        # 构造 forbidden_mask 来控制合并规则
        forbidden_mask = torch.zeros_like(logits, dtype=torch.bool)
        
        # 规则 1：控制点不能分配给别人 (即控制点所在的行，除了自己，其他都是 True)
        is_control = (node_type == control_node_type)[0]  # 找到所有控制点，形状 (N,)
        forbidden_mask[is_control, :] = True
        
        # 规则 2：别人不能分配给控制点 (即控制点所在的列，除了自己，其他都是 True)
        forbidden_mask[:, is_control] = True
        
        # 规则 3：只有邻居点才可以合并 (adj_mask[i,j] == 0 表示不是邻居，不能合并)
        if adj_mask is not None:
            # adj_mask 为 0 的位置表示不是邻居，这些位置不能合并
            non_neighbor_mask = (adj_mask == 0)
            forbidden_mask = forbidden_mask | non_neighbor_mask
        
        # 规则 4：只有 node_type 相同的点才可以合并
        # 构造一个矩阵，其中 [i,j] 为 True 表示 node_type[i] != node_type[j]
        node_type_1d = node_type[0] if node_type.dim() > 1 else node_type  # (N,)
        type_diff_mask = node_type_1d.unsqueeze(0) != node_type_1d.unsqueeze(1)  # (N, N)
        forbidden_mask = forbidden_mask | type_diff_mask
        
        # 规则 5：允许所有点（包括控制点）分配给自己 (对角线始终允许)
        diag_indices = torch.arange(N, device=logits.device)
        forbidden_mask[diag_indices, diag_indices] = False 
        
        # 将不允许分配的路径 logits 设为极小值，Gumbel-softmax 采样时概率将趋近于 0
        logits = logits.masked_fill(forbidden_mask, -1e9)
        
        # 计算 soft version 用于梯度反向传播
        P_soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        
        if prev_P is not None:
            # 使用直通估计器：前向传播使用 prev_P，反向传播通过 P_soft 传递梯度到 logits
            # P_full = prev_P + (P_soft - P_soft).detach() 这种形式不对
            # 正确的直通估计器形式：P_full = P_soft + (prev_P - P_soft).detach()
            # 这样前向传播时 P_full = prev_P，反向传播时梯度通过 P_soft 传递
            P_full = P_soft + (prev_P - P_soft).detach()
        else:
            # 不使用 prev_P 时，使用标准的 hard gumbel softmax
            P_full = (P_soft == P_soft.max(dim=-1, keepdim=True)[0]).float()
        
        # 计算 r_soft 用于 loss 计算（使用 soft version 计算）
        r_soft = P_soft.max(dim=0)[0].sum()

        # --- Step 5: 提取降阶投影矩阵 P_proj ---
        active_mask = P_full.sum(dim=0) > 0
        P_proj = P_full[:, active_mask] # 形状：(N, r)
        dynamic_r = P_proj.shape[1]

        # ==========================================
        # ★ Step 6: 特征与位置的下采样 ★
        # ==========================================
        pos_hat = node_pos[:, active_mask]  
        cluster_sizes = P_proj.sum(dim=0, keepdim=True) 
        P_proj_normalized = P_proj / cluster_sizes.clamp(min=1e-9)
        x_hat = P_proj_normalized.transpose(-1, -2) @ x 

        A_orig = adj_mask.float()

        A_new = P_proj.transpose(-1, -2) @ A_orig @ P_proj
        A_new.fill_diagonal_(0.0)
        
        src, dst = torch.where(A_new > 0)
        gs = torch.stack([src, dst], dim=0) 

        # --- Step 8: 物理系统降阶与 Loss 计算 ---
        # 直接用 P_proj 的转置去乘质量向量
        # (..., r, N) @ (..., N, 1) -> (..., r, 1)
        node_mass_hat = node_mass @ P_proj
        node_type_hat = node_type[:, active_mask]

        # (修复) lambda_reg 现已通过 self 访问
        intra_cluster_loss = torch.trace(P_proj.transpose(-1, -2) @ D_mat @ P_proj)
        total_loss = intra_cluster_loss + self.lambda_reg * r_soft
        
        ids = torch.where(active_mask)[0].cpu().numpy().tolist()

        return ids, gs, x_hat, pos_hat, node_mass_hat, node_type_hat, total_loss, dynamic_r, P_full
    
class End2End_amp_base(MessagePassing):
    def __init__(self, latent_dim, hidden_layer, pos_dim, lagrangian, node_type_embedding_dim=8, mass_embedding_dim=8):
        super().__init__(aggr='add', flow='target_to_source')
        self.mlp_node_delta = MLP(2 * latent_dim, latent_dim, latent_dim, hidden_layer, True)
        # edge_info_in_len: 2 * latent_dim for node i and j
        # + 2 * pos_dim + 2 for lagrangian (dir and norm_w and norm_m)
        # + 20 for spring_encoded
        # + 2 * node_type_embedding_dim for node type embedding (i and j)
        # + 2 * mass_embedding_dim for node mass embedding (i and j)
        edge_info_in_len = 2 * latent_dim + 2 * pos_dim + 2 + 20 + 2 * node_type_embedding_dim + 2 * mass_embedding_dim if lagrangian else 2 * latent_dim + pos_dim + 1 + 20 + 2 * node_type_embedding_dim + 2 * mass_embedding_dim
        self.mlp_edge_info = MLP(edge_info_in_len, latent_dim, latent_dim, hidden_layer, True)
        self.mlp_edge_weight = Seq(*[MLP(latent_dim, latent_dim, 1, hidden_layer, False)])
        self.lagrangian = lagrangian
        self.pos_dim = pos_dim
        self.latent_dim = latent_dim
        
        # Node type embedding: 4 types (0: object, 1: surface, 2: interior, 3: controller)
        self.node_type_embedding = torch.nn.Embedding(4, node_type_embedding_dim)
        
        # Mass embedding: use MLP to embed scalar mass value
        self.mass_embedding = MLP(1, mass_embedding_dim, mass_embedding_dim, 2, True)

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

    def forward(self, x, g, pos, node_type=None, node_mass=None):
        """
        Args:
            x: node features, shape (T, N, F) or (N, F)
            g: edge index, tuple of (src, dst) tensors
            pos: node positions, shape (T, N, 3) or (N, 3)
            node_type: node types, shape (N,) or (T, N), values in {0, 1, 2, 3}
            node_mass: node masses, shape (N,) or (N, 1) or (T, N, 1)
        """
        i = g[0]
        j = g[1]
        if len(x.shape) == 3:
            T, _, _ = x.shape
            x_i = x[:, i]
            x_j = x[:, j]
        elif len(x.shape) == 2:
            x_i = x[i]
            x_j = x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")

        if len(pos.shape) == 3:
            pi = pos[:, i]
            pj = pos[:, j]
        elif len(pos.shape) == 2:
            pi = pos[i]
            pj = pos[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        dir = pi - pj  # in shape (T),N,dim

        if self.lagrangian:
            norm_w = torch.norm(dir[..., :self.pos_dim], dim=-1, keepdim=True)  # in shape (T),N,1
            norm_m = torch.norm(dir[..., self.pos_dim:], dim=-1, keepdim=True)  # in shape (T),N,1
            fiber = torch.cat([dir, norm_w, norm_m], dim=-1)
        else:
            norm = torch.norm(dir, dim=-1, keepdim=True)  # in shape (T),N,1
            fiber = torch.cat([dir, norm], dim=-1)

        # normalize spring_rest_length into [-1,1]
        spring_rest_length = fiber[..., 3]
        spring_min = spring_rest_length.min(dim=1, keepdim=True)[0]
        spring_max = spring_rest_length.max(dim=1, keepdim=True)[0]
        normalized_spring_rest_length = (2 * (spring_rest_length - spring_min) / (spring_max - spring_min) - 1)[0]

        # apply NeRF-style positional encoding to spring lengths
        spring_encoded = self.positional_encoding(normalized_spring_rest_length[:, None], num_freq_bands=10)[None]

        # Compute node type and mass embeddings and concatenate to fiber
        if node_type is not None and node_mass is not None:
            # Ensure node_type is 1D
            if node_type.dim() == 2:
                node_type = node_type.squeeze(0)
            # Ensure node_mass is 2D (N, 1)
            if node_mass.dim() == 1:
                node_mass = node_mass.unsqueeze(-1)
            elif node_mass.dim() == 3:
                node_mass = node_mass.squeeze(1)
            
            # Get embeddings for nodes i and j
            node_type_i = self.node_type_embedding(node_type[i])[None]  # (N, type_emb_dim)
            node_type_j = self.node_type_embedding(node_type[j])[None]  # (N, type_emb_dim)
            node_mass_i = self.mass_embedding(node_mass[:, i, None]) # (N, mass_emb_dim)
            node_mass_j = self.mass_embedding(node_mass[:, j, None])  # (N, mass_emb_dim)
            
            # Concatenate embeddings to fiber
            fiber = torch.cat([fiber, node_type_i, node_type_j, node_mass_i, node_mass_j], dim=-1)

        if len(x.shape) == 3 and len(pos.shape) == 2:
            tmp = torch.cat([fiber.unsqueeze(0).repeat(T, 1, 1), x_i, x_j], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j, spring_encoded], dim=-1)
        
        edge_embedding = self.mlp_edge_info(tmp)
        edge_weight = self.mlp_edge_weight(edge_embedding)
        edge_weight = scatter_softmax(edge_weight, j, dim=-2)
        edge_embedding = edge_embedding * edge_weight

        aggr_out = scatter(edge_embedding, j, dim=-2, dim_size=x.shape[-2], reduce="sum")
        
        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_node_delta(tmp) + x, edge_weight, edge_embedding

class End2EndReduction(EvoMesh):
    def __init__(self, l_n, pre_l_n, bottom_ln, ld, hidden_layer, pos_dim, lagrangian, enhance=True, agg_conv_pos=False, edge_set_num=1,
                 transformer_hidden_dim=128, transformer_num_layers=2, pos_encoding_dim=30,
                 node_type_embedding_dim=8, mass_embedding_dim=8,
                 pooling_num_heads=1, k_eigenvectors=8, pooling_lambda_reg=0.1, pooling_sigma=1.0):
        super(EvoMesh, self).__init__()
        self.down_gmps = nn.ModuleList()
        self.up_gmps = nn.ModuleList()
        self.downpools = nn.ModuleList()  # 使用 PhysicsAwareAttentionPooling 进行下采样
        self.unpools = nn.ModuleList()
        self.l_n = l_n
        self.edge_conv = WeightedEdgeConv()
        self.pre_l_n = pre_l_n
        self.enhance = enhance
        self.agg_conv_pos = agg_conv_pos
        self.bottom_ln = bottom_ln
        self.node_type_embedding_dim = node_type_embedding_dim
        self.mass_embedding_dim = mass_embedding_dim
        self.bottom_gmp = nn.ModuleList(
            End2End_amp_base(ld, hidden_layer, pos_dim, lagrangian, node_type_embedding_dim, mass_embedding_dim) 
            for _ in range(self.bottom_ln)
        )

        for _ in range(self.l_n):
            self.down_gmps.append(
                End2End_amp_base(ld, hidden_layer, pos_dim, lagrangian, node_type_embedding_dim, mass_embedding_dim)
            )
            self.downpools.append(
                PhysicsAwareAttentionPooling(ld, pooling_num_heads, k_eigenvectors, pooling_lambda_reg, pooling_sigma)
            )
            self.up_gmps.append(
                End2End_amp_base(ld, hidden_layer, pos_dim, lagrangian, node_type_embedding_dim, mass_embedding_dim)
            )
            self.unpools.append(Unpool())
        self.esn = edge_set_num
        self.lagrangian = lagrangian

        # Transformer components for encoding reduced node features
        self.ld = ld
        self.pos_dim = pos_dim
        self.pos_encoding_dim = pos_encoding_dim
        
        # NeRF-style positional encoding for 3D positions
        # Output dim: 3 * 2 * num_freq_bands = pos_encoding_dim
        self.num_freq_bands = pos_encoding_dim // 6
        
        # Position encoding to transformer hidden dim projection
        self.pos_to_transformer = nn.Linear(self.pos_encoding_dim, transformer_hidden_dim)
        
        # Node feature to transformer hidden dim projection
        self.node_to_transformer = nn.Linear(ld, transformer_hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_dim,
            nhead=max(1, transformer_hidden_dim // 32),
            dim_feedforward=transformer_hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
        
        # Transformer output projection back to original feature dim
        self.transformer_output_proj = nn.Linear(transformer_hidden_dim, ld)

        self.edge_weight = MLP(2*ld, ld, ld, hidden_layer, True)
        # Node type embedding: 4 types (0: object, 1: surface, 2: interior, 3: controller)
        self.node_type_embedding = torch.nn.Embedding(4, ld)

    def positional_encoding_3d(self, positions: torch.Tensor) -> torch.Tensor:
        """
        NeRF-style positional encoding for 3D positions
        Args:
            positions: (B, N, 3) or (N, 3) 3D coordinates (normalized to [-1, 1])
        Returns:
            encoded: (B, N, pos_encoding_dim) or (N, pos_encoding_dim)
        """
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        B, N, D = positions.shape
        device, dtype = positions.device, positions.dtype
        
        # Frequency bands: 2^0, 2^1, ..., 2^(num_freq_bands-1)
        freq_bands = 2.0 ** torch.arange(self.num_freq_bands, dtype=dtype, device=device)  # [L]
        
        # Expand for broadcasting: positions [B, N, 3, 1], freq_bands [1, 1, 1, L]
        pos_expanded = positions.unsqueeze(-1)  # [B, N, 3, 1]
        freq_expanded = freq_bands.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, L]
        
        # Compute scaled positions: [B, N, 3, L]
        scaled_pos = math.pi * pos_expanded * freq_expanded
        
        # Apply sin and cos: [B, N, 3, L]
        sin_encoded = torch.sin(scaled_pos)
        cos_encoded = torch.cos(scaled_pos)
        
        # Stack and reshape: [B, N, 3, L, 2] -> [B, N, 3*L*2]
        encoded = torch.stack([sin_encoded, cos_encoded], dim=-1)  # [B, N, 3, L, 2]
        encoded = encoded.reshape(B, N, -1)  # [B, N, 3*L*2]
        
        if squeeze_output:
            encoded = encoded.squeeze(0)
        
        return encoded

    def _pool_tensor(self, tensor, node_mask):
        tensor = node_mask[None, :, None].float() * tensor
        return tensor[:, node_mask]

    def forward(self, node_in, m_ids, m_gs, m_pos, m_mass, m_type, temp=0.1, m_proj=None):
        """
        使用 PhysicsAwareAttentionPooling 进行内部下采样的前向传播。
        流程：先 pool -> down_gmps -> edge conv -> PhysicsAwareAttentionPooling 获取下采样的 gs, proj 等
        
        Args:
            node_in: 节点输入特征，shape (N, F) 或 (T, N, F)
            m_ids: 每层节点索引列表
            m_gs: 每层边索引列表
            m_proj: 每层投影矩阵列表
            m_node_pos: 每层节点位置列表
            m_node_mass: 每层节点质量列表
            m_node_type: 每层节点类型列表
            temp: Gumbel-softmax 温度参数
            adj_mask: 邻接掩码，shape (N, N)，可选
        """
        down_ids = []
        down_gs = []
        down_outs = []
        down_ps = []
        down_mass = []
        down_type = []
        cts = []
        pooling_losses = []
        down_proj_list = []  # 存储 PhysicsAwareAttentionPooling 生成的投影矩阵
        
        # 确保输入有 batch 维度
        if node_in.dim() == 2:
            node_in = node_in[None, ...]  # (1, N, F)
        
        
        # down pass
        ds_edge_embedding = []
        
        gs = m_gs[0]
        pos = m_pos[0][None]
        node_mass = m_mass[0][None]
        node_type = m_type[0][None]

        down_gs = [gs]
        down_ps = [pos]
        down_mass = [node_mass]
        down_type = [node_type]
        
        for i in range(self.l_n):
            # Step 2: down_gmps - 使用 AMP 处理边信息
            node_in, ew, edge_embedding =\
                self.down_gmps[i](node_in, gs, pos, 
                                    node_type=node_type, 
                                    node_mass=node_mass)
            
            # Step 3: edge conv - 使用边卷积更新节点特征
            node_in = self.edge_conv(node_in, gs, ew)

            down_outs.append(node_in)
            ds_edge_embedding.append(edge_embedding)
            cts.append(ew)
            
            # Step 4: PhysicsAwareAttentionPooling - 获取下采样的 gs, proj 等
            if not m_proj:
                idx, gs, node_in, pos, node_mass, node_type, pool_loss, dynamic_r, P_proj =\
                    self.downpools[i](
                        node_in, pos, node_mass, node_type, gs, 
                        tau=temp)
            else:
                idx, gs, node_in, pos, node_mass, node_type, pool_loss, dynamic_r, P_proj =\
                    self.downpools[i](
                        node_in, pos, node_mass, node_type, gs, 
                        temp, m_proj[i])
                
            down_ids.append(idx)
            down_gs.append(gs)
                
            pooling_losses.append(pool_loss)
            down_ps.append(pos)
            down_mass.append(node_mass)
            down_type.append(node_type)
            down_proj_list.append(P_proj)
            
        # Prepare node_type and node_mass for bottom layer
        for l in range(self.bottom_ln):
            node_in, ew, _ = self.down_gmps[i](node_in, 
                                               down_gs[self.l_n], 
                                               down_ps[self.l_n], 
                                               node_type=node_type, 
                                               node_mass=node_mass)

        # up pass
        mlvl_edge_embedding = []

        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            # 使用 downpools 生成的 gs 进行 up pass
            gs = down_gs[up_idx] 

            node_in = self.unpools[i](node_in, 
                                      down_outs[up_idx].shape[-2], 
                                      down_ids[up_idx])
            
            node_in= self.edge_conv(node_in, gs, cts[up_idx], aggragating=False)
            # Prepare node_type and node_mass for up pass
            node_type_up = down_type[up_idx]
            node_mass_up = down_mass[up_idx]

            node_in, _, edge_embedding = self.up_gmps[i](
                node_in, gs, down_ps[up_idx], 
                node_type=node_type_up, node_mass=node_mass_up
            )
                        
            node_in = node_in + down_outs[up_idx]

            edge_embedding = edge_embedding + ds_edge_embedding[up_idx]            
            num_edge = edge_embedding.shape[1]

            # Average symmetric edges
            edge_feature = edge_embedding[:, :num_edge//2] + edge_embedding[:, num_edge//2:]
            mlvl_edge_embedding.insert(0, edge_feature)
        
        down_ids.insert(0, [x for x in range(node_in.shape[1])])
        # Return downsample results for warp construction
        downsample_results = {
            'down_ids': down_ids,
            'down_gs': down_gs,
            'down_proj': down_proj_list,
            'down_ps': down_ps,
            'down_mass': down_mass,
            'down_type': down_type,
        }
        
        return mlvl_edge_embedding, pooling_losses, downsample_results
