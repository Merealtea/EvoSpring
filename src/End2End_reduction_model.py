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
                adj_mask: torch.Tensor = None, tau: float = 1.0):
        """
        输入:
        x: (N, in_features) 节点特征
        node_pos: (N, D_pos) 节点空间坐标
        node_mass: (N,) 或 (N, 1) 节点质量
        node_type: (N,) 节点类型，里面的类型包含4类，其中第4类为控制点，我们现在不对控制点进行合并，只对其他点进行合并
        """
        N = x.shape[1]
        control_node_type = 3

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

        if adj_mask is not None:
            penalty_value = D_mat_base.max().detach() * 10.0
            D_mat = D_mat_base.masked_fill(~adj_mask, penalty_value)
        else:
            D_mat = D_mat_base

        # --- Step 4: Gumbel-Softmax 动态分配 ---
        logits = -D_mat 
        # [新增]: 构造掩码，隔离控制点
        is_control = (node_type == control_node_type)[0]  # 找到所有控制点，形状 (N,)
        forbidden_mask = torch.zeros_like(logits, dtype=torch.bool)
        
        # 规则 1：控制点不能分配给别人 (即控制点所在的行，除了自己，其他都是 True)
        forbidden_mask[is_control, :] = True
        # 规则 2：别人不能分配给控制点 (即控制点所在的列，除了自己，其他都是 True)
        forbidden_mask[:, is_control] = True
        # 规则 3：允许所有点（包括控制点）分配给自己
        diag_indices = torch.arange(N, device=logits.device)
        forbidden_mask[diag_indices, diag_indices] = False 
        
        # 将不允许分配的路径 logits 设为极小值，Gumbel-softmax 采样时概率将趋近于 0
        logits = logits.masked_fill(forbidden_mask, -1e9)
        P_full = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)

        # --- Step 5: 提取降阶投影矩阵 P_proj ---
        r_soft = P_full.max(dim=0)[0].sum()
        active_mask = P_full.sum(dim=0) > 0
        P_proj = P_full[:, active_mask] # 形状: (N, r)
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
        node_mass_hat = P_proj.transpose(-1, -2) @ node_mass
        node_type_hat = node_type[:, active_mask]

        # (修复) lambda_reg 现已通过 self 访问
        intra_cluster_loss = torch.trace(P_proj.transpose(-1, -2) @ D_mat @ P_proj)
        total_loss = intra_cluster_loss + self.lambda_reg * r_soft

        idx = torch.where(active_mask)[0].cpu().numpy().tolist()

        return idx, gs, x_hat, pos_hat, node_mass_hat, node_type_hat, total_loss, dynamic_r
    
class End2End_amp_base(MessagePassing):
    def __init__(self, latent_dim, hidden_layer, pos_dim, lagrangian):
        super().__init__(aggr='add', flow='target_to_source')
        self.mlp_node_delta = MLP(2 * latent_dim, latent_dim, latent_dim, hidden_layer, True)
        edge_info_in_len = 2 * latent_dim + 2 * pos_dim + 2 + 2 if lagrangian else 2 * latent_dim + pos_dim + 1
        self.mlp_edge_info = MLP(edge_info_in_len, latent_dim, latent_dim, hidden_layer, True)
        self.mlp_edge_weight = Seq(*[MLP(latent_dim, latent_dim, 1, hidden_layer, False)])
        self.lagrangian = lagrangian
        self.pos_dim = pos_dim
        self.latent_dim = latent_dim

    def forward(self, x, g, pos):
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
        
        if len(x.shape) == 3 and len(pos.shape) == 2:
            tmp = torch.cat([fiber.unsqueeze(0).repeat(T, 1, 1), x_i, x_j], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j], dim=-1)
        
        edge_embedding = self.mlp_edge_info(tmp)
        edge_weight = self.mlp_edge_weight(edge_embedding)
        edge_weight = scatter_softmax(edge_weight, j, dim=-2)
        edge_embedding = edge_embedding * edge_weight

        aggr_out = scatter(edge_embedding, j, dim=-2, dim_size=x.shape[-2], reduce="sum")
        
        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_node_delta(tmp) + x, edge_weight, edge_embedding

class End2EndReduction(EvoMesh):
    def __init__(self, l_n, pre_l_n, bottom_ln, ld, hidden_layer, pos_dim, lagrangian, enhance=True, agg_conv_pos=False, edge_set_num=1):
        super(EvoMesh, self).__init__()
        self.down_gmps = nn.ModuleList()
        self.up_gmps = nn.ModuleList()
        self.downpools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = l_n
        self.edge_conv = WeightedEdgeConv()
        self.pre_l_n = pre_l_n
        self.enhance = enhance
        self.agg_conv_pos = agg_conv_pos
        self.bottom_ln = bottom_ln
        self.bottom_gmp = nn.ModuleList(End2End_amp_base(ld, hidden_layer, pos_dim, lagrangian) for _ in range(self.bottom_ln))

        for _ in range(self.l_n):
            self.down_gmps.append(End2End_amp_base(ld, hidden_layer, pos_dim, lagrangian))
            self.downpools.append(PhysicsAwareAttentionPooling(ld))
            self.up_gmps.append(End2End_amp_base(ld, hidden_layer, pos_dim, lagrangian))
            self.unpools.append(Unpool())
        self.esn = edge_set_num
        self.lagrangian = lagrangian

        # Transformer components for attention map calculation
        self.attn_nhead = 8
        self.ld = ld
        self.pos_dim = pos_dim

        # Position encoding generator
        self.pos_encoder = nn.Linear(pos_dim, ld)

        self.edge_weight = MLP(2*ld, ld, ld, hidden_layer, True)

    def pool_edge(self, g, idx, num_nodes, num_orignal_edge):
        idx = idx.to(torch.long)
        idx_new_valid = torch.arange(len(idx), dtype=torch.long, device=g.device)
        idx_new_all = -1 * torch.ones(num_nodes, dtype=torch.long, device=g.device)
        idx_new_all[idx] = idx_new_valid
        new_g = -1 * torch.ones_like(g, dtype=torch.long, device=g.device)
        new_g[0] = idx_new_all[g[0]]
        new_g[1] = idx_new_all[g[1]]

        # new_g[:, :num_orignal_edge] is the original edge
        both_valid = (new_g[0] >= 0) & (new_g[1] >= 0)
        e_idx = torch.where(both_valid)[0]

        original_valid = e_idx[e_idx < num_orignal_edge]
        new_g = new_g[:, e_idx]

        new_edge_parent = torch.full((new_g.shape[1], 1), -1, dtype=torch.long, device=g.device)
        new_edge_parent[:len(original_valid)] = torch.arange(num_orignal_edge, dtype=torch.long, device=g.device).unsqueeze(-1)[original_valid]
        return new_g, new_edge_parent

    def forward(self, node_in, m_ids, m_gs, pos, node_mass, node_type, temp=0.1):
        # node_in is in shape of (T), N, F
        # if edge_set_num>1, then m_g is in shape: Level,(Set),2,Edges, the 0th Set is main/material graph
        # pos is in (T),N,D

        down_outs = []
        down_ps = []
        cts = []

        # w = pos.new_ones((pos.shape[-2], 1)) if weights is None else weights
        node_in = node_in[None, ...]
        pos = pos[None, ...]
        node_type = node_type[None, ...]

        # down pass
        l_n = self.l_n 
        num_nodes_list = []
        ds_edge_embedding = []
        ds_node_mass = []
        ds_node_type = []

        for i in range(l_n):
            num_nodes = node_in.shape[-2] #.shape[0]
            num_nodes_list.append(num_nodes)

            ds_node_mass.append(node_mass)
            ds_node_type.append(node_type)
            
            gs = m_gs[i]
            # 2. 初始化全 False 的邻接矩阵 (False 暂时代指没有边)
            adj_mat = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=node_in.device)
            
            # 3. 填入存在的边 (src -> dst 和 dst -> src 保证对称)
            src, dst = gs[0], gs[1]
            adj_mat[src, dst] = True
            adj_mat[dst, src] = True
            
            node_in, ew, edge_embedding = self.down_gmps[i](node_in, gs, pos)
            if i == 0 and self.lagrangian:
                node_in, ew = self.down_gmps[i](node_in, gs, pos)

            ds_edge_embedding.append(edge_embedding)

            # record the info
            down_outs.append(node_in)
            down_ps.append(pos)
            # inter-level fusion
            tmp_g = gs
            node_in = self.edge_conv(node_in, tmp_g, ew)

            cts.append(ew)

            # add merge here
            idx, gs, node_in, pos, node_mass, node_type, auxiliary_loss, dynamic_r \
                = self.downpools[i](node_in, pos, node_mass, node_type, adj_mat, temp)
            
            m_gs.append(gs)
            m_ids.append(idx)
            
        
        for l in range(self.bottom_ln):
            node_in, ew, _ = self.bottom_gmp[l](node_in, m_gs[l_n], pos)
            if self.lagrangian and l == 0:
                node_in, ew, _ = self.bottom_gmp[l](node_in, m_gs[l_n], pos)

        # up pass
        mlvl_edge_embedding = []
        m_gs_out = []
        for i in range(l_n):
            up_idx = l_n - i - 1
            g, idx = m_gs[up_idx], m_ids[up_idx]
            try:
                node_in = self.unpools[i](node_in, down_outs[up_idx].shape[-2], idx)
            except:
                import pdb; pdb.set_trace()
            tmp_g = g[0] if self.esn > 1 else g
            node_in= self.edge_conv(node_in, tmp_g, cts[up_idx], aggragating=False)
            node_in, ew_u, edge_embedding = self.up_gmps[i](node_in, g, down_ps[up_idx])
     
            # if up_idx == 0 and self.lagrangian:
            #     node_in, ew_u = self.up_gmps[i](node_in, edge_mech_in, g, down_ps[up_idx])
            node_in = node_in + down_outs[up_idx]

            edge_embedding = edge_embedding + ds_edge_embedding[up_idx]            
            num_edge = edge_embedding.shape[1]

            # Average symmetric edges
            edge_feature = edge_embedding[:, :num_edge//2] + edge_embedding[:, num_edge//2:]
            m_gs_out.insert(0, g[:, :num_edge // 2])
            mlvl_edge_embedding.insert(0, edge_feature)
        
        # print(num_nodes_list)
        m_ids.insert(0, [i for i in range(num_nodes_list[0])])
        m_ids.pop(-1)
        return mlvl_edge_embedding, ds_node_mass, ds_node_type, m_gs_out, m_ids