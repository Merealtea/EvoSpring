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

class spring_mass_amp(MessagePassing):
    def __init__(self, latent_dim, hidden_layer, pos_dim, lagrangian):
        super().__init__(aggr='add', flow='target_to_source')
        self.mlp_node_delta = MLP(2 * latent_dim, latent_dim, latent_dim, hidden_layer, True)
        # egde info : 2 * latent_dim for node i and j
        # + 2 * pos_dim +2 for lagrangian (dir and norm_w and norm_m)
        # + 2 * pos_dim for vel_diff
        # + 3 for init_length, spring_Y and dashpot_damping
        edge_info_in_len = 2 * latent_dim + 2 * pos_dim + 2 * pos_dim + 2 + 3 if lagrangian else 2 * latent_dim + pos_dim + pos_dim + 3 + 6
        self.weightnet = True
        if self.weightnet:
            self.mlp_edge_info = MLP(edge_info_in_len, latent_dim, latent_dim, hidden_layer, True)
            self.mlp_edge_weight = MLP(latent_dim, latent_dim, 1, hidden_layer, False)
        else:
            self.mlp_edge_info = MLP(edge_info_in_len, latent_dim, latent_dim + 1, hidden_layer, False)
        self.mlp_gumbel = Seq(*[MLP(2 * latent_dim, latent_dim, 2, hidden_layer, False)])
        self.lagrangian = lagrangian
        self.pos_dim = pos_dim
        self.latent_dim = latent_dim

    def forward(self, x, edge_mech_input, g, pos, vel, temp):
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
            vi = vel[:, i]
            vj = vel[:, j]
        elif len(pos.shape) == 2:
            pi = pos[i]
            pj = pos[j]
            vi = vel[i]
            vj = vel[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        dir = pi - pj  # in shape (T),N,dim
        vel_diff = vi - vj
        

        if self.lagrangian:
            norm_w = torch.norm(dir[..., :self.pos_dim], dim=-1, keepdim=True)  # in shape (T),N,1
            norm_m = torch.norm(dir[..., self.pos_dim:], dim=-1, keepdim=True)  # in shape (T),N,1
            fiber = torch.cat([dir, vel_diff, norm_w, norm_m], dim=-1)
        else:
            norm = torch.norm(dir, dim=-1, keepdim=True)  # in shape (T),N,1
            fiber = torch.cat([dir, vel_diff, norm, edge_mech_input], dim=-1)
        
        if len(x.shape) == 3 and len(pos.shape) == 2:
            tmp = torch.cat([fiber.unsqueeze(0).repeat(T, 1, 1), x_i, x_j, edge_mech_input], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j], dim=-1)
        
        edge_embedding = self.mlp_edge_info(tmp)
        if self.weightnet:
            edge_weight = self.mlp_edge_weight(edge_embedding)
        else:
            edge_embedding, edge_weight = edge_embedding[..., :-1], edge_embedding[..., [-1]]
        edge_weight = scatter_softmax(edge_weight, j, dim=-2)

        edge_embedding = edge_embedding * edge_weight

        aggr_out = scatter(edge_embedding, j, dim=-2, dim_size=x.shape[-2], reduce="sum")

        tmp = torch.cat([x, aggr_out], dim=-1)

        logits = self.mlp_gumbel(tmp)
        logits = torch.mean(logits, axis=0)
        
        hard = True
        y_soft, y_hard = gumbel_softmax(logits, temp, hard=hard)  # tau越小越接近one-hot

        return self.mlp_node_delta(tmp) + x, edge_weight, y_hard
    
class spring_mass_amp_base(MessagePassing):
    def __init__(self, latent_dim, hidden_layer, pos_dim, lagrangian):
        super().__init__(aggr='add', flow='target_to_source')
        self.mlp_node_delta = MLP(2 * latent_dim, latent_dim, latent_dim, hidden_layer, True)
        edge_info_in_len = 2 * latent_dim + 2 * pos_dim + 2 + 3 if lagrangian else 2 * latent_dim + 2 * pos_dim + 3 + 6
        self.mlp_edge_info = MLP(edge_info_in_len, latent_dim, latent_dim, hidden_layer, True)
        self.mlp_edge_weight = Seq(*[MLP(latent_dim, latent_dim, 1, hidden_layer, False)])
        self.lagrangian = lagrangian
        self.pos_dim = pos_dim
        self.latent_dim = latent_dim

    def forward(self, x, edge_mech_info, g, pos, vel):
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
            vi = vel[:, i]
            vj = vel[:, j]
        elif len(pos.shape) == 2:
            pi = pos[i]
            pj = pos[j]
            vi = vel[i]
            vj = vel[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        dir = pi - pj  # in shape (T),N,dim
        vel_diff = vi - vj
        
        if self.lagrangian:
            norm_w = torch.norm(dir[..., :self.pos_dim], dim=-1, keepdim=True)  # in shape (T),N,1
            norm_m = torch.norm(dir[..., self.pos_dim:], dim=-1, keepdim=True)  # in shape (T),N,1
            fiber = torch.cat([dir, vel_diff, norm_w, norm_m], dim=-1)
        else:
            norm = torch.norm(dir, dim=-1, keepdim=True)  # in shape (T),N,1
            fiber = torch.cat([dir, vel_diff, norm, edge_mech_info], dim=-1)

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

class SpringMassEvoMesh(EvoMesh):
    def __init__(self, l_n, pre_l_n, bottom_ln, ld, hidden_layer, pos_dim, lagrangian, enhance=True, agg_conv_pos=False, edge_set_num=1):
        super(EvoMesh, self).__init__()
        self.down_gmps = nn.ModuleList()
        self.up_gmps = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = l_n
        self.edge_conv = WeightedEdgeConv()
        self.pre_l_n = pre_l_n
        self.enhance = enhance
        self.agg_conv_pos = agg_conv_pos
        self.bottom_ln = bottom_ln
        self.bottom_gmp = nn.ModuleList(spring_mass_amp_base(ld, hidden_layer, pos_dim, lagrangian) for _ in range(self.bottom_ln))
        for _ in range(self.l_n):
            if _ < self.pre_l_n:
                self.down_gmps.append(spring_mass_amp_base(ld, hidden_layer, pos_dim, lagrangian))
            else:
                self.down_gmps.append(spring_mass_amp(ld, hidden_layer, pos_dim, lagrangian))
            self.up_gmps.append(spring_mass_amp_base(ld, hidden_layer, pos_dim, lagrangian))
            self.unpools.append(Unpool())
        self.esn = edge_set_num
        self.lagrangian = lagrangian
    
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

    def forward(self, node_in, edge_in, mm_ids, mm_gs, mm_gs_parent, pos, vel, temp=0.1, weights=None):
        # node_in is in shape of (T), N, F
        # if edge_set_num>1, then m_g is in shape: Level,(Set),2,Edges, the 0th Set is main/material graph
        # pos is in (T),N,D

        down_outs = []
        down_ps = []
        multi_level_spring_force_info = []
        cts = []
        
        # w = pos.new_ones((pos.shape[-2], 1)) if weights is None else weights

        # # compute force
        # spring_force = 
        
        # mm_ids is node kept in current layer
        # mm_gs is the edge connection
        # mm_gs_parent is the parent edge idx for current edge 
        m_ids = mm_ids[:self.pre_l_n]
        m_gs = mm_gs[:self.pre_l_n + 1]
        m_gs_parent = mm_gs_parent[:self.pre_l_n]

        # down pass
        l_n = self.l_n 
        num_nodes_list = []
        for i in range(l_n):
            num_nodes = node_in.shape[-2] if i == 0 else len(m_ids[i-1]) #.shape[0]
            num_nodes_list.append(num_nodes)
            # # We don't need node self loop right now
            # if self.esn > 1:
            #     gs = []
            #     gs_main, _ = add_remaining_self_loops(m_gs[i][0]) 
            #     gs_cont, _ = add_remaining_self_loops(m_gs[i][1]) 
            #     gs = [gs_main, gs_cont]
            # else:
            #     gs, _ = add_remaining_self_loops(m_gs[i]) 

            gs = m_gs[i]
            multi_level_spring_force_info.append(edge_in)

            if i < self.pre_l_n:
                node_in, ew, _ = self.down_gmps[i](node_in, edge_in, gs, pos, vel)
                if i == 0 and self.lagrangian:
                    node_in, ew = self.down_gmps[i](node_in, edge_in, gs, pos, vel)
                y_hard = None
            else:
                # deeper downsample
                # dowmsample more to make the network more compact
                node_in, ew, y_hard = self.down_gmps[i](node_in, edge_in, gs, pos, vel, temp)
                if i == 0 and self.lagrangian:
                    node_in, ew, y_hard = self.down_gmps[i](node_in, edge_in, gs, pos, vel, temp)
                N = node_in.shape[1]
                edge_index = m_gs[i]
                if self.enhance:
                    adj = to_torch_csr_tensor(edge_index, size=(N, N))
                    edge_index2, _ = to_edge_index(adj @ adj)
                    # edge_index2, _ = remove_self_loops(edge_index2)
                    edge_index2 = torch.cat([edge_index, edge_index2], dim=1)
                else:
                    edge_index2 = edge_index
                
                m_idx = (y_hard[..., 0] == 1).nonzero().unique()
                num_original_edge = edge_index.shape[1]

                g, gs_parent_new = self.pool_edge(edge_index2, m_idx, num_nodes, num_original_edge)
                g, gs_parent_new = coalesce(g, gs_parent_new, num_nodes=len(m_idx), reduce='max')

                m_gs.append(g)
                m_ids.append(m_idx)
                m_gs_parent.append(gs_parent_new)
                assert len(m_ids) == i + 1
            # record the info
            down_outs.append(node_in)
            down_ps.append(pos)
            # inter-level fusion
            tmp_g = gs
            node_in = self.edge_conv(node_in, tmp_g, ew)
            if self.agg_conv_pos:
                pos = self.edge_conv(pos, tmp_g, ew)
            cts.append(ew)
            # pooling
            node_in = self._pool_tensor(node_in, m_ids, y_hard, i, i < self.pre_l_n)
            pos = self._pool_tensor(pos, m_ids, y_hard, i, i < self.pre_l_n)

            # Downsample edge_mech_in using gs_parent
            gs_parent = m_gs_parent[i]
            parent_edge_1 = edge_in[:, gs_parent[:, 0], :]  # T×M×2
            mask = gs_parent[:, 0] != -1  # M
            edge_in = parent_edge_1.clone()
            # TODO(cxy) add later
            # if mask.any():
            #     edge_in[:, mask, :] = torch.mean(edge_in, dim=1)

        multi_level_spring_force_info.append(edge_in)
        
        for l in range(self.bottom_ln):
            node_in, ew, _ = self.bottom_gmp[l](node_in, edge_in, m_gs[l_n], pos, vel)
            if self.lagrangian and l == 0:
                node_in, ew, _ = self.bottom_gmp[l](node_in, edge_in, m_gs[l_n], pos, vel)

        # up pass
        for i in range(l_n):
            up_idx = l_n - i - 1
            g, idx, edge_in = m_gs[up_idx], m_ids[up_idx], multi_level_spring_force_info[up_idx]
            # if self.esn > 1:
            #     g_main, _ = add_remaining_self_loops(g[0])
            #     g_cont, _ = add_remaining_self_loops(g[1]) 
            #     g = [g_main, g_cont]
            # else:
            #     g, _ = add_remaining_self_loops(g)
            
            node_in = self.unpools[i](node_in, down_outs[up_idx].shape[-2], idx)
            tmp_g = g[0] if self.esn > 1 else g
            node_in= self.edge_conv(node_in, tmp_g, cts[up_idx], aggragating=False)
            node_in, ew_u, edge_embedding = self.up_gmps[i](node_in, edge_in, g, down_ps[up_idx], vel)
     
            # if up_idx == 0 and self.lagrangian:
            #     node_in, ew_u = self.up_gmps[i](node_in, temporal_edge_mech_in, g, down_ps[up_idx], vel)
            node_in = node_in + down_outs[up_idx]
        # print(num_nodes_list)
        return node_in, edge_embedding
    
class TemporalFeatureAggregation(nn.Module):
    """
        Edge input shape is T * N * C
        Now I want to compress the feature along the T dimension with conv
        The final output shape should be N * C
    """
    def __init__(self, in_channels, hidden_channels=None, num_layers=2, kernel_size=3):
        """
        Args:
            in_channels: Number of input channels (C)
            hidden_channels: Number of hidden channels for conv layers. If None, uses in_channels
            num_layers: Number of temporal conv layers (default: 2)
            kernel_size: Kernel size for temporal convolution (default: 3)
        """
        super().__init__()

        if hidden_channels is None:
            hidden_channels = in_channels

        # Build temporal convolution layers
        # Input will be reshaped to (N, C, T) for Conv1d
        conv_layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = hidden_channels if i < num_layers - 1 else in_channels
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(out_ch) if i < num_layers - 1 else nn.Identity()
            ])

        self.temporal_conv = nn.Sequential(*conv_layers)

        # Adaptive pooling to compress any T to size 1
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (T, N, C)
        Returns:
            Output tensor of shape (N, C)
        """
        T, N, C = x.shape

        # Reshape to (N, C, T) for Conv1d
        x = x.permute(1, 2, 0)  # (T, N, C) -> (N, C, T)

        # Apply temporal convolution
        x = self.temporal_conv(x)  # (N, C, T)

        # Apply adaptive pooling to compress T dimension to 1
        x = self.adaptive_pool(x)  # (N, C, 1)

        # Squeeze the temporal dimension
        x = x.squeeze(-1)  # (N, C)

        return x