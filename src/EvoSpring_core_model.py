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

class evospring_amp(MessagePassing):
    def __init__(self, latent_dim, hidden_layer, pos_dim, lagrangian):
        super().__init__(aggr='add', flow='target_to_source')
        self.mlp_node_delta = MLP(2 * latent_dim, latent_dim, latent_dim, hidden_layer, True)
        # egde info : 2 * latent_dim for node i and j
        # + 2 * pos_dim +2 for lagrangian (dir and norm_w and norm_m)
        # + 2 for edge_mech_input (rest_length and dashpot_damping)
        edge_info_in_len = 2 * latent_dim + 2 * pos_dim + 2 + 2 if lagrangian else 2 * latent_dim + pos_dim + 1 + 2
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

    def forward(self, x, edge_mech_input, g, pos, temp):
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
            tmp = torch.cat([fiber.unsqueeze(0).repeat(T, 1, 1), x_i, x_j, edge_mech_input], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j, edge_mech_input], dim=-1)
        
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
    
class evospring_amp_base(MessagePassing):
    def __init__(self, latent_dim, hidden_layer, pos_dim, lagrangian):
        super().__init__(aggr='add', flow='target_to_source')
        self.mlp_node_delta = MLP(2 * latent_dim, latent_dim, latent_dim, hidden_layer, True)
        edge_info_in_len = 2 * latent_dim + 2 * pos_dim + 2 + 2 if lagrangian else 2 * latent_dim + pos_dim + 1 + 2
        self.mlp_edge_info = MLP(edge_info_in_len, latent_dim, latent_dim, hidden_layer, True)
        self.mlp_edge_weight = Seq(*[MLP(latent_dim, latent_dim, 1, hidden_layer, False)])
        self.lagrangian = lagrangian
        self.pos_dim = pos_dim
        self.latent_dim = latent_dim

    def forward(self, x, edge_mech_info, g, pos):
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
            tmp = torch.cat([fiber.unsqueeze(0).repeat(T, 1, 1), x_i, x_j, edge_mech_info], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j, edge_mech_info], dim=-1)
        
        edge_embedding = self.mlp_edge_info(tmp)

        edge_weight = self.mlp_edge_weight(edge_embedding)
        edge_weight = scatter_softmax(edge_weight, j, dim=-2)

        edge_embedding = edge_embedding * edge_weight

        aggr_out = scatter(edge_embedding, j, dim=-2, dim_size=x.shape[-2], reduce="sum")
        
        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_node_delta(tmp) + x, edge_weight, edge_embedding

class EvoSpring(EvoMesh):
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
        self.bottom_gmp = nn.ModuleList(evospring_amp_base(ld, hidden_layer, pos_dim, lagrangian) for _ in range(self.bottom_ln))
        for _ in range(self.l_n):
            if _ < self.pre_l_n:
                self.down_gmps.append(evospring_amp_base(ld, hidden_layer, pos_dim, lagrangian))
            else:
                self.down_gmps.append(evospring_amp(ld, hidden_layer, pos_dim, lagrangian))
            self.up_gmps.append(evospring_amp_base(ld, hidden_layer, pos_dim, lagrangian))
            self.unpools.append(Unpool())
        self.esn = edge_set_num
        self.lagrangian = lagrangian

    def forward(self, node_in, edge_mech_in, mm_ids, mm_gs, mm_gs_parent, pos, temp=0.1, weights=None):
        # node_in is in shape of (T), N, F
        # if edge_set_num>1, then m_g is in shape: Level,(Set),2,Edges, the 0th Set is main/material graph
        # pos is in (T),N,D

        down_outs = []
        down_ps = []
        multi_level_edge_mech_info = []
        cts = []

        w = pos.new_ones((pos.shape[-2], 1)) if weights is None else weights
        T = node_in.shape[0]

        temporal_edge_mech_in = edge_mech_in[None].repeat(T, 1, 1)
        
        # mm_ids is node kept in current layer
        # mm_gs is the edge connection
        # mm_gs_parent is the parent edge idx for current edge 
        m_ids = mm_ids[:self.pre_l_n]
        m_gs = mm_gs[:self.pre_l_n + 1]
        m_gs_parent = mm_gs_parent[:self.pre_l_n]

        # down pass
        l_n = self.l_n 
        for i in range(l_n):
            num_nodes = node_in.shape[-2] if i == 0 else len(m_ids[i-1]) #.shape[0]
     
            # # We don't need node self loop right now
            # if self.esn > 1:
            #     gs = []
            #     gs_main, _ = add_remaining_self_loops(m_gs[i][0]) 
            #     gs_cont, _ = add_remaining_self_loops(m_gs[i][1]) 
            #     gs = [gs_main, gs_cont]
            # else:
            #     gs, _ = add_remaining_self_loops(m_gs[i]) 

            gs = m_gs[i]
            multi_level_edge_mech_info.append(temporal_edge_mech_in)

            if torch.isnan(node_in).any():
                import pdb; pdb.set_trace()

            if i < self.pre_l_n:
                node_in, ew, _ = self.down_gmps[i](node_in, temporal_edge_mech_in, gs, pos)
                if i == 0 and self.lagrangian:
                    node_in, ew, _ = self.down_gmps[i](node_in, temporal_edge_mech_in, gs, pos)
                y_hard = None
            else:
                # deeper downsample
                # dowmsample more to make the network more compact
                node_in, ew, y_hard = self.down_gmps[i](node_in, temporal_edge_mech_in, gs, pos, temp)
                if i == 0 and self.lagrangian:
                    node_in, ew, y_hard = self.down_gmps[i](node_in, temporal_edge_mech_in, gs, pos, temp)
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
                
                g = self.pool_edge(edge_index2, m_idx, num_nodes=num_nodes)
                g, _ = coalesce(g, None, num_nodes=len(m_idx))

                # Generate m_gs_parent by mapping g to gs
                gs_parent_new = torch.full((g.shape[1], 2), -1, dtype=torch.long, device=g.device)

                # For each edge in g, find corresponding edge(s) in gs
                # Sparse version using torch_geometric utilities to avoid dense matrices

                # Map back to original node indices
                src_nodes = g[0]  # shape: (num_edges,)
                tgt_nodes = g[1]  # shape: (num_edges,)
                src_orig = m_idx[src_nodes]  # shape: (num_edges,)
                tgt_orig = m_idx[tgt_nodes]  # shape: (num_edges,)

                # Create edge tuples for efficient lookup
                # Use hash-based lookup instead of dense matrix
                num_g_edges = g.shape[1]
                num_gs_edges = gs.shape[1]

                # Build a dictionary mapping (src, tgt) -> edge_idx in gs
                gs_edge_dict = {}
                for edge_idx in range(num_gs_edges):
                    key = (gs[0, edge_idx].item(), gs[1, edge_idx].item())
                    if key not in gs_edge_dict:
                        gs_edge_dict[key] = edge_idx

                # Find direct edges using hash lookup
                for g_edge_idx in range(num_g_edges):
                    src = src_orig[g_edge_idx].item()
                    tgt = tgt_orig[g_edge_idx].item()
                    key = (src, tgt)

                    if key in gs_edge_dict:
                        # Direct edge exists
                        gs_parent_new[g_edge_idx, 0] = gs_edge_dict[key]
                    else:
                        # Find double-hop path through intermediate nodes
                        # Find neighbors of src in gs
                        src_out_mask = gs[0] == src
                        src_neighbors = gs[1, src_out_mask]
                        src_edge_indices = torch.where(src_out_mask)[0]

                        # Find in-neighbors of tgt in gs
                        tgt_in_mask = gs[1] == tgt
                        tgt_in_neighbors = gs[0, tgt_in_mask]

                        # Find common intermediate nodes using torch.isin for faster lookup
                        common_mask = torch.isin(src_neighbors, tgt_in_neighbors)

                        if common_mask.any():
                            # Get first common intermediate node
                            first_common_idx = torch.where(common_mask)[0][0]
                            intermediate = src_neighbors[first_common_idx].item()
                            edge1_idx = src_edge_indices[first_common_idx]

                            # Find edge from intermediate to tgt using hash lookup
                            key2 = (intermediate, tgt)
                            if key2 in gs_edge_dict:
                                gs_parent_new[g_edge_idx, 0] = edge1_idx
                                gs_parent_new[g_edge_idx, 1] = gs_edge_dict[key2]

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

            # Downsample edge_mech_in: compute new rest_length and inherit dashpot_damping
            if i < l_n - 1:  # Not the last layer
                next_g = m_gs[i + 1]

                # Compute new edge lengths as rest_length
                src_idx = next_g[0]
                tgt_idx = next_g[1]

                if len(pos.shape) == 3:
                    # pos is T × N × D
                    edge_vec = pos[:, src_idx] - pos[:, tgt_idx]  # T × M_new × D
                    new_rest_length = torch.norm(edge_vec, dim=-1, keepdim=True)  # T × M_new × 1
                else:
                    # pos is N × D
                    edge_vec = pos[src_idx] - pos[tgt_idx]  # M_new × D
                    new_rest_length = torch.norm(edge_vec, dim=-1, keepdim=True)  # M_new × 1
                    new_rest_length = new_rest_length.unsqueeze(0).repeat(T, 1, 1)  # T × M_new × 1

                # Inherit dashpot_damping from parent edges (average if two parents)
                gs_parent = m_gs_parent[i]
                parent_damping_1 = temporal_edge_mech_in[:, gs_parent[:, 0], 1:2]  # T × M_new × 1
                mask = gs_parent[:, 1] != -1
                damping_downsampled = parent_damping_1.clone()
                if mask.any():
                    parent_damping_2 = temporal_edge_mech_in[:, gs_parent[mask, 1], 1:2]
                    damping_downsampled[:, mask, :] = (parent_damping_1[:, mask, :] + parent_damping_2) / 2.0

                # Combine new edge_mech_in
                temporal_edge_mech_in = torch.cat([new_rest_length, damping_downsampled], dim=-1)

        multi_level_edge_mech_info.append(temporal_edge_mech_in)
        
        
        for l in range(self.bottom_ln):
            node_in, ew, _ = self.bottom_gmp[l](node_in, temporal_edge_mech_in, m_gs[l_n], pos)
            if self.lagrangian and l == 0:
                node_in, ew, _ = self.bottom_gmp[l](node_in, temporal_edge_mech_in, m_gs[l_n], pos)

        # up pass
        for i in range(l_n):
            up_idx = l_n - i - 1
            g, idx, temporal_edge_mech_in = m_gs[up_idx], m_ids[up_idx], multi_level_edge_mech_info[up_idx]
            # if self.esn > 1:
            #     g_main, _ = add_remaining_self_loops(g[0])
            #     g_cont, _ = add_remaining_self_loops(g[1]) 
            #     g = [g_main, g_cont]
            # else:
            #     g, _ = add_remaining_self_loops(g)
            
            node_in = self.unpools[i](node_in, down_outs[up_idx].shape[-2], idx)
            tmp_g = g[0] if self.esn > 1 else g
            node_in= self.edge_conv(node_in, tmp_g, cts[up_idx], aggragating=False)
            node_in, ew_u, edge_embedding = self.up_gmps[i](node_in, temporal_edge_mech_in, g, down_ps[up_idx])

            # if up_idx == 0 and self.lagrangian:
            #     node_in, ew_u = self.up_gmps[i](node_in, temporal_edge_mech_in, g, down_ps[up_idx])
            node_in = node_in + down_outs[up_idx]
        return node_in, edge_embedding