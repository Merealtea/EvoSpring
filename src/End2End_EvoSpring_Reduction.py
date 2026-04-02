from models import ModelGeneral
import torch
from torch_geometric.utils import degree
from End2End_reduction_model import End2EndReduction
from core_model import MLP, gumbel_softmax
from qqtt.model.diff_simulator import (
    SpringMassSystemWarp,
)
import numpy as np
import open3d as o3d
import warp as wp
from qqtt.utils import logger, cfg
"""
This model works for mechanical information estimations of spring-mass systems.
"""

class FourierFeatureTransform(torch.nn.Module):
    """
    Applies Fourier feature transformation to the input features.
    """
    def __init__(self, input_dim, num_freqs=6):
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        # Output dimension: original features + sin/cos for each frequency
        self.out_dim = input_dim * (2 * num_freqs + 1)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands.to(x.device):
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1)

class End2EndReduction_EvoSpring(ModelGeneral):
    def __init__(self, pos_dim, ld, layer_num, pre_layer_num, bottom_layer_num, mlp_hidden_layer, MP_times, enhance, agg_conv_pos, default_spring_Y, default_drag_damping=None, default_dashpot_damping=None, default_collision_elas=None, default_collision_fric=None, default_collision_object_elas=None, default_collision_object_fric=None):
        # in: d_x(used for driven nodes only),type
        # out: d_x
        in_dim = 3 + 1 + 1 + 60# init_pos, node_mass, node_damping, node_type, pos_encoding
  
        self.lagrangian = False
        super(ModelGeneral, self).__init__()
        edge_set_num = 1

        self.encode = MLP(in_dim, ld, ld, mlp_hidden_layer, True)
        
        # layer_num = 1
        # pre_layer_num = 1
        self.process = End2EndReduction(layer_num, pre_layer_num, bottom_layer_num, ld, mlp_hidden_layer, pos_dim,
                                    self.lagrangian, enhance, agg_conv_pos, edge_set_num)

        self.edge_decode = MLP(ld, ld, 1, 3, False)

        # Node type embedding: 4 types (0: object, 1: surface, 2: interior, 3: controller)
        self.node_type_embedding = torch.nn.Embedding(4, ld)
        self.MP_times = MP_times
        self.pos_dim = pos_dim
        self.gamma = 0.5

        self.temp = torch.tensor(1.0)
        self.S_0 = default_spring_Y

        # Store default damping values
        self.drag_damping_0 = float(default_drag_damping) if default_drag_damping is not None else 0.5
        self.dashpot_damping_0 = float(default_dashpot_damping) if default_dashpot_damping is not None else 0.5

        # Store default collision values
        self.collision_elas_0 = float(default_collision_elas) if default_collision_elas is not None else 0.5
        self.collision_fric_0 = float(default_collision_fric) if default_collision_fric is not None else 0.5
        # self.collision_object_elas_0 = float(default_collision_object_elas) if default_collision_object_elas is not None else 0.7
        # self.collision_object_fric_0 = float(default_collision_object_fric) if default_collision_object_fric is not None else 0.3

        # --- 新增的四个参数 Decoders ---
        self.drag_damping_decode = MLP(ld*2, ld, 1, 3, False)
        self.dashpot_damping_decode = MLP(ld*2, ld, 1, 3, False)
        self.collision_elas_decode = MLP(ld*2, ld, 1, 3, False)
        self.collision_fric_decode = MLP(ld*2, ld, 1, 3, False)

    def load_warp_simulator(self, simulator):
        self.simulator = simulator

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def _get_pos_type(self, node_in):
        # 0:3 current_pos, 3:6 next_pos 6:9 mesh_pos, 9:12 node_vel 12:13 node_mass 13:16 node_damping, 16: type
        pos_mat_world = node_in[..., :self.pos_dim].clone() 

        node_mass = node_in[..., -2:-1].clone()

        # torch.cat((node_in[..., self.pos_dim:2 * self.pos_dim], node_in[..., :self.pos_dim]), dim=-1)
        node_type = node_in[..., -1].clone()
        return pos_mat_world, node_mass, node_type

    def _get_vel(self, node_in):
        return node_in[..., self.pos_dim * 3 : self.pos_dim * 4]

    def _pre(self, node_in, node_tar, node_type):
        # 0 : object node, 1 : surface node, 2: interior node, 3: controller node
        controller_node = (node_type == 3).bool()
        object_node = (node_type == 0).bool()

        # 0:3 current_pos, 3:6 next_pos 6:9 mesh_pos, 9:12 node_vel 12:13 node_mass 13:16 node_damping, 16: type
        node_in[controller_node][..., self.pos_dim:2*self.pos_dim] = node_tar[controller_node]
        node_in[object_node][..., self.pos_dim:2*self.pos_dim] = node_tar[object_node]
        return node_in

    def _mask(self, node_in, node_tar, node_type, node_predict):
        # only measure int nodes(0)
        int_node = (node_type == 0).bool().unsqueeze(-1)
        # assert int_node[0].sum() == 1577
        mask = torch.where(int_node, torch.ones_like(node_tar), torch.zeros_like(node_tar))
        node_predict = torch.where(int_node, node_predict, node_tar)
        return node_predict, mask

    def _get_nodal_latent_input(self, node_in):
        # in_dim for nodal encoding: [world, mesh_pos, node_mass, velocity, node_damping, type] out of [world, mesh_pos, node_mass, velocity, node_damping, type]
        return node_in
    
    def _get_mesh_pos(self, node_info):
        return node_info[..., : self.pos_dim].clone()

    def _EMD(self, node_feature, m_ids, multi_gs, pos, node_mass, node_type):
        node_feature = self._get_nodal_latent_input(node_feature)
     
        x = self.encode(node_feature)
       
        # Add node type embeddings
        # type_emb = self.node_type_embedding(node_type.long())
        # x = x + type_emb
        mlvl_edge_feature, mlvl_masses, mlvl_node_type, mlvl_spring_graph, new_node_idx\
              = self.process(x, m_ids, multi_gs, pos, node_mass, node_type, self.temp)

        mlvl_node_index = [new_node_idx[0]]
        for i in range(1, len(new_node_idx)):
            mlvl_node_index.append(np.array(mlvl_node_index[i-1])[new_node_idx[i]].tolist())

        return mlvl_edge_feature, mlvl_masses, mlvl_node_type, mlvl_spring_graph, mlvl_node_index 
    
    def _enforce_min_degree(self, m_gs, edge_mask, num_nodes):
        """
        Fast, parallelized constraint enforcement to ensure node degree >= min(2, original_degree).
        Runs entirely on GPU using vectorized operations.
        """
        device = edge_mask.device
        
        # Handle potential batch dimensions in m_idx (assuming shape [2, E] or [1, 2, E])
        src = m_gs[0][0][:edge_mask.shape[0]]
        
        # 1. Calculate target minimum degree based on original graph connections
        orig_degree = degree(src, num_nodes=num_nodes, dtype=torch.long)
        target_degree = torch.clamp(orig_degree, max=2)
        
        constrained_mask = edge_mask.long().clone()
        
        # 2. Max possible deficit is 2, so we only ever need at most 2 iterations
        for _ in range(2):
            # Calculate current degree based on edges currently kept (mask == 1)
            current_degree = degree(src[constrained_mask == 1], num_nodes=num_nodes, dtype=torch.long)
            deficit = target_degree - current_degree
            
            # Identify nodes that still need more edges
            deficient_nodes = torch.nonzero(deficit > 0).squeeze(-1)
            if deficient_nodes.numel() == 0:
                break # All constraints satisfied!
                
            # 3. Find available edges: they must belong to a deficient node AND be currently dropped (0)
            is_deficient_src = deficit[src] > 0
            available_edges_mask = is_deficient_src & (constrained_mask == 0)
            available_edge_indices = torch.nonzero(available_edges_mask).squeeze(-1)
            
            if available_edge_indices.numel() == 0:
                break # No more valid original edges to add
                
            # 4. Pick ONE valid edge for each deficient node in parallel
            # Use scatter_ to place edge indices into a node-sized array.
            # Multiple edges for the same node will overwrite each other, leaving the last one.
            selected_edge_per_node = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
            selected_edge_per_node.scatter_(0, src[available_edge_indices], available_edge_indices)

            # 5. Flip the selected random edges back to 1
            edges_to_add = selected_edge_per_node[selected_edge_per_node != -1]
            constrained_mask[edges_to_add] = 1

        return constrained_mask
    
    def forward(self, m_idx, m_gs, node_in):
        if self.temp > 0.1 :
            self.temp *= self.gamma
            self.temp = torch.clamp(self.temp, 0.1)

        mlvl_s_out = []
        mlvl_drag_damping_out = []
        mlvl_dashpot_damping_out = []
        mlvl_collide_elas_out = []
        mlvl_collide_fric_out = []

        # get mat pos and type
        node_pos, node_mass, node_type = self._get_pos_type(node_in)

        # Process current frame through EMD
        mlvl_edge_feature, mlvl_masses, mlvl_node_type, mlvl_spring_graph, mlvl_node_index \
                = self._EMD(node_in, m_idx, m_gs, node_pos, node_mass, node_type)

        for edge_feature in mlvl_edge_feature:
            # predict edge_mech_bias
            edge_mech_in_bias = self.edge_decode(edge_feature)[0,:,0]
            s_out = ((self.S_0 + edge_mech_in_bias * 1e3))
            s_out = torch.clip(s_out, 1e-8, cfg.spring_Y_max)
            mlvl_s_out.append(s_out)

            # 将 Mean 和 Max 拼接 (需要在 __init__ 中把 Decoder 的输入维度改为 2 * ld)
            mean_feat = torch.mean(edge_feature, dim=1)
            max_feat = torch.max(edge_feature, dim=1)[0]
            global_feature = torch.cat([mean_feat, max_feat], dim=-1) # shape: [bs, 2 * C]

            # 2. predict drag_damping bias
            
            drag_bias = self.drag_damping_decode(global_feature)[0] # 获取标量
            drag_out = self.drag_damping_0 + drag_bias
            # drag_out = torch.clip(drag_out, 1e-8, 100.0) 
            mlvl_drag_damping_out.append(drag_out)

            # 3. predict dashpot_damping bias
            dashpot_bias = self.dashpot_damping_decode(global_feature)[0]
            dashpot_out = self.dashpot_damping_0 + dashpot_bias
            # dashpot_out = torch.clip(dashpot_out, 1e-8, 100.0)
            mlvl_dashpot_damping_out.append(dashpot_out)

            # 4. predict collision_elas bias (Restitution)
            elas_bias = self.collision_elas_decode(global_feature)[0]
            elas_out = self.collision_elas_0 + elas_bias * 0.001
            elas_out = torch.clip(elas_out, 0.0, 1.0) 
            mlvl_collide_elas_out.append(elas_out)

            # 5. predict collision_fric bias (Friction)
            fric_bias = self.collision_fric_decode(global_feature)[0]
            fric_out = self.collision_fric_0 + fric_bias * 0.001
            fric_out = torch.clip(fric_out, 0.0, 2.0) 
            mlvl_collide_fric_out.append(fric_out)
            
        return mlvl_s_out, mlvl_drag_damping_out, mlvl_dashpot_damping_out, \
            mlvl_collide_fric_out, mlvl_collide_elas_out, mlvl_masses, \
                mlvl_spring_graph, mlvl_node_index, mlvl_node_type
    