from models import ModelGeneral
import torch
from torch_geometric.utils import degree
from End2End_core_model import End2End
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

class End2End_EvoSpring(ModelGeneral):
    def __init__(self, pos_dim, ld, layer_num, pre_layer_num, bottom_layer_num, mlp_hidden_layer, MP_times, enhance, agg_conv_pos, default_spring_Y, default_drag_damping=None, default_dashpot_damping=None, default_collision_elas=None, default_collision_fric=None, default_collision_object_elas=None, default_collision_object_fric=None):
        # in: d_x(used for driven nodes only),type
        # out: d_x
        in_dim = 60 # init_pos, node_mass, node_damping, node_type, pos_encoding
  
        self.lagrangian = False
        super(ModelGeneral, self).__init__()
        edge_set_num = 1

        self.encode = MLP(in_dim, ld, ld, mlp_hidden_layer, True)
        # Fourier feature transform module
        self.fourier_transform = FourierFeatureTransform(input_dim=ld, num_freqs=6)

        # layer_num = 1
        # pre_layer_num = 1
        self.process = End2End(layer_num, pre_layer_num, bottom_layer_num, ld, mlp_hidden_layer, pos_dim,
                                    self.lagrangian, enhance, agg_conv_pos, edge_set_num)

        self.edge_decode = MLP(ld, ld, 1, 3, False)
        # Edge selection MLP: predicts logits for keeping/discarding each edge
        self.edge_selector = MLP(ld, ld, 2, 3, False)

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
        self.collision_object_elas_0 = float(default_collision_object_elas) if default_collision_object_elas is not None else 0.7
        self.collision_object_fric_0 = float(default_collision_object_fric) if default_collision_object_fric is not None else 0.3

        # Learnable damping bias parameters (initialized to 0)
        self.drag_damping_bias = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.dashpot_damping_bias = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

        # Learnable collision bias parameters (initialized to 0)
        self.collision_elas_bias = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.collision_fric_bias = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.collision_object_elas_bias = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.collision_object_fric_bias = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

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
        pos_mat_world = node_in[..., :self.pos_dim] 

        # torch.cat((node_in[..., self.pos_dim:2 * self.pos_dim], node_in[..., :self.pos_dim]), dim=-1)
        node_type = node_in[..., -1].clone()
        return pos_mat_world, node_type

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
        return node_in[:, 3:-1].clone()
    
    def _get_mesh_pos(self, node_info):
        return node_info[..., 2*self.pos_dim: 3*self.pos_dim].clone()

    def _EMD(self, node_feature, edge_mech_in, m_ids, multi_gs, m_gs_parent, pos, node_type, attn_mask=None):
        node_feature = self._get_nodal_latent_input(node_feature)

        # TODO: Generate edge features for spring-mass system if needed
        x = self.encode(node_feature)
       
        # Add node type embeddings
        type_emb = self.node_type_embedding(node_type.long())
        x = x + type_emb
        edge_feature = self.process(x, edge_mech_in, m_ids, multi_gs, m_gs_parent, pos, self.temp, attn_mask)

        return edge_feature
    
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
    
    def forward(self, m_idx, m_gs, m_gs_parent, node_in, edge_mech_in, attn_mask=None):
        if self.temp > 0.1 :
            self.temp *= self.gamma
            self.temp = torch.clamp(self.temp, 0.1)

        # get mat pos and type
        node_pos, node_type = self._get_pos_type(node_in)

        # Process current frame through EMD
        edge_feature = self._EMD(node_in, edge_mech_in, m_idx, m_gs, m_gs_parent, node_pos, node_type, attn_mask)

        edge_mech_in_bias = self.edge_decode(edge_feature)[0,:,0]  # Remove time dimension

        print("edge bias min is {}, edge bias max is {}".format(edge_mech_in_bias.min(), edge_mech_in_bias.max()))

        s_out = ((self.S_0 + edge_mech_in_bias * 1e3))
        s_out = torch.clip(s_out, 1e-8, cfg.spring_Y_max)

        # # Return spring stiffness and damping parameters (default + learnable bias)
        drag_damping_out = self.drag_damping_0 + self.drag_damping_bias * 100
        dashpot_damping_out = self.dashpot_damping_0 + self.dashpot_damping_bias * 1000

        return s_out, drag_damping_out, dashpot_damping_out
    