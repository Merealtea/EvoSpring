from models import ModelGeneral
import torch
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
    def __init__(self, pos_dim, ld, layer_num, pre_layer_num, bottom_layer_num, mlp_hidden_layer, MP_times, enhance, agg_conv_pos, default_spring_Y, default_drag_damping=None, default_dashpot_damping=None):
        # in: d_x(used for driven nodes only),type
        # out: d_x
        in_dim = 60 # init_pos, node_mass, node_damping, node_type, pos_encoding
        out_dim = pos_dim
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

        # Learnable damping bias parameters (initialized to 0)
        self.drag_damping_bias = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.dashpot_damping_bias = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

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
        edge_weight, edge_feature = self.process(x, edge_mech_in, m_ids, multi_gs, m_gs_parent, pos, self.temp, attn_mask)

        return edge_weight, edge_feature
        
    
    def forward(self, m_idx, m_gs, m_gs_parent, node_in, edge_mech_in, attn_mask=None):
        if self.temp > 0.1 :
            self.temp *= self.gamma
            self.temp = torch.clamp(self.temp, 0.1)

        # get mat pos and type
        node_pos, node_type = self._get_pos_type(node_in)

        # Process current frame through EMD
        edge_weight, edge_feature = self._EMD(node_in, edge_mech_in, m_idx, m_gs, m_gs_parent, node_pos, node_type, attn_mask)

        # 6. Apply Gumbel-Softmax for edge selection
        edge_logits = self.edge_selector(edge_weight)  # Shape: [batch, num_edges, 2]
        edge_probs_soft, edge_probs_hard = gumbel_softmax(edge_logits, self.temp, hard=False)
        # edge_keep_prob = edge_probs_soft[..., 1:2]  # Probability of keeping the edge, shape: [batch, num_edges, 1]

        # 统计 edge_probs_hard[..., 0] 中为 1 的边的数量和比例
        edges_with_one = (edge_probs_hard[..., 0] == 1).sum().item()
        total_edges = edge_probs_hard[..., 0].numel()
        ratio = edges_with_one / total_edges if total_edges > 0 else 0
        print(f"Edges with value 1: {edges_with_one}/{total_edges} ({ratio*100:.2f}%)")

        edge_mech_in_bias = self.edge_decode(edge_feature)[0,:,0]  # Remove time dimension

        print("edge bias min is {}, edge bias max is {}".format(edge_mech_in_bias.min(), edge_mech_in_bias.max()))

        s_out = ((self.S_0 + edge_mech_in_bias * 1e2)) * edge_probs_hard[0, :, 0]
        s_out = torch.clip(s_out, 0, cfg.spring_Y_max)

        # Return spring stiffness and damping parameters (default + learnable bias)
        drag_damping_out = self.drag_damping_0 + self.drag_damping_bias * 10
        dashpot_damping_out = self.dashpot_damping_0 + self.dashpot_damping_bias * 100
      
        return s_out, drag_damping_out, dashpot_damping_out, edge_probs_soft
    