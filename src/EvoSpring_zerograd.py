from models import ModelGeneral
import torch
from EvoSpring_core_model import EvoSpring
from core_model import MLP
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

class EvoSpringModel(ModelGeneral):
    def __init__(self, pos_dim, ld, layer_num, pre_layer_num, bottom_layer_num, mlp_hidden_layer, MP_times, enhance, agg_conv_pos, default_spring_Y):
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
        self.process = EvoSpring(layer_num, pre_layer_num, bottom_layer_num, ld, mlp_hidden_layer, pos_dim, 
                                    self.lagrangian, enhance, agg_conv_pos, edge_set_num)
        
        self.edge_decode = MLP(self.fourier_transform.out_dim, ld, 1, 3, False)
        # Node type embedding: 4 types (0: object, 1: surface, 2: interior, 3: controller)
        self.node_type_embedding = torch.nn.Embedding(4, ld)
        self.MP_times = MP_times
        self.pos_dim = pos_dim
        self.gamma = 0.999
        
        self.temp = torch.tensor(0.1)
        self.S_0 = default_spring_Y

    def load_warp_simulator(
            self,
            dt,
            init_vertices,
            init_springs,
            init_spring_Y,
            init_rest_lengths,
            init_masses,
            num_object_springs,
            init_masks,
            init_velocities,
            num_all_points,
            num_surface_points,
            num_original_points,
            controller_points,
            object_points,
            object_visibilities,
            object_motions_valid,
            collide_elas,
            collide_fric,
            dashpot_damping,
            drag_damping,
            collide_object_elas,
            collide_object_fric,
            collision_dist,
            reverse_z,
            spring_Y_min,
            spring_Y_max,
            self_collision,
            num_substeps = 5,
            device=None,
    ):
        if hasattr(self, "simulator"):
            self.simulator = None
        
        self.init_vertices = torch.FloatTensor(init_vertices).contiguous().to(device)
        self.init_springs = torch.tensor(init_springs.T, dtype=torch.int32).contiguous().to(device)
        self.init_rest_lengths = torch.FloatTensor(init_rest_lengths).contiguous().to(device)
        self.init_masses = torch.FloatTensor(init_masses).contiguous().to(device)
        self.init_spring_Y = init_spring_Y
        self.object_visibilities = torch.FloatTensor(object_visibilities).contiguous().to(device)
        self.object_motions_valid = torch.FloatTensor(object_motions_valid).contiguous().to(device)
        self.init_velocities = torch.FloatTensor(init_velocities).contiguous().to(device)
        self.controller_points = torch.FloatTensor(controller_points).contiguous().to(device)\
              if controller_points is not None else None
        self.object_points = torch.FloatTensor(object_points).contiguous().to(device)

        self.num_object_springs = num_object_springs
        self.dt = dt
        self.collide_elas = collide_elas
        self.collide_fric = collide_fric
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping
        self.collide_object_elas = collide_object_elas
        self.collide_object_fric = collide_object_fric
        self.collision_dist = collision_dist
        self.reverse_z = reverse_z
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max
        self.self_collision = self_collision
        self.num_all_points = num_all_points
        self.num_surface_points = num_surface_points
        self.num_original_points = num_original_points
        self.num_substeps = num_substeps

        self.simulator = SpringMassSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=self.dt,
            num_substeps=self.num_substeps,
            spring_Y=self.init_spring_Y, 
            collide_elas=self.collide_elas, 
            collide_fric=self.collide_fric, 
            dashpot_damping=int(self.dashpot_damping), # DEBUG
            drag_damping=int(self.drag_damping), # DEBUG
            collide_object_elas=self.collide_object_elas,
            collide_object_fric=self.collide_object_fric,
            init_masks=init_masks,
            collision_dist=self.collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=self.reverse_z,
            spring_Y_min=self.spring_Y_min,
            spring_Y_max=self.spring_Y_max,
            gt_object_points=self.object_points, 
            gt_object_visibilities=self.object_visibilities.bool(),
            gt_object_motions_valid=self.object_motions_valid.bool(),
            self_collision=self.self_collision,
        )

        self.simulator.set_init_state(
                self.simulator.wp_init_vertices,
                self.simulator.wp_init_velocities
        )

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

    def _EMD(self, node_feature, edge_mech_in, m_ids, multi_gs, m_gs_parent, pos, node_type):


        node_feature = self._get_nodal_latent_input(node_feature)

        # TODO: Generate edge features for spring-mass system if needed
        x = self.encode(node_feature)

        # Add node type embeddings
        type_emb = self.node_type_embedding(node_type.long())
        x = x + type_emb

        for _ in range(self.MP_times):
            x, edge_feature = self.process(x, edge_mech_in, m_ids, multi_gs, m_gs_parent, pos, self.temp)

        return x, edge_feature
        
    
    def forward(self, m_idx, m_gs, m_gs_parent, node_in, edge_mech_in):
        if self.temp > 0.1 :
            self.temp *= self.gamma
            self.temp = torch.clamp(self.temp, 0.1)

        # get mat pos and type
        node_pos, node_type = self._get_pos_type(node_in)

        # Process current frame through EMD
        _, edge_feature = self._EMD(node_in, edge_mech_in, m_idx, m_gs, m_gs_parent, node_pos, node_type)

        # the edge_feature is symmetry
        num_edge = edge_feature.shape[1]

        # Average symmetric edges
        edge_feature = edge_feature[:, :num_edge//2] + edge_feature[:, num_edge//2:]
  
        # 5. Apply Fourier feature transform
        edge_feature = self.fourier_transform(edge_feature)

        edge_mech_in_bias = self.edge_decode(edge_feature)[0,:,0]  # Remove time dimension

        print("edge bias min is {}, edge bias max is {}".format(edge_mech_in_bias.min(), edge_mech_in_bias.max()))
        s_out = self.S_0 + edge_mech_in_bias * 1e2
        s_out = torch.clip(s_out, 1e-2, cfg.spring_Y_max)

        return s_out