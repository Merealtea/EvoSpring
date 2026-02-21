from models import ModelGeneral
import torch
from spring_mass_core_model import SpringMassEvoMesh, TemporalFeatureAggregation
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

class SpringMass(ModelGeneral):
    def __init__(self, pos_dim, ld, layer_num, pre_layer_num, bottom_layer_num, mlp_hidden_layer, MP_times, enhance, agg_conv_pos):
        # in: d_x(used for driven nodes only),type
        # out: d_x
        in_dim = 17 # current_pos, next_pos, mesh_pos, node_vel, node_mass, node_damping, type
        out_dim = pos_dim
        self.lagrangian = False
        super(ModelGeneral, self).__init__()
        edge_set_num = 1

        self.encode = MLP(in_dim, ld, ld, mlp_hidden_layer, True)
        self.process = SpringMassEvoMesh(layer_num, pre_layer_num, bottom_layer_num, ld, mlp_hidden_layer, pos_dim, self.lagrangian, enhance, agg_conv_pos, edge_set_num)
        self.temporal_feature_compression = TemporalFeatureAggregation(ld, ld, 3)
        self.edge_decode = MLP(ld, ld, out_dim, mlp_hidden_layer, False)
        self.MP_times = MP_times
        self.pos_dim = pos_dim
        self.mse = torch.nn.MSELoss(reduction='none')
        
        self.gamma = 0.999
        self.register_buffer('temp', torch.tensor(5.))

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

    def _init_start(
        self,
        object_points,
        controller_points,
        object_radius=0.02,
        object_max_neighbours=30,
        controller_radius=0.04,
        controller_max_neighbours=50,
        mask=None,
    ):
        object_points = object_points.cpu().numpy()
        if controller_points is not None:
            controller_points = controller_points.cpu().numpy()
        if mask is None:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(object_points)
            pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

            # Connect the springs of the objects first
            points = np.asarray(object_pcd.points)
            spring_flags = np.zeros((len(points), len(points)))
            springs = []
            rest_lengths = []
            for i in range(len(points)):
                [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                    points[i], object_radius, object_max_neighbours
                )
                idx = idx[1:]
                for j in idx:
                    rest_length = np.linalg.norm(points[i] - points[j])
                    if (
                        spring_flags[i, j] == 0
                        and spring_flags[j, i] == 0
                        and rest_length > 1e-4
                    ):
                        spring_flags[i, j] = 1
                        spring_flags[j, i] = 1
                        springs.append([i, j])
                        rest_lengths.append(np.linalg.norm(points[i] - points[j]))

            num_object_springs = len(springs)

            if controller_points is not None:
                # Connect the springs between the controller points and the object points
                num_object_points = len(points)
                points = np.concatenate([points, controller_points], axis=0)
                for i in range(len(controller_points)):
                    [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                        controller_points[i],
                        controller_radius,
                        controller_max_neighbours,
                    )
                    for j in idx:
                        springs.append([num_object_points + i, j])
                        rest_lengths.append(
                            np.linalg.norm(controller_points[i] - points[j])
                        )

            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(points))
            return (
                torch.tensor(points, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )
        else:
            mask = mask.cpu().numpy()
            # Get the unique value in masks
            unique_values = np.unique(mask)
            vertices = []
            springs = []
            rest_lengths = []
            index = 0
            # Loop different objects to connect the springs separately
            for value in unique_values:
                temp_points = object_points[mask == value]
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(temp_points)
                temp_tree = o3d.geometry.KDTreeFlann(temp_pcd)
                temp_spring_flags = np.zeros((len(temp_points), len(temp_points)))
                temp_springs = []
                temp_rest_lengths = []
                for i in range(len(temp_points)):
                    [k, idx, _] = temp_tree.search_hybrid_vector_3d(
                        temp_points[i], object_radius, object_max_neighbours
                    )
                    idx = idx[1:]
                    for j in idx:
                        rest_length = np.linalg.norm(temp_points[i] - temp_points[j])
                        if (
                            temp_spring_flags[i, j] == 0
                            and temp_spring_flags[j, i] == 0
                            and rest_length > 1e-4
                        ):
                            temp_spring_flags[i, j] = 1
                            temp_spring_flags[j, i] = 1
                            temp_springs.append([i + index, j + index])
                            temp_rest_lengths.append(rest_length)
                vertices += temp_points.tolist()
                springs += temp_springs
                rest_lengths += temp_rest_lengths
                index += len(temp_points)

            num_object_springs = len(springs)

            vertices = np.array(vertices)
            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(vertices))

            return (
                torch.tensor(vertices, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )

    def _get_pos_type(self, node_in):
        # 0:3 current_pos, 3:6 next_pos 6:9 mesh_pos, 9:12 node_vel 12:13 node_mass 13:16 node_damping, 16: type
        pos_mat_world = node_in[..., :self.pos_dim] 

        vel_mat_world = self._get_vel(node_in)
        # torch.cat((node_in[..., self.pos_dim:2 * self.pos_dim], node_in[..., :self.pos_dim]), dim=-1)
        node_type = node_in[..., -1].clone()
        return pos_mat_world, vel_mat_world, node_type

    def _get_vel(self, node_in):
        return node_in[..., self.pos_dim * 3 : self.pos_dim * 4]

    def _update_states(self, node_in, node_tar, node_type, out):
        vel = self._get_vel(node_in)
        out = out + node_in[..., :self.pos_dim] + vel  
        return out

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
        return node_in.clone()
    
    def _get_mesh_pos(self, node_info):
        return node_info[..., 2*self.pos_dim: 3*self.pos_dim].clone()

    def _EMD(self, node_feature, edge_mech_in, m_ids, multi_gs, m_gs_parent, pos, vel):

        mesh_pos = self._get_mesh_pos(node_feature)[0]
        node_feature = self._get_nodal_latent_input(node_feature)

        # TODO: Generate edge features for spring-mass system if needed
        x = self.encode(node_feature)

        for _ in range(self.MP_times):
            x, edge_feature = self.process(x, edge_mech_in, m_ids, multi_gs, m_gs_parent, pos, vel, self.temp, mesh_pos)

        return x, edge_feature
        
    
    def forward(self, m_idx, m_gs, m_gs_parent, node_in, edge_mech_in, node_tar, pen_coeff=None, prev_edge_feature=None):
        if self.temp > 0.1 :
            self.temp *= self.gamma
            self.temp = torch.clamp(self.temp, 0.1)
        # node_in shape T * N * C
        # edge_mech_in shape T * E * C_edge

        # get mat pos and type
        node_pos, node_vel, node_type = self._get_pos_type(node_in)

        # preprocess: set scripted bcs
        node_in = self._pre(node_in, node_tar, node_type)

        # infer: encode->MP->decode->time integrate to update states
        # out [1, N, 3], the last three dimensions are spring modulus, reset length, dashpot_damping

        # RNN-style frame-by-frame processing
        T = node_in.shape[0]  # Total number of time frames

        # Process each frame sequentially
        for t in range(T):
            # Get current frame data
            node_in_frame = node_in[t:t+1]  # Shape: (1, num_nodes, feature_dim)
            node_pos_frame = node_pos[t:t+1]
            node_vel_frame = node_vel[t:t+1]

            # Process current frame through EMD
            _, edge_feature_frame = self._EMD(node_in_frame, edge_mech_in, m_idx, m_gs, m_gs_parent, node_pos_frame, node_vel_frame)

            # Fusion: Add previous frame's edge feature to current frame (RNN-style)
            if prev_edge_feature is not None:
                edge_feature_frame = edge_feature_frame + prev_edge_feature

            # Update prev_edge_feature for next iteration
            prev_edge_feature = edge_feature_frame

        # Use the last frame's edge feature for final prediction
        edge_feature = edge_feature_frame

        # the edge_feature is symmetry
        num_edge = edge_feature.shape[1]

        # Average symmetric edges
        edge_feature = edge_feature[:, :num_edge//2] + edge_feature[:, num_edge//2:]

        edge_mech_in_bias = self.edge_decode(edge_feature.squeeze(0))  # Remove time dimension

        edge_mech_in_bias[..., 1] = edge_mech_in_bias[..., 1] * 0.005

        # denormolization
        edge_mech_in[..., 0] = torch.log(edge_mech_in[..., 0] * (cfg.spring_Y_max - cfg.spring_Y_min) + cfg.spring_Y_min)
        edge_mech_in[..., 1] = edge_mech_in[..., 1] * (cfg.object_radius - 2e-5) + 2e-5

        half_edge_mech_in = edge_mech_in_bias + edge_mech_in[0, :num_edge//2]

        c0 = half_edge_mech_in[..., 0]
        c1 = half_edge_mech_in[..., 1]
        c2 = half_edge_mech_in[..., 2]  # 注意这里维度的匹配

        # 2. 对分量进行 Clip (非原位)
        c0_clipped = torch.log(torch.clip(torch.exp(c0), cfg.spring_Y_min, cfg.spring_Y_max))
        c1_clipped = torch.clip(c1, 2e-5, cfg.object_radius)

        # 3. 重新组合 (Stack)
        # 这样生成的是全新的 Tensor，没有原地修改任何历史变量
        half_edge_mech_in_new = torch.stack([c0_clipped, c1_clipped, c2], dim=-1)

        return half_edge_mech_in_new
