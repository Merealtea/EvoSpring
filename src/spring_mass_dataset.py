from datasets import MeshGeneralDataset, MeshType
import torch
import os
from helpers_bistride import generate_multi_layer_stride, SeedingHeuristic
from helpers_mesh import tetras_to_edges, triangles_to_edges, quads_to_edges, lines_to_edges
from torch_geometric.data import Data, Dataset
import numpy as np
import h5py
import json
import pickle

class MeshSpringMassDataset(MeshGeneralDataset):
    def __init__(self, root, layer_num, stride, mode='train', noise_shuffle=False, noise_level=None, noise_gamma=1.0, 
                 refine_steps=[], recal_mesh=False, consist_mesh=False, 
                 object_case=None, train_frame=None, test_frame=None, args=None):
        in_normal_feature_list, out_normal_feature_list, roll_normal_feature_list = ['world_pos', 'velocities'], ['world_pos'], ['world_pos', 'velocities']
        # NOTE instance_id for a specific transient seq; each instance is in shape T,N,F

        self.mode = mode
        self.object_case = args.object_case
        self.data_dir = os.path.join(root, object_case)
        self.train_frame = train_frame
        self.test_frame = test_frame

        self.layer_num = layer_num
        self.recal_mesh = recal_mesh
        self.consist_mesh = consist_mesh
        mesh_type = MeshType.Line
        self.mesh_type = mesh_type
        self.has_contact = False
        self.has_self_contact = False
        self.dirichelet_markers = []
        self.seed_heuristic = SeedingHeuristic.MinAve
        self.save_cells = True
        self.refine_steps = refine_steps
        self.condition_steps = 0
        self.prediction_steps = 0

        # read all features indicated in meta
        with open(os.path.join(self.data_dir, 'meta.json'), 'r') as fp:
            self.meta = json.loads(fp.read())
        field_names = self.meta['field_names']
        fields = dict()
        with h5py.File(os.path.join(self.data_dir, 'init_spring_mass.h5'), 'r') as f:
            for name in field_names:
                if name == "cells":
                    fields[name] = np.array(f[name])
                    self.cells = fields[name][0]
                else:
                    fields[name] = torch.tensor(np.array(f[name]), dtype=torch.float)

            self.object_point = np.array(f['object_points'])
            self.mesh_pos = np.array(f['mesh_pos'])[0]
            self.spring_Y = np.array(f['init_spring_Y'])[0]
            self.spring_reset_length = np.array(f['spring_reset_length'])[0,:,0]
            self.spring_dashpot_damping = np.array(f['spring_dashpot_damping'])[0,:,0]
            self.masses = np.array(f['mass'])[0,:,0]
            self.velocity = np.array(f['velocities'])[0]
            self.controller_point = np.array(f['controller_point'])
            self.object_visibilities = np.array(f['object_visibilities'])
            self.object_motions_valid = np.array(f['object_motions_valid'])
            self.node_type = np.array(f["node_type"])

        # 根据SpringMassSystemWarp 类初始化需要的参数从meta中读取所需要的参数
        self.dt = self.meta['dt']
        self.drag_damping = self.meta['drag_damping']
        self.collide_object_elas = self.meta['collide_object_elas']
        self.collide_object_fric = self.meta['collide_object_fric']
        self.collide_elas = self.meta['collide_elas']
        self.collide_fric = self.meta['collide_fric']
        self.collision_dist = self.meta['collision_dist']
        self.dashpot_damping = self.meta['dashpot_damping']
        self.num_object_points = self.meta['num_object_points']
        self.num_surface_points = self.meta['num_surface_points']
        self.num_original_points = self.meta['num_original_points']

        # read normalization info
        self._read_normalization_info(in_normal_feature_list, out_normal_feature_list, roll_normal_feature_list)
        # shuffle noise and enhance data, determine do or not
        if noise_level is None or not noise_shuffle:
            self.noise_shuffle = False
            self.noise_shuffle = None
            self.noise_gamma = 1.0
        else:
            self.noise_shuffle = True
            self.noise_level = torch.tensor(noise_level, dtype=torch.float)
            self.noise_gamma = noise_gamma
        # cal len of dataset
        self.stride = stride
        self.strided_idx = list(range(0, fields["mesh_pos"].shape[0], stride))
        self.L = len(self.strided_idx) - 1
        for name in field_names:
            fields[name] = fields[name][self.strided_idx]
       
        self._cal_multi_mesh()
        super(MeshGeneralDataset,self).__init__(root)
        self.L = self.L - 1

    def get(self, idx):
        # idx in time seq (enhanced by noise shuffle)
        # also return the midx and mgs, for combining
        if self.condition_steps > 0:
            condition_feat = self.in_feature[idx: idx+self.condition_steps]
            condition_feat = condition_feat.transpose(0, 1)
        else:
            condition_feat = self.in_feature[idx]
            condition_edge_feat = self.edge_mech_feature[idx]

        if self.prediction_steps > 0:
            prediction_feat = self.tar_feature[idx+self.condition_steps: idx+self.condition_steps+self.prediction_steps]
            prediction_feat = prediction_feat.transpose(0, 1)
        else:
            prediction_feat = self.tar_feature[idx+self.condition_steps]
        if self.condition_steps > 0:
            t = torch.arange(idx, idx+self.condition_steps) / (self.L + 1)
            data = Data(x=condition_feat, y=prediction_feat, t=t)
        else:
            data = Data(x=condition_feat, edge_mech=condition_edge_feat, y=prediction_feat, t=idx)
        return data

    def _cal_multi_mesh(self):
        if not self.has_contact:
            if self.consist_mesh:
                mmfile = os.path.join(self.data_dir, 'mmesh_layer_' + str(self.layer_num) + '.dat')
            else:
                mmfile = os.path.join(self.data_dir, self.object_case + '_mmesh_layer_' + str(self.layer_num) + '.dat')
            mmexist = os.path.isfile(mmfile)
            if self.recal_mesh or not mmexist:
                if self.mesh_type == MeshType.Triangle:
                    edge_i = triangles_to_edges(self.cells)
                if self.mesh_type == MeshType.Tetrahedron:
                    edge_i = tetras_to_edges(self.cells)
                if self.mesh_type == MeshType.Quad:
                    edge_i = quads_to_edges(self.cells)
                if self.mesh_type == MeshType.Line:
                    edge_i = lines_to_edges(self.cells)
                if self.mesh_type == MeshType.Flat:
                    edge_i = self.cells

                m_gs, m_ids, m_edge_parents = generate_multi_layer_stride(edge_i,
                                                          self.layer_num,
                                                          seed_heuristic=self.seed_heuristic,
                                                          n=self.mesh_pos.shape[0],
                                                          pos_mesh=self.mesh_pos)
                m_mesh = {'m_gs': m_gs, 'm_ids': m_ids, 'm_edge_parents': m_edge_parents}
                pickle.dump(m_mesh, open(mmfile, 'wb'))
            else:
                m_mesh = pickle.load(open(mmfile, 'rb'))
                m_gs, m_ids = m_mesh['m_gs'], m_mesh['m_ids']
                m_edge_parents = m_mesh.get('m_edge_parents', None)  # Handle old pickle files without edge parents
            self.m_g = m_gs
            self.m_idx = m_ids
            self.m_edge_parents = m_edge_parents
        else:
            raise NotImplementedError("Contact mesh generation is not implemented yet")

        self.recal_mesh = False
        return 

    def _normalize(self, t_in, t_out):
        device = t_in.device
        x_in = t_in.clone()
        x_out = t_out.clone()

        x_in[..., :self.in_norm_l] = (x_in[..., :self.in_norm_l] - self.mean_in.to(device)) / self.std_in.to(device)
        x_in[..., 2*self.in_norm_l:3*self.in_norm_l] = (x_in[..., 2*self.in_norm_l:3*self.in_norm_l] - self.mean_in.to(device)) / self.std_in.to(device)
        x_out[..., :self.out_norm_l] = (x_out[..., :self.out_norm_l] - self.mean_out.to(device)) / self.std_out.to(device)
        return x_in, x_out

    def _preprocess(self, node_pos, node_vel, node_mass, log_spring_Y, spring_reset_length, spring_dashpot_damping, drag_damping, mode='train'):
        # node_type : 0 object point, 1 surface point, 2 interior point, 3 controller point
        if mode == "train":
            node_type = torch.LongTensor(self.node_type[:self.train_frame]).to(node_pos.device)[:-1]
        else:
            node_type = torch.LongTensor(self.node_type[self.train_frame:]).to(node_pos.device)[:-1]

        # concat interior feature and controller feature separately
        node_info_inp = node_pos[:-1].clone()
        node_info_tar = node_pos[1:].clone()  # this is the final target
        node_vel = node_vel[:-1].clone()

        # recalculate it for surface and interior nodes later
        node_damping = (node_vel * drag_damping).clone()

        # enhance by noise level
        if self.noise_shuffle:
            # collect special nodes
            no_noise_node = (node_type == 3).bool() # dont add noise to controller nodes
            noise_base = torch.ones_like(node_info_tar)
            noise_base[:, :] = self.noise_level
            noise = torch.normal(0.0, noise_base)

            # for dirichelet nodes, the noise is zero
            noise = torch.where(no_noise_node, torch.zeros_like(noise), noise)
            node_info_inp += noise
            node_info_tar += (1.0 - self.noise_gamma) * noise

        T = node_type.shape[0]
        # 0:3 current_pos, 3:6 next_pos, 6:9 mesh_pos, 9:12 node_vel 12:13 node_mass 13:16 node_damping, 16: type
        node_mesh = torch.FloatTensor(self.mesh_pos[None].repeat(T, axis=0)).to(node_info_tar.device)
        node_mass = node_mass[None, :, None].repeat(T, 1, 1)

        node_info_inp = torch.cat((node_info_inp, torch.zeros_like(node_info_tar), 
                                   node_vel, node_mesh, node_mass, 
                                   node_damping, node_type), dim=-1)

        
        spring_Y_max, spring_Y_min = 1e5, 0

        spring_Y = (torch.exp(log_spring_Y) - spring_Y_min) / (spring_Y_max - spring_Y_min)

        spring_reset_length_max, spring_reset_length_min = 0.02, 2e-5
        spring_reset_length = (spring_reset_length - spring_reset_length_min) / (spring_reset_length_max - spring_reset_length_min)

        # 0: edge_init_spring, 1: edge_init_reset_length, 2: edge_init_dashpot_damping
        edge_mech_info_inp = torch.stack((spring_Y, spring_reset_length, spring_dashpot_damping * torch.ones_like(spring_reset_length)), dim=-1)
        edge_mech_info_inp = torch.cat([edge_mech_info_inp, edge_mech_info_inp], dim=0)

        # TODO : add normalization for edge features
        self.in_feature, self.tar_feature = self._normalize(node_info_inp, node_info_tar)

        # Input, and target, but in our case, we have to targets, one for node position and velocity, another egde prediction
        return node_info_inp, edge_mech_info_inp, node_info_tar
