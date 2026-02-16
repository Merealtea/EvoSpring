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
from qqtt.utils import logger, cfg

class MeshSpringMassDataset(MeshGeneralDataset):
    def __init__(self, root, layer_num, stride, mode='train', noise_shuffle=False, noise_level=None, noise_gamma=1.0, 
                 refine_steps=[], recal_mesh=False, consist_mesh=False, 
                 object_case=None, args=None):
        in_normal_feature_list, out_normal_feature_list, roll_normal_feature_list = ['world_pos', 'velocities'], ['world_pos'], ['world_pos', 'velocities']
        # NOTE instance_id for a specific transient seq; each instance is in shape T,N,F

        self.mode = mode
        self.object_case = args.object_case
        self.data_dir = os.path.join(root, object_case)

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
            meta = json.loads(fp.read())
        field_names = meta['field_names']
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
            self.init_spring_Y = np.array(f['init_spring_Y'])
            self.spring_reset_length = np.array(f['spring_reset_length'])[0,:,0]
            self.spring_dashpot_damping = np.array(f['spring_dashpot_damping'])[0,:,0]
            self.masses = np.array(f['mass'])[0,:,0]
            self.velocity = np.array(f['velocities'])[0]
            self.controller_point = np.array(f['controller_point'])
            self.object_visibilities = np.array(f['object_visibilities'])
            self.object_motions_valid = np.array(f['object_motions_valid'])
            self.node_type = np.array(f["node_type"])[0:1]
            
        # 根据SpringMassSystemWarp 类初始化需要的参数从meta中读取所需要的参数
        self.dt = meta['dt']
        self.drag_damping = meta['drag_damping']
        self.collide_object_elas = meta['collide_object_elas']
        self.collide_object_fric = meta['collide_object_fric']
        self.collide_elas = meta['collide_elas']
        self.collide_fric = meta['collide_fric']
        self.collision_dist = meta['collision_dist']
        self.dashpot_damping = meta['dashpot_damping']
        self.num_object_points = meta['num_object_points']
        self.num_surface_points = meta['num_surface_points']
        self.num_original_points = meta['num_original_points']
        self.controller_radius = meta['controller_radius']
        self.object_radius = meta['object_radius']
        self.normalization_info = meta['normalization_info']
        self.train_frame = meta['train_frame']
        self.test_frame = meta['test_frame'] - meta['train_frame']

        self.spring_Y_max = cfg.spring_Y_max
        self.spring_Y_min = cfg.spring_Y_min
        self.max_radius = max(self.object_radius, self.controller_radius)

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

        
        # for name in field_names:
        #     fields[name] = fields[name][self.strided_idx]
       
        self._cal_multi_mesh()
        super(MeshGeneralDataset,self).__init__(root)
        self.L = self.L - 1

    def _read_normalization_info(self, in_normal_feature_list, out_normal_feature_list, roll_normal_feature_list):
        # collect in normalization
        for i, fea in enumerate(in_normal_feature_list):
            temp_std = torch.tensor(self.normalization_info[fea]['std'], dtype=torch.float)
            temp_mean = torch.tensor(self.normalization_info[fea]['mean'], dtype=torch.float)
            if i == 0:
                self.std_in = temp_std
                self.mean_in = temp_mean
            else:
                self.std_in = torch.cat((self.std_in, temp_std), dim=-1)
                self.mean_in = torch.cat((self.mean_in, temp_mean), dim=-1)
        # collect out normalization
        for i, fea in enumerate(out_normal_feature_list):
            temp_std = torch.tensor(self.normalization_info[fea]['std'], dtype=torch.float)
            temp_mean = torch.tensor(self.normalization_info[fea]['mean'], dtype=torch.float)
            if i == 0:
                self.std_out = temp_std
                self.mean_out = temp_mean
            else:
                self.std_out = torch.cat((self.std_out, temp_std), dim=-1)
                self.mean_out = torch.cat((self.mean_out, temp_mean), dim=-1)
        # collect roll-out normalization
        self.roll_l = 0
        for i, fea in enumerate(roll_normal_feature_list):
            temp_std = torch.tensor(self.normalization_info[fea]['std'], dtype=torch.float)
            self.roll_l += temp_std.shape[-1]
        # NOTE assume/let all leading features align with the list ordering here
        self.in_norm_l = 3
        self.out_norm_l = 3

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

    def _normalize(self, node_pos, edge_spring, spring_rest_length):
        device = node_pos.device
        node_pos = node_pos.clone()

        node_pos[..., :self.in_norm_l] = (node_pos[..., :self.in_norm_l] - self.mean_in[None, :self.in_norm_l].to(device)) / self.std_in[None, :self.in_norm_l].to(device)
        edge_spring = (torch.exp(edge_spring) - self.spring_Y_min) / (self.spring_Y_max - self.spring_Y_min)

        spring_rest_length = spring_rest_length / self.max_radius

        return node_pos, edge_spring, spring_rest_length
    
    def _denormalize(self, node_pos, edge_spring, spring_rest_length):
        device = node_pos.device
        node_pos = node_pos.clone()

        node_pos[..., :self.out_norm_l] = node_pos[..., :self.out_norm_l] * self.std_out.to(device) + self.mean_out.to(device)
        edge_spring = edge_spring * (self.spring_Y_max - self.spring_Y_min) + self.spring_Y_min
        
        spring_rest_length = spring_rest_length * self.max_radius
        return node_pos, edge_spring, spring_rest_length

    def _preprocess(self, node_pos, node_vel, node_mass, 
                    log_spring_Y, spring_rest_length, 
                    drag_damping, 
                    spring_force, dashpot_force, overall_forces, 
                    device, mode='train'):
        T, num_node, geo_dim = node_pos.shape

        # normalization for node_pos 和 log_spring_Y, spring_reset_length
        # TODO : add normalization for edge features

        node_pos, spring_Y, spring_rest_length = self._normalize(node_pos, 
                                                       log_spring_Y, 
                                                       spring_rest_length)

        # recalculate it for surface and interior nodes later
        node_damping = (node_vel * drag_damping).clone()

        # node_type : 0 object point, 1 surface point, 2 interior point, 3 controller point
        node_type = torch.LongTensor(self.node_type).to(node_pos.device)

        # concat historical position and velocity of every node together as input, and the target is the next step position, velocity is optional to predict
        node_info = torch.cat((node_pos, node_vel, node_damping), dim=-1)
        
        node_info_his = node_info[:-1]
        node_info_tar = node_info[1:]

        node_inp_info = torch.cat([node_info_his, node_info_tar], dim=-1)
     
        node_mass = node_mass[None, :, None].repeat(T-1, 1, 1)
        node_type = node_type.repeat(T-1, 1, 1)

        # [node_pos, node_vel] * T, node_ 12:13 node_mass 13:16 node_damping, 16: type
        node_inp_info = torch.cat((node_inp_info, node_mass, node_type), dim=-1)

        # concat spring_Y, spring_reset_length, dashpot_damping as edge features, and repeat for both directions
        edge_in_info = torch.cat((spring_force, dashpot_force), dim=-1)[:-1]

        spring_Y = spring_Y[None, :].repeat(T-1, 1, 1)
        spring_rest_length = spring_rest_length[None, :].repeat(T-1, 1, 1)
        # spring_dashpot_damping = torch.ones_like(spring_Y) * self.dashpot_damping  
        
        edge_in_info = torch.cat((edge_in_info, spring_Y, spring_rest_length), dim = -1)

        # enhance by noise level
        if self.noise_shuffle:
            # collect special nodes
            no_noise_node = (node_type == 3).bool() # dont add noise to controller nodes
            noise_base = torch.ones_like(node_info_tar)
            noise_base[:, :] = self.noise_level
            noise = torch.normal(0.0, noise_base)

            # for dirichelet nodes, the noise is zero
            noise = torch.where(no_noise_node, torch.zeros_like(noise), noise)
            node_inp_info += noise
            node_info_tar += (1.0 - self.noise_gamma) * noise

        # Load data to GPU
        node_inp_info = node_inp_info.to(device)
        edge_in_info = edge_in_info.to(device)

        # Input, and target, but in our case, we have to targets, one for node position and velocity, another egde prediction
        return node_inp_info, edge_in_info
