from datasets import MeshGeneralDataset, MeshType
import torch
import os
from helpers_bistride import generate_multi_layer_stride, SeedingHeuristic
from helpers_mesh import tetras_to_edges, triangles_to_edges, quads_to_edges, lines_to_edges
from torch_geometric.data import Data, Dataset
import numpy as np
import json
import pickle
from qqtt.utils import logger, cfg
from qqtt.data import RealData, SimpleData
import sys
from qqtt.model.diff_simulator import SpringMassSystemWarp
import open3d as o3d

class End2EndReductionDataset(MeshGeneralDataset):
    def __init__(self, root, layer_num, stride, mode='train',
                 refine_steps=[], recal_mesh=False, consist_mesh=False, 
                 object_case=None, args=None, device = None, mask_path = None,
                    velocity_path = None,):
      
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

        opt_file_path = './data/different_types'
        split_json_path = os.path.join(opt_file_path, self.object_case, "split.json")

        # read the train/test split
        with open(split_json_path, "r") as f:
            split = json.load(f)

        train_frame = split["train"][1]
        test_frame = split["test"][1] - train_frame

        if "cloth" in self.object_case or "package" in self.object_case:
            cfg.load_from_yaml("configs/phystwin_configs/cloth.yaml")
        else:
            cfg.load_from_yaml("configs/phystwin_configs/real.yaml")

        base_dir = f"experiments_optimization/{self.object_case}"
        data_path = os.path.join(opt_file_path, self.object_case, 'final_data.pkl')

        cfg.data_path = data_path
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = self.object_case
        cfg.train_frame = train_frame
        cfg.test_frame = test_frame
        # Set the intrinsic and extrinsic parameters for visualization
        with open(f"{opt_file_path}/{self.object_case}/calibrate.pkl", "rb") as f:
            c2ws = pickle.load(f)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        cfg.c2ws = np.array(c2ws)
        cfg.w2cs = np.array(w2cs)
        with open(f"{opt_file_path}/{self.object_case}/metadata.json", "r") as f:
            data = json.load(f)
        cfg.intrinsics = np.array(data["intrinsics"])
        cfg.WH = data["WH"]
        cfg.overlay_path = f"{opt_file_path}/{self.object_case}/color"

        self.init_masks = None
        self.init_velocities = None
        # Load the data
        if cfg.data_type == "real":
            self.dataset = RealData(visualize=False, save_gt=False)
            # Get the object points and controller points
            self.object_points = self.dataset.object_points
            self.object_colors = self.dataset.object_colors
            self.object_visibilities = self.dataset.object_visibilities
            self.object_motions_valid = self.dataset.object_motions_valid
            self.controller_points = self.dataset.controller_points
            self.structure_points = self.dataset.structure_points
            self.num_original_points = self.dataset.num_original_points
            self.num_surface_points = self.dataset.num_surface_points
            self.num_all_points = self.dataset.num_all_points
        elif cfg.data_type == "synthetic":
            self.dataset = SimpleData(visualize=False)
            self.object_points = self.dataset.data
            self.object_colors = None
            self.object_visibilities = None
            self.object_motions_valid = None
            self.controller_points = None
            self.structure_points = self.dataset.data[0]
            self.num_original_points = None
            self.num_surface_points = None
            self.num_all_points = len(self.dataset.data[0])
            # Prepare for the multiple object case
            if mask_path is not None:
                mask = np.load(mask_path)
                self.init_masks = torch.tensor(
                    mask, dtype=torch.float32, device=cfg.device
                )
            if velocity_path is not None:
                velocity = np.load(velocity_path)
                self.init_velocities = torch.tensor(
                    velocity, dtype=torch.float32, device=cfg.device
                )
        else:
            raise ValueError(f"Data type {cfg.data_type} not supported")

        # Initialize the vertices, springs, rest lengths and masses
        if self.controller_points is None:
            firt_frame_controller_points = None
        else:
            firt_frame_controller_points = self.controller_points[0]

        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            self.num_object_springs,
        ) = self._init_start(
            self.structure_points,
            firt_frame_controller_points,
            object_radius=cfg.object_radius,
            object_max_neighbours=cfg.object_max_neighbours,
            controller_radius=cfg.controller_radius,
            controller_max_neighbours=cfg.controller_max_neighbours,
            mask=self.init_masks,
        )
        
        # Use data from _init_start and simulator initialization

        # cells: spring connectivity topology
        self.cells = self.init_springs.cpu().numpy().T

        # node_type: 0=object, 1=surface, 2=inner, 3=controller
        num_points = len(self.init_vertices)
        self.node_type = np.zeros((1, num_points, 1), dtype=np.int32)

        if self.num_original_points is not None:
            # Object points (original points from observation)
            self.node_type[0, :self.num_original_points, 0] = 0
            # Surface points
            if self.num_surface_points is not None:
                self.node_type[0, self.num_original_points:self.num_surface_points, 0] = 1
                # Inner points
                self.node_type[0, self.num_surface_points:self.num_all_points, 0] = 2

        # Controller points
        if self.controller_points is not None:
            self.node_type[0, self.num_all_points:, 0] = 3


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

    def normalize(self, value, min, max):
        assert min < max, "The minimum value should be less than the maximum value"
        return (value - min) / (max - min)

    def denormalize(self, value, min, max):
        assert min < max, "The minimum value should be less than the maximum value"
        return value * (max - min) + min
    
    def create_spring_mass_sim(self,
                               init_vertices,
                               init_springs,
                               init_rest_lengths,
                               init_masses,
                               node_type,
                               gt_object_points,
                               gt_object_visibilities,
                               gt_object_motions_valid,
                               ):
        num_original_points = sum(node_type==0).item()
        num_surface_points = num_original_points + sum(node_type==1).item() 
        num_all_points = num_surface_points + sum(node_type==2).item()

        # Initialize parameters from cfg
        simulator = SpringMassSystemWarp(
            init_vertices,
            init_springs,
            init_rest_lengths,
            init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas,
            collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=cfg.collision_dist,
            init_velocities=torch.zeros_like(init_vertices),
            num_object_points=num_all_points,
            num_surface_points=num_surface_points,
            num_original_points=num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=gt_object_points,
            gt_object_visibilities=gt_object_visibilities,
            gt_object_motions_valid=gt_object_motions_valid,
            self_collision=cfg.self_collision,
        )
        
        return simulator