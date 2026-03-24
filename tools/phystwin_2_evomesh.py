import pickle
import h5py
import numpy as np
import open3d as o3d
import json
import os
import torch

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def compute_normalization_info(data_dict):
    """
    计算每个字段的 Mean 和 Std。
    """
    norm_info = {}
    for name, data in data_dict.items():
        if name == 'cells' or name == 'node_type' or name == 'init_spring_Y':
            continue  # 跳过非浮点数据
        data_float = data.astype(np.float64)
        # 对 (T, N, C) 数据，沿着 T 和 N 聚合 (0, 1)
        if data.ndim >= 2:
            axis = tuple(range(data.ndim - 1))
        else:
            axis = 0
            
        mean_val = np.mean(data_float, axis=axis)
        std_val = np.std(data_float, axis=axis)
        
        if mean_val.ndim == 0:
            mean_output = mean_val.item()
            std_output = std_val.item()
        else:
            mean_output = mean_val.tolist()
            std_output = std_val.tolist()
            
        norm_info[name] = {
            "mean": mean_output,
            "std": std_output
        }
    return norm_info

def create_optimization_dataset(opt_pkl_path, connect_params_pkl_path, mech_file, split_json_path, output_h5_path, output_json_path):
    print(f"--- 开始处理 Optimization 数据 ---")
    
    # ==============================
    # 1. 读取数据
    # ==============================
    print(f"读取数据: {opt_pkl_path}")
    zero_grad_opt_data = load_pickle(opt_pkl_path)
    
    # 获取两组点云 (T, N, 3)
    # Original object points captured by Depth Camera
    object_points = zero_grad_opt_data['object_points']
    # Controller points captured by Depth Camera
    controller_points = zero_grad_opt_data['controller_points']

    # surface points and interior points generate from shape prior, only for first frame
    surface_points = zero_grad_opt_data['surface_points']
    interior_points = zero_grad_opt_data['interior_points']

    object_colors = zero_grad_opt_data["object_colors"]
    object_visibilities = zero_grad_opt_data["object_visibilities"]
    object_motions_valid = zero_grad_opt_data["object_motions_valid"]

    T, N_obj, D = object_points.shape
    _, N_ctrl, _ = controller_points.shape

    N_surface = surface_points.shape[0]
    N_interior = interior_points.shape[0]
    
    print(f"Object points: {object_points.shape}")
    print(f"Controller points: {controller_points.shape}")
    print(f"Surface points: {surface_points.shape}")
    print(f"Interior points: {interior_points.shape}")
    
    first_frame_structure_point = \
        np.concatenate([object_points[0], surface_points, interior_points], axis=0)

    world_pos = np.concatenate([object_points, 
                                surface_points[None, :, :].repeat(T, axis=0),
                                interior_points[None, :, :].repeat(T, axis=0),
                                controller_points], axis=1)
    N_total = N_obj + N_ctrl + N_surface + N_interior
    print(f"合并后 World Pos: {first_frame_structure_point.shape}")

    # ==============================
    # 2. 读取图参数
    # ==============================
    print(f"读取参数: {connect_params_pkl_path}")
    connect_params = load_pickle(connect_params_pkl_path)
    
    # Object-Object 参数
    obj_radius = connect_params['object_radius']
    obj_max_nn = connect_params['object_max_neighbours']

    # Controller-Object 参数
    ctrl_radius = connect_params['controller_radius']
    ctrl_max_nn = connect_params['controller_max_neighbours']
 
    # dashdot_damping 参数
    dashpot_damping = connect_params['dashpot_damping']
    drag_damping = connect_params['drag_damping']
    import pdb; pdb.set_trace()
    global_spring_Y = connect_params['global_spring_Y']

    # collide 参数
    collide_elas = connect_params['collide_elas']
    collide_fric = connect_params['collide_fric']
    collide_object_elas = connect_params['collide_object_elas']
    collide_object_fric = connect_params['collide_object_fric']
    collision_dist = connect_params['collision_dist']
    
    print(f"Obj-Obj 参数: R={obj_radius}, Max={obj_max_nn}")
    print(f"Ctrl-Obj 参数: R={ctrl_radius}, Max={ctrl_max_nn}")

    # ==============================
    # 3. 构建混合图 (基于第0帧)
    # ==============================
    print("正在构建静态图连接 (Frame 0)...")
    
    # 获取第0帧数据
    obj_p0 = first_frame_structure_point.astype(np.float64)
    ctrl_p0 = controller_points[0].astype(np.float64)
    
    # --- 核心逻辑 ---
    # 我们只对 Object 建立 KDTree，因为无论是 Object 还是 Controller，
    # 都是去寻找周围的 Object 建立连接 (参考 trainer_warp 逻辑)
    
    pcd_obj = o3d.geometry.PointCloud()
    pcd_obj.points = o3d.utility.Vector3dVector(obj_p0)
    obj_tree = o3d.geometry.KDTreeFlann(pcd_obj)
    
    edges_list = []
    
    # Connect the springs of the objects first
    points = np.asarray(pcd_obj.points)
    spring_flags = np.zeros((len(points), len(points)))

    rest_lengths = []
    for i in range(len(points)):
        [k, idx, _] = obj_tree.search_hybrid_vector_3d(
            points[i], obj_radius, obj_max_nn
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
                edges_list.append([i, j])
                rest_lengths.append(np.linalg.norm(points[i] - points[j]))

    # Part B: Controller 连接 Object (src: Controller -> dst: Object)
    # Controller 的全局索引从 N_obj 开始
    # 索引范围: N_obj ~ N_total-1
    num_object_points = len(points)
    points = np.concatenate([points, ctrl_p0], axis=0)
    for i in range(len(ctrl_p0)):
        [k, idx, _] = obj_tree.search_hybrid_vector_3d(
            ctrl_p0[i],
            ctrl_radius,
            ctrl_max_nn,
        )
        for j in idx:
            edges_list.append([num_object_points + i, j])
            rest_lengths.append(
                np.linalg.norm(ctrl_p0[i] - points[j])
            )

    # 转为 Numpy
    if len(edges_list) > 0:
        base_edges = np.array(edges_list, dtype=np.int32)
    else:
        base_edges = np.zeros((0, 2), dtype=np.int32)
        print("警告: 未生成任何边！")
        
    K = len(base_edges)
    print(f"生成边总数 (K): {K}")

    # ==============================
    # 4. 读取优化前的机械属性初始估计
    # ==============================

    print(f"读取初始机械属性估计: {connect_params_pkl_path}")

    # ==============================
    # 4. 读取优化后的机械属性
    # ==============================
    print(f"读取机械属性: {mech_file}")
    opt_spring = torch.load(mech_file, map_location='cpu')

    spring_property = opt_spring['spring_Y'].numpy().astype(np.float32)

    # assert K == spring_property.shape[0], "边数量与机械属性不匹配！"
    # ==============================
    # 5. 组装数据字典
    # ==============================
    
    # 1. World Pos (Float32)
    data_pos = world_pos.astype(np.float32)
    mesh_pos = np.concatenate([first_frame_structure_point, controller_points[0]], axis=0)[None].repeat(T, axis=0)  # Mesh Pos 只用第0帧位置，形状 (T, N_total, 3)

    # 2. Cells (Int32, Tile to T frames)
    data_cells = np.tile(base_edges[np.newaxis, :, :], (T, 1, 1)).transpose(0, 2, 1)  # 形状 (T, 2, K)
    
    # 3. Node Type (Int32)
    # 根据要求: Object设为0, Controller设为0
    # 形状 (T, N_total, 1)
    # Measured object is type 0, surface points is type 1, 
    # interior points is type 2, controller points is type 3
    data_node_type = np.zeros((T, N_total, 1), dtype=np.int32)
    data_node_type[:, N_obj:N_obj + N_surface, :] = 1 
    data_node_type[:, N_obj + N_surface: N_obj + N_surface + N_interior, :] = 2 
    data_node_type[:, N_obj + N_surface + N_interior:, :] = 3 

    # 4. Particle Speed (Float32) 形状 (T, N_total, 3), 默 认为0
    data_speed = np.zeros((T, N_total, 3), dtype=np.float32)

    # 5. masses
    masses = np.ones((T, N_total, 1), dtype=np.float32) # Follow same settings in PhysTwin
    rest_lengths = np.array(rest_lengths, dtype=np.float32)

    print("Max reset_length :{}, Min reset_length : {}".format(rest_lengths.max(), rest_lengths.min()))

    # 汇总
    data_dict = {
        "world_pos": data_pos,
        "object_points" : object_points,
        "mesh_pos": mesh_pos,
        "cells": data_cells,
        "node_type": data_node_type,
        "mass" :  masses,
        "velocities": data_speed,
        'init_spring_Y' : global_spring_Y,
        'spring_Y' : spring_property[None, :, None].repeat(T, axis=0),
        'spring_dashpot_damping' : dashpot_damping * np.ones((T, K, 1), dtype=np.float32),
        'spring_rest_length' : rest_lengths[None, :, None].repeat(T, axis=0),
        "controller_point" : data_pos[:, N_obj + N_surface + N_interior:],
        "object_colors" : object_colors,
        "object_visibilities" : object_visibilities,
        "object_motions_valid" : object_motions_valid,
    }

    # ==============================
    # 5. 计算 Normalization 并保存
    # ==============================
    print("计算 Normalization Info...")
    normalization_info = compute_normalization_info(data_dict)
    
    # special case for mass and velocities
    normalization_info['mass']['mean'] = [0]
    normalization_info['mass']['std'] = [1]
    normalization_info['velocities']['mean'] = [0]
    normalization_info['velocities']['std'] = [1]
    
    # 写入 H5
    print(f"写入 H5: {output_h5_path}")
    with h5py.File(output_h5_path, 'w') as f:
        for name, data in data_dict.items():
            f.create_dataset(name, data=data)
        
        # # 保存参数属性
        # f['cells'].attrs['obj_radius'] = obj_radius
        # f['cells'].attrs['ctrl_radius'] = ctrl_radius

    # 写入 JSON
    print(f"写入 Meta JSON: {output_json_path}")
    fps = 30.0

    # read the train/test split
    with open(split_json_path, "r") as f:
        split = json.load(f)
    train_frame = split["train"][1]
    test_frame = split["test"][1]

    meta_data = {
        "simulator": "comsol",
        'FPS': 30,
        'num_substeps': 667,
        "dt": 5e-5,
        'object_radius' : obj_radius,
        'controller_radius' : ctrl_radius,
        "collision_radius": None,
        "collide_elas" : float(collide_elas),
        "collide_fric" : float(collide_fric),
        "collide_object_elas" : float(collide_object_elas),
        "collide_object_fric" : float(collide_object_fric),
        "drag_damping" : drag_damping,
        "dashpot_damping" : dashpot_damping,
        "num_original_points" : N_obj,
        "num_surface_points" : N_obj + N_surface,
        "num_object_points" : N_obj + N_surface + N_interior,
        "collision_dist" : float(collision_dist),
        "split": split,
        "train_frame": train_frame,
        "test_frame": test_frame,
        "features": {
            name: {
                "type": "static",
                "shape": list(data.shape),
                "dtype": str(data.dtype)
            } for name, data in data_dict.items()
        },
        "field_names": list(data_dict.keys()),
        "trajectory_length": T,
        "normalization_info": normalization_info,
        "reverse_z" : True,
    }

    with open(output_json_path, 'w') as f:
        json.dump(meta_data, f, indent=4)
        
    print("--- 处理完成 ---")

if __name__ == "__main__":
    # 配置文件路径
    object_cases = os.listdir('./data/different_types')
    opt_file_path = './data/different_types'
    param_file_path = './experiments_optimization'
    mech_gt_path = './experiments'

    output_dir = './evomesh_optimization_outputs'
    os.makedirs(output_dir, exist_ok=True)
    for obj_case in object_cases:
        object_cases = 'double_stretch_sloth'
        opt_file = os.path.join(opt_file_path, obj_case, "final_data.pkl")
        params_file = os.path.join(param_file_path, obj_case, "optimal_params.pkl")
        split_file = os.path.join(opt_file_path, obj_case, "split.json")
        mech_file = os.path.join(mech_gt_path, obj_case, "train")
        for mech_gt_file in os.listdir(mech_file):
            if mech_gt_file.startswith('best'):
                mech_file = os.path.join(mech_file, mech_gt_file)
                break
        obj_output_dir = os.path.join(output_dir, obj_case)
        os.makedirs(obj_output_dir, exist_ok=True)

        print(f"Processing object: {obj_case}")
        output_h5 = os.path.join(obj_output_dir, f"init_spring_mass.h5")
        output_json = os.path.join(obj_output_dir, f"meta.json")

        create_optimization_dataset(opt_file, params_file, mech_file, split_file, output_h5, output_json)