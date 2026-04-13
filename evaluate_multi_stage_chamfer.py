import glob
import pickle
import json
import torch
import csv
import numpy as np
import os
from pytorch3d.loss import chamfer_distance
from scipy.spatial import KDTree
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

# 配置参数
method_case_name = 'End2End_Reduction'  # 'neural_spring_field'  # 'evospring' # 'End2End' # End2End_Reduction
prediction_dir = f"./res/{method_case_name}/"
base_path = "./data/different_types/"
output_file = "results/final_results.csv"

# 多阶段评估配置
NUM_STAGES = 2  # 假设有 5 个下采样阶段

if not os.path.exists("results"):
    os.makedirs("results")


def evaluate_prediction(
    start_frame,
    end_frame,
    vertices,
    object_points,
    object_visibilities,
    object_motions_valid,
    num_original_points,
    num_surface_points,
):
    """
    评估预测轨迹与真值物体点云之间的 Chamfer Distance
    """
    chamfer_errors = []

    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(object_points, torch.Tensor):
        object_points = torch.tensor(object_points, dtype=torch.float32)
    if not isinstance(object_visibilities, torch.Tensor):
        object_visibilities = torch.tensor(object_visibilities, dtype=torch.bool)
    if not isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = torch.tensor(object_motions_valid, dtype=torch.bool)

    for frame_idx in range(start_frame, end_frame):
        x = vertices[frame_idx]
        current_object_points = object_points[frame_idx]
        current_object_visibilities = object_visibilities[frame_idx]
        # The motion valid indicates if the tracking is valid from prev_frame
        current_object_motions_valid = object_motions_valid[frame_idx - 1]

        # Compute the single-direction chamfer loss for the object points
        chamfer_object_points = current_object_points[current_object_visibilities]
        chamfer_x = x[:num_surface_points]
        # The GT chamfer_object_points can be partial,first find the nearest in second
        chamfer_error = chamfer_distance(
            chamfer_object_points.unsqueeze(0),
            chamfer_x.unsqueeze(0),
            single_directional=True,
            norm=1,  # Get the L1 distance
        )[0]

        chamfer_errors.append(chamfer_error.item())

    chamfer_errors = np.array(chamfer_errors)

    results = {
        "frame_len": len(chamfer_errors),
        "chamfer_error": np.mean(chamfer_errors),
    }

    return results


def load_stage_data(dir_name, stage_idx):
    """
    从指定阶段加载轨迹数据和顶点索引
    返回：vertices (所有 level 的列表), gt_indices (如果存在)
    """
    traj_path = f"{dir_name}/trajectories/stage_{stage_idx}_best_trajectory.pkl"
    
    if not os.path.exists(traj_path):
        print(f"Warning: Trajectory file not found: {traj_path}")
        return None, None
    
    with open(traj_path, "rb") as f:
        data = pickle.load(f)
    
    # vertices 现在是一个列表，包含所有 level 的轨迹
    vertices = data.get('vertices', None)  # List of [T, N_level, 3] for each level
    gt_indices = data.get('gt_indices', None)  # 真值下采样索引
    
    return vertices, gt_indices


def apply_downsample_to_gt(gt_data, stage_idx, downsample_method='cluster'):
    """
    对真值点云应用下采样，使其与预测结果的分辨率匹配
    
    Args:
        gt_data: 真值点云数据（可以是 vertices 或 object_points）
        stage_idx: 阶段索引
        downsample_method: 下采样方法 ('cluster', 'random', 'kdtree')
    
    Returns:
        downsampled_data: 下采样后的点云
    """
    # 如果 gt_data 是列表（多帧），对每一帧应用下采样
    if isinstance(gt_data, list) or (isinstance(gt_data, np.ndarray) and gt_data.ndim > 2):
        downsampled_frames = []
        for frame_idx, frame_data in enumerate(gt_data):
            downsampled_frames.append(downsample_frame(frame_data, stage_idx, downsample_method))
        return downsampled_frames
    else:
        return downsample_frame(gt_data, stage_idx, downsample_method)


def downsample_frame(frame_data, stage_idx, method='cluster'):
    """
    对单帧点云进行下采样
    """
    if method == 'kdtree':
        # 使用 KDTree 找到最近的点
        # 这里需要根据实际的下采样索引来提取
        # 由于我们不知道具体的下采样索引，使用简单的间隔采样
        num_points = frame_data.shape[0]
        # 假设每个阶段下采样率为 0.7（可根据实际情况调整）
        sample_rate = 0.7 ** stage_idx
        num_sample = max(int(num_points * sample_rate), 10)
        
        # 均匀采样
        indices = np.linspace(0, num_points - 1, num_sample, dtype=int)
        return frame_data[indices]
    
    elif method == 'cluster':
        # 简单的聚类下采样（每 n 个点取平均）
        num_points = frame_data.shape[0]
        sample_rate = 0.7 ** stage_idx
        num_sample = max(int(num_points * sample_rate), 10)
        cluster_size = num_points // num_sample
        
        downsampled = []
        for i in range(num_sample):
            start_idx = i * cluster_size
            end_idx = min((i + 1) * cluster_size, num_points)
            cluster_mean = frame_data[start_idx:end_idx].mean(axis=0)
            downsampled.append(cluster_mean)
        
        return np.array(downsampled)
    
    else:
        # 默认：简单间隔采样
        num_points = frame_data.shape[0]
        sample_rate = 0.7 ** stage_idx
        num_sample = max(int(num_points * sample_rate), 10)
        indices = np.linspace(0, num_points - 1, num_sample, dtype=int)
        return frame_data[indices]


def evaluate_chamfer_distance(pred_vertices, gt_object_points, gt_visibilities, 
                               num_original_points, stage_idx=None):
    """
    计算 Chamfer Distance between predicted vertices and ground truth object points
    
    Args:
        pred_vertices: 预测的顶点轨迹 [T, N, 3]
        gt_object_points: 真值物体点云 [T, M, 3]
        gt_visibilities: 真值可见性掩码 [T, M]
        num_original_points: 原始表面点数量
        stage_idx: 阶段索引（用于调整评估的点云范围）
    
    Returns:
        chamfer_errors: 每帧的 Chamfer 误差列表
    """
    chamfer_errors = []
    
    if not isinstance(pred_vertices, torch.Tensor):
        pred_vertices = torch.tensor(pred_vertices, dtype=torch.float32)
    if not isinstance(gt_object_points, torch.Tensor):
        gt_object_points = torch.tensor(gt_object_points, dtype=torch.float32)
    if not isinstance(gt_visibilities, torch.Tensor):
        gt_visibilities = torch.tensor(gt_visibilities, dtype=torch.bool)
    
    for frame_idx in range(len(pred_vertices)):
        pred_frame = pred_vertices[frame_idx]
        gt_frame = gt_object_points[frame_idx]
        gt_vis = gt_visibilities[frame_idx]
        
        # 获取可见的真值点
        visible_gt_points = gt_frame[gt_vis]
        
        if len(visible_gt_points) == 0 or len(pred_frame) == 0:
            chamfer_errors.append(0.0)
            continue
        
        # 计算双向 Chamfer Distance
        # 从 GT 到 Pred 的距离
        tree_pred = KDTree(pred_frame.cpu().numpy())
        dist_gt_to_pred, _ = tree_pred.query(visible_gt_points.cpu().numpy(), k=1)
        
        # 从 Pred 到 GT 的距离
        tree_gt = KDTree(visible_gt_points.cpu().numpy())
        dist_pred_to_gt, _ = tree_gt.query(pred_frame.cpu().numpy(), k=1)
        
        # Chamfer distance = mean(dist_gt_to_pred) + mean(dist_pred_to_gt)
        chamfer_dist = np.mean(dist_gt_to_pred) + np.mean(dist_pred_to_gt)
        chamfer_errors.append(chamfer_dist)
    
    return chamfer_errors


def evaluate_multi_stage():
    """
    多阶段 Chamfer 评估主函数
    评估每个阶段的 Chamfer Distance
    """
    results = []
    
    dir_names = glob.glob(f"{prediction_dir}/*")
    for dir_name in dir_names:
        case_name = dir_name.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"Processing {case_name}!!!!!!!!!!!!!!!")
        print(f"{'='*60}\n")
        
        # 读取数据分割信息
        with open(f"{base_path}/{case_name}/split.json", "r") as f:
            split = json.load(f)
        frame_len = split["frame_len"]
        train_frame = split["train"][1]
        test_frame = split["test"][1]
        
        # 读取真值数据
        with open(f"{base_path}/{case_name}/final_data.pkl", "rb") as f:
            data = pickle.load(f)
        
        object_points = data["object_points"]
        object_visibilities = data["object_visibilities"]
        object_motions_valid = data["object_motions_valid"]
        num_original_points = object_points.shape[1]
        num_surface_points = num_original_points + data["surface_points"].shape[0]
        
        case_results = {
            'case_name': case_name,
            'stages': []
        }
        
        # 遍历所有阶段
        for stage_idx in range(NUM_STAGES):
            print(f"\n--- Evaluating Stage {stage_idx} ---")
            
            # 加载当前阶段的预测轨迹
            vertices, pred_indices = load_stage_data(dir_name, stage_idx)
            
            if vertices is None:
                print(f"Skipping stage {stage_idx} due to missing trajectory data")
                case_results['stages'].append({
                    'stage': stage_idx,
                    'train_chamfer_error': None,
                    'test_chamfer_error': None,
                    'train_frame_num': 0,
                    'test_frame_num': 0
                })
                continue
            
            # vertices 现在是一个列表，包含所有 level 的轨迹
            # 需要评估所有 level 的结果
            if isinstance(vertices, list):
                num_levels = len(vertices)
                print(f"Stage {stage_idx} has {num_levels} levels")
                
                # 评估每个 level
                level_results = []
                for level_idx, level_vertices in enumerate(vertices):
                    print(f"  Evaluating Level {level_idx}...")
                    
                    # 使用第一层（level 0）的 GT 数据进行评估
                    # 因为 GT 数据是完整的点云，而预测结果是下采样后的
                    results_train = evaluate_prediction(
                        1,
                        train_frame,
                        level_vertices,
                        object_points,
                        object_visibilities,
                        object_motions_valid,
                        num_original_points,
                        num_surface_points,
                    )
                    
                    results_test = evaluate_prediction(
                        train_frame,
                        test_frame,
                        level_vertices,
                        object_points,
                        object_visibilities,
                        object_motions_valid,
                        num_original_points,
                        num_surface_points,
                    )
                    
                    level_results.append({
                        'level': level_idx,
                        'train_chamfer_error': results_train["chamfer_error"],
                        'test_chamfer_error': results_test["chamfer_error"],
                        'train_frame_num': results_train['frame_len'],
                        'test_frame_num': results_test['frame_len']
                    })
                    
                    print(f"    Level {level_idx}: Train Chamfer Error: {results_train['chamfer_error']:.6f}, Test Chamfer Error: {results_test['chamfer_error']:.6f}")
                
                # 计算所有 level 的平均误差
                avg_train_error = np.mean([r['train_chamfer_error'] for r in level_results])
                avg_test_error = np.mean([r['test_chamfer_error'] for r in level_results])
                
                print(f"Stage {stage_idx}:")
                print(f"  Average Train Chamfer Error: {avg_train_error:.6f}")
                print(f"  Average Test Chamfer Error: {avg_test_error:.6f}")
                
                # 保存每个 level 的结果
                case_results['stages'].append({
                    'stage': stage_idx,
                    'num_levels': num_levels,
                    'level_results': level_results,
                    'train_chamfer_error': avg_train_error,
                    'test_chamfer_error': avg_test_error,
                    'train_frame_num': level_results[0]['train_frame_num'],
                    'test_frame_num': level_results[0]['test_frame_num']
                })
            else:
                # 向后兼容：如果 vertices 不是列表，按原方式处理
                results_train = evaluate_prediction(
                    1,
                    train_frame,
                    vertices,
                    object_points,
                    object_visibilities,
                    object_motions_valid,
                    num_original_points,
                    num_surface_points,
                )
                
                results_test = evaluate_prediction(
                    train_frame,
                    test_frame,
                    vertices,
                    object_points,
                    object_visibilities,
                    object_motions_valid,
                    num_original_points,
                    num_surface_points,
                )
                
                train_chamfer_error = results_train["chamfer_error"]
                test_chamfer_error = results_test["chamfer_error"]
                
                print(f"Stage {stage_idx}:")
                print(f"  Train Chamfer Error: {train_chamfer_error:.6f} ({results_train['frame_len']} frames)")
                print(f"  Test Chamfer Error: {test_chamfer_error:.6f} ({results_test['frame_len']} frames)")
                
                case_results['stages'].append({
                    'stage': stage_idx,
                    'train_chamfer_error': train_chamfer_error,
                    'test_chamfer_error': test_chamfer_error,
                    'train_frame_num': results_train['frame_len'],
                    'test_frame_num': results_test['frame_len']
                })
        
        results.append(case_results)
    
    return results


def write_results_to_csv(results):
    """
    将多阶段评估结果写入 CSV 文件
    每个 stage 的每个 level 都单独列出一行
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 写入表头
        header = [
            "Case Name",
            "Stage",
            "Level",
            "Train Frame Num",
            "Train Chamfer Error",
            "Test Frame Num", 
            "Test Chamfer Error"
        ]
        writer.writerow(header)
        
        # 写入数据 - 每个 level 单独一行
        for case_result in results:
            case_name = case_result['case_name']
            for stage_result in case_result['stages']:
                stage_idx = stage_result.get('stage', 0)
                
                # 检查是否有 level_results（多 level 情况）
                if 'level_results' in stage_result:
                    for level_result in stage_result['level_results']:
                        row = [
                            case_name,
                            stage_idx,
                            level_result.get('level', 0),
                            level_result.get('train_frame_num', 'N/A'),
                            level_result.get('train_chamfer_error', 'N/A'),
                            level_result.get('test_frame_num', 'N/A'),
                            level_result.get('test_chamfer_error', 'N/A')
                        ]
                        writer.writerow(row)
                else:
                    # 向后兼容：没有 level 区分的旧格式
                    row = [
                        case_name,
                        stage_idx,
                        0,  # 默认 level 为 0
                        stage_result.get('train_frame_num', 'N/A'),
                        stage_result.get('train_chamfer_error', 'N/A'),
                        stage_result.get('test_frame_num', 'N/A'),
                        stage_result.get('test_chamfer_error', 'N/A')
                    ]
                    writer.writerow(row)
    
    print(f"\nResults saved to {output_file}")


def visualize_stage0_trajectory(case_name, vertices, gt_object_points, gt_visibilities, output_dir=None):
    """
    Visualize stage 0 trajectory comparison between ground truth and prediction.
    Creates a video showing GT object points and corresponding predicted points.
    
    Args:
        case_name: Case name for output filename
        vertices: Predicted trajectory [T, N, 3] or list of levels
        gt_object_points: Ground truth object points [T, M, 3]
        gt_visibilities: GT visibility mask [T, M]
        output_dir: Output directory (default: ./visualization)
    """
    if output_dir is None:
        output_dir = "./visualization"
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{case_name}_stage0_trajectory_comparison.mp4')
    
    # Get frame range
    frame_len = gt_object_points.shape[0]
    
    print(f"Starting stage 0 trajectory visualization for {case_name}...")
    
    # Use level 0 vertices if vertices is a list
    if isinstance(vertices, list):
        level_vertices = vertices[0]
        print(f"Using level 0 vertices from {len(vertices)} levels")
    else:
        level_vertices = vertices
    
    # Create figure with two subplots (GT and Pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})
    fig.suptitle(f'Stage 0: Ground Truth vs Prediction - {case_name}', fontsize=16)
    
    # Compute axis limits from all data
    all_verts = []
    for t in range(min(frame_len, len(level_vertices))):
        gt_valid = gt_object_points[t][gt_visibilities[t]]
        if len(gt_valid) > 0:
            all_verts.append(gt_valid)
        all_verts.append(level_vertices[t])
    all_verts = np.vstack(all_verts)
    
    x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
    y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()
    z_min, z_max = all_verts[:, 2].min(), all_verts[:, 2].max()
    
    margin = 0.15
    x_margin = max(margin * (x_max - x_min), 0.05)
    y_margin = max(margin * (y_max - y_min), 0.05)
    z_margin = max(margin * (z_max - z_min), 0.05)
    
    # Create temp directory for frames
    temp_images_dir = os.path.join(output_dir, f'temp_{case_name}_stage0_traj')
    os.makedirs(temp_images_dir, exist_ok=True)
    
    # Render each frame
    for t in tqdm(range(frame_len), desc="Rendering stage 0 trajectory"):
        ax1.clear()
        ax2.clear()
        
        # Plot GT - only visible points
        gt_valid = gt_object_points[t][gt_visibilities[t]]
        if len(gt_valid) > 0:
            ax1.scatter(gt_valid[:, 0], gt_valid[:, 1], gt_valid[:, 2], 
                       c='green', s=10, alpha=0.8, label='Ground Truth')
        ax1.set_xlim(x_min - x_margin, x_max + x_margin)
        ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        ax1.set_zlim(z_min - z_margin, z_max + z_margin)
        ax1.set_title(f'Ground Truth (Frame {t})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Plot Prediction
        pred_verts = level_vertices[t]
        ax2.scatter(pred_verts[:, 0], pred_verts[:, 1], pred_verts[:, 2], 
                   c='blue', s=10, alpha=0.8, label='Prediction')
        ax2.set_xlim(x_min - x_margin, x_max + x_margin)
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)
        ax2.set_zlim(z_min - z_margin, z_max + z_margin)
        ax2.set_title(f'Prediction (Frame {t})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Save frame
        frame_path = os.path.join(temp_images_dir, f'frame_{t:05d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
    
    plt.close(fig)
    
    # Convert images to video
    _images_to_video(temp_images_dir, output_path)
    
    # Cleanup
    _cleanup_temp_dir(temp_images_dir)
    
    print(f"Stage 0 trajectory comparison video saved to: {output_path}")
    return output_path


def _images_to_video(images_dir, output_path):
    """Convert a directory of PNG images to MP4 video."""
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    
    if len(image_files) == 0:
        print(f"No images found in {images_dir}")
        return
    
    # Read first image to get dimensions
    first_frame = cv2.imread(os.path.join(images_dir, image_files[0]))
    height, width = first_frame.shape[:2]
    
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                             30, (width, height))
    
    for image_file in tqdm(image_files, desc="Creating video"):
        image_path = os.path.join(images_dir, image_file)
        frame = cv2.imread(image_path)
        writer.write(frame)
    
    writer.release()
    print(f"Video saved: {output_path}")


def _cleanup_temp_dir(temp_dir):
    """Remove temporary directory and its contents."""
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def visualize_trajectory(case_name, vertices, gt_object_points, gt_visibilities, stage_idx, level_idx, output_dir=None):
    """
    Visualize trajectory comparison between ground truth and prediction.
    
    Args:
        case_name: Case name for output filename
        vertices: Predicted trajectory [T, N, 3]
        gt_object_points: Ground truth object points [T, M, 3]
        gt_visibilities: GT visibility mask [T, M]
        stage_idx: Stage index for output filename
        level_idx: Level index for output filename
        output_dir: Output directory (default: ./visualization)
    """
    if output_dir is None:
        output_dir = "./visualization"
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{case_name}_stage{stage_idx}_level{level_idx}_comparison.mp4')
    
    # Get frame range
    frame_len = gt_object_points.shape[0]
    
    print(f"Starting stage {stage_idx} level {level_idx} trajectory visualization for {case_name}...")
    
    # Create figure with two subplots (GT and Pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})
    fig.suptitle(f'Stage {stage_idx} Level {level_idx}: Ground Truth vs Prediction - {case_name}', fontsize=16)
    
    # Compute axis limits from all data
    all_verts = []
    for t in range(min(frame_len, len(vertices))):
        gt_valid = gt_object_points[t][gt_visibilities[t]]
        if len(gt_valid) > 0:
            all_verts.append(gt_valid)
        all_verts.append(vertices[t])
    all_verts = np.vstack(all_verts)
    
    x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
    y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()
    z_min, z_max = all_verts[:, 2].min(), all_verts[:, 2].max()
    
    margin = 0.15
    x_margin = max(margin * (x_max - x_min), 0.05)
    y_margin = max(margin * (y_max - y_min), 0.05)
    z_margin = max(margin * (z_max - z_min), 0.05)
    
    # Create temp directory for frames
    temp_images_dir = os.path.join(output_dir, f'temp_{case_name}_stage{stage_idx}_level{level_idx}')
    os.makedirs(temp_images_dir, exist_ok=True)
    
    # Render each frame
    for t in tqdm(range(frame_len), desc=f"Rendering stage {stage_idx} level {level_idx}"):
        ax1.clear()
        ax2.clear()
        
        # Plot GT - only visible points
        gt_valid = gt_object_points[t][gt_visibilities[t]]
        if len(gt_valid) > 0:
            ax1.scatter(gt_valid[:, 0], gt_valid[:, 1], gt_valid[:, 2], 
                       c='green', s=10, alpha=0.8, label='Ground Truth')
        ax1.set_xlim(x_min - x_margin, x_max + x_margin)
        ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        ax1.set_zlim(z_min - z_margin, z_max + z_margin)
        ax1.set_title(f'Ground Truth (Frame {t})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Plot Prediction
        pred_verts = vertices[t]
        ax2.scatter(pred_verts[:, 0], pred_verts[:, 1], pred_verts[:, 2], 
                   c='blue', s=10, alpha=0.8, label='Prediction')
        ax2.set_xlim(x_min - x_margin, x_max + x_margin)
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)
        ax2.set_zlim(z_min - z_margin, z_max + z_margin)
        ax2.set_title(f'Prediction (Frame {t})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Save frame
        frame_path = os.path.join(temp_images_dir, f'frame_{t:05d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
    
    plt.close(fig)
    
    # Convert images to video
    _images_to_video(temp_images_dir, output_path)
    
    # Cleanup
    _cleanup_temp_dir(temp_images_dir)
    
    print(f"Stage {stage_idx} Level {level_idx} comparison video saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    print(f"Starting multi-stage Chamfer evaluation for {method_case_name}...")
    print(f"Prediction directory: {prediction_dir}")
    print(f"Base path: {base_path}")
    print(f"Number of stages: {NUM_STAGES}")
    
    results = evaluate_multi_stage()
    write_results_to_csv(results)
    
    # # Visualize stage 0 level 0 and stage 1 level 1 trajectories for each case
    # print("\nGenerating trajectory visualizations...")
    # dir_names = glob.glob(f"{prediction_dir}/*")
    # for dir_name in dir_names:
    #     case_name = dir_name.split("/")[-1]
    #     print(f"\nVisualizing {case_name}...")
        
    #     # Load GT data
    #     with open(f"{base_path}/{case_name}/final_data.pkl", "rb") as f:
    #         data = pickle.load(f)
        
    #     gt_object_points = data["object_points"]
    #     gt_visibilities = data["object_visibilities"]
        
    #     # Visualize stage 0 level 0
    #     vertices_0, _ = load_stage_data(dir_name, 0)
    #     if vertices_0 is not None and isinstance(vertices_0, list) and len(vertices_0) > 0:
    #         visualize_trajectory(case_name, vertices_0[0], gt_object_points, gt_visibilities, 
    #                             stage_idx=0, level_idx=0)
        
    #     # Visualize stage 1 level 1
    #     vertices_1, _ = load_stage_data(dir_name, 1)
    #     if vertices_1 is not None and isinstance(vertices_1, list) and len(vertices_1) > 1:
    #         visualize_trajectory(case_name, vertices_1[1], gt_object_points, gt_visibilities, 
    #                             stage_idx=1, level_idx=1)
    
    # print("\nEvaluation and visualization completed!")
