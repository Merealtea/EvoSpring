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
output_file = "results/final_global_results.csv"

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
        current_object_motions_valid = object_motions_valid[frame_idx - 1]

        # Compute the single-direction chamfer loss for the object points
        chamfer_object_points = current_object_points[current_object_visibilities]
        chamfer_x = x[:num_surface_points]
        chamfer_error = chamfer_distance(
            chamfer_object_points.unsqueeze(0),
            chamfer_x.unsqueeze(0),
            single_directional=True,
            norm=1,
        )[0]

        chamfer_errors.append(chamfer_error.item())

    chamfer_errors = np.array(chamfer_errors)

    results = {
        "frame_len": len(chamfer_errors),
        "chamfer_error": np.mean(chamfer_errors),
    }

    return results


def load_global_trajectory(dir_name):
    """
    从全局最优轨迹文件中加载数据
    
    返回：vertices (所有 level 的列表), masses, edges, gt_indices
    """
    lastest_timestamp = sorted(os.listdir(dir_name))[-1]

    traj_path = f"{dir_name}/{lastest_timestamp}/trajectories/global_best_trajectory.pkl"
    
    if not os.path.exists(traj_path):
        print(f"Warning: Global trajectory file not found: {traj_path}")
        return None, None, None, None
    
    with open(traj_path, "rb") as f:
        data = pickle.load(f)
    
    vertices = data.get('vertices', None)  # List of [T, N_level, 3] for each level
    masses = data.get('masses', None)      # List of [N_level] for each level
    edges = data.get('edges', None)        # List of [2, E_level] for each level
    gt_indices = data.get('gt_indices', None)  # 真值下采样索引（第 0 层的）
    
    return vertices, masses, edges, gt_indices


def evaluate_chamfer_distance(pred_vertices, gt_object_points, gt_visibilities, 
                               num_original_points, num_surface_points):
    """
    计算 Chamfer Distance between predicted vertices and ground truth object points
    
    Args:
        pred_vertices: 预测的顶点轨迹 [T, N, 3]
        gt_object_points: 真值物体点云 [T, M, 3]
        gt_visibilities: 真值可见性掩码 [T, M]
        num_original_points: 原始表面点数量
        num_surface_points: 表面点 + 内部点总数
    
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


def evaluate_global_results():
    """
    全局最优结果评估主函数
    评估全局最优轨迹中每个 level 的 Chamfer Distance
    """
    results = []
    
    dir_names = glob.glob(f"{prediction_dir}/*")
    for dir_name in dir_names:
        case_name = dir_name.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"Processing {case_name}")
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
            'levels': []
        }
        
        # 加载全局最优轨迹
        vertices, masses, edges, gt_indices = load_global_trajectory(dir_name)
        
        if vertices is None:
            print(f"Skipping {case_name} due to missing trajectory data")
            continue
        
        # vertices 是一个列表，包含所有 level 的轨迹
        num_levels = len(vertices)
        print(f"Global best trajectory has {num_levels} levels")
        
        level_results = []
        for level_idx in range(num_levels):
            print(f"\n--- Evaluating Level {level_idx} ---")
            
            level_vertices = vertices[level_idx]
            
            # 评估训练集
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
            
            # 评估测试集
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
                'test_frame_num': results_test['frame_len'],
                'num_nodes': level_vertices.shape[1]
            })
            
            print(f"  Level {level_idx}:")
            print(f"    Num nodes: {level_vertices.shape[1]}")
            print(f"    Train Chamfer Error: {results_train['chamfer_error']:.6f} ({results_train['frame_len']} frames)")
            print(f"    Test Chamfer Error: {results_test['chamfer_error']:.6f} ({results_test['frame_len']} frames)")
        
        # 计算所有 level 的平均误差
        avg_train_error = np.mean([r['train_chamfer_error'] for r in level_results])
        avg_test_error = np.mean([r['test_chamfer_error'] for r in level_results])
        
        print(f"\n{'='*60}")
        print(f"Global Average Train Chamfer Error: {avg_train_error:.6f}")
        print(f"Global Average Test Chamfer Error: {avg_test_error:.6f}")
        print(f"{'='*60}\n")
        
        case_results['levels'] = level_results
        case_results['avg_train_error'] = avg_train_error
        case_results['avg_test_error'] = avg_test_error
        case_results['num_levels'] = num_levels
        
        results.append(case_results)
    
    return results


def write_results_to_csv(results):
    """
    将全局最优结果写入 CSV 文件
    每个 level 单独列出一行
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 写入表头
        header = [
            "Case Name",
            "Level",
            "Num Nodes",
            "Train Frame Num",
            "Train Chamfer Error",
            "Test Frame Num", 
            "Test Chamfer Error"
        ]
        writer.writerow(header)
        
        # 写入数据 - 每个 level 单独一行
        for case_result in results:
            case_name = case_result['case_name']
            for level_result in case_result['levels']:
                row = [
                    case_name,
                    level_result.get('level', 0),
                    level_result.get('num_nodes', 'N/A'),
                    level_result.get('train_frame_num', 'N/A'),
                    level_result.get('train_chamfer_error', 'N/A'),
                    level_result.get('test_frame_num', 'N/A'),
                    level_result.get('test_chamfer_error', 'N/A')
                ]
                writer.writerow(row)
            
            # 添加每个 case 内部所有 level 的平均行
            avg_row = [
                case_name,
                'AVG',
                'N/A',
                'N/A',
                case_result.get('avg_train_error', 'N/A'),
                'N/A',
                case_result.get('avg_test_error', 'N/A')
            ]
            writer.writerow(avg_row)
        
        # 添加一个空行分隔
        writer.writerow([])
        
        # 计算每一层所有 case 的平均值
        writer.writerow(["Layer-wise Average across all cases"])
        
        # 找到最大的 level 数量
        max_levels = max(len(case_result['levels']) for case_result in results)
        
        for level_idx in range(max_levels):
            train_errors = []
            test_errors = []
            num_nodes_list = []
            
            for case_result in results:
                if level_idx < len(case_result['levels']):
                    level_result = case_result['levels'][level_idx]
                    train_errors.append(level_result.get('train_chamfer_error', 0))
                    test_errors.append(level_result.get('test_chamfer_error', 0))
                    num_nodes_list.append(level_result.get('num_nodes', 0))
            
            avg_train_error = np.mean(train_errors) if train_errors else 0
            avg_test_error = np.mean(test_errors) if test_errors else 0
            avg_num_nodes = np.mean(num_nodes_list) if num_nodes_list else 0
            
            row = [
                "Average",
                level_idx,
                f"{avg_num_nodes:.2f}",
                "N/A",
                f"{avg_train_error:.6f}",
                "N/A",
                f"{avg_test_error:.6f}"
            ]
            writer.writerow(row)
    
    print(f"\nResults saved to {output_file}")


def visualize_global_trajectory(case_name, vertices, gt_object_points, gt_visibilities, output_dir=None):
    """
    Visualize global best trajectory comparison between ground truth and prediction.
    Creates a video showing GT object points and corresponding predicted points for all levels.
    
    Args:
        case_name: Case name for output filename
        vertices: Predicted trajectory list (each level has [T, N, 3])
        gt_object_points: Ground truth object points [T, M, 3]
        gt_visibilities: GT visibility mask [T, M]
        output_dir: Output directory (default: ./visualization)
    """
    if output_dir is None:
        output_dir = "./visualization"
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{case_name}_global_best_comparison.mp4')
    
    # Get frame range
    frame_len = gt_object_points.shape[0]
    
    print(f"Starting global best trajectory visualization for {case_name}...")
    print(f"Number of levels: {len(vertices)}")
    
    # 过滤掉含有 nan 的 level，只保留正常的 level 进行可视化
    valid_vertices = []
    valid_level_indices = []
    for level_idx, level_vertices in enumerate(vertices):
        if isinstance(level_vertices, np.ndarray):
            has_nan = np.isnan(level_vertices).any()
        else:
            has_nan = torch.isnan(level_vertices).any().item()
        
        if has_nan:
            print(f"Level {level_idx} contains NaN vertices, skipping visualization")
        else:
            valid_vertices.append(level_vertices)
            valid_level_indices.append(level_idx)
    
    # 使用过滤后的 vertices
    vertices = valid_vertices
    num_levels = len(vertices)
    
    if num_levels == 0:
        print("No valid levels to visualize (all levels contain NaN)")
        return None
    
    # Create figure with multiple subplots (one for each level + GT)
    fig, axes = plt.subplots(1, num_levels + 1, figsize=(4 * (num_levels + 1), 8), subplot_kw={'projection': '3d'})
    
    if num_levels == 0:
        axes = [axes]
    elif num_levels == 1:
        axes = list(axes)
    
    # Compute axis limits from all data
    all_verts = []
    for t in range(min(frame_len, len(vertices[0]))):
        gt_valid = gt_object_points[t][gt_visibilities[t]]
        if len(gt_valid) > 0:
            all_verts.append(gt_valid)
        for level_vertices in vertices:
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
    temp_images_dir = os.path.join(output_dir, f'temp_{case_name}_global')
    os.makedirs(temp_images_dir, exist_ok=True)
    
    # Render each frame
    for t in tqdm(range(frame_len), desc="Rendering global trajectory"):
        # Plot GT
        axes[0].clear()
        gt_valid = gt_object_points[t][gt_visibilities[t]]
        if len(gt_valid) > 0:
            axes[0].scatter(gt_valid[:, 0], gt_valid[:, 1], gt_valid[:, 2], 
                       c='green', s=10, alpha=0.8, label='Ground Truth')
        axes[0].set_xlim(x_min - x_margin, x_max + x_margin)
        axes[0].set_ylim(y_min - y_margin, y_max + y_margin)
        axes[0].set_zlim(z_min - z_margin, z_max + z_margin)
        axes[0].set_title(f'Ground Truth (Frame {t})')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_zlabel('Z')
        
        # Plot each level
        for level_idx, level_vertices in enumerate(vertices):
            ax = axes[level_idx + 1]
            ax.clear()
            
            pred_verts = level_vertices[t]
            ax.scatter(pred_verts[:, 0], pred_verts[:, 1], pred_verts[:, 2], 
                       c='blue', s=10, alpha=0.8, label=f'Level {level_idx}')
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            ax.set_zlim(z_min - z_margin, z_max + z_margin)
            ax.set_title(f'Level {level_idx} (Frame {t})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        # Save frame
        frame_path = os.path.join(temp_images_dir, f'frame_{t:05d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    # Convert images to video
    _images_to_video(temp_images_dir, output_path)
    
    # Cleanup
    _cleanup_temp_dir(temp_images_dir)
    
    print(f"Global best trajectory comparison video saved to: {output_path}")
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
                             10, (width, height))
    
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


if __name__ == "__main__":
    print(f"Starting global best Chamfer evaluation for {method_case_name}...")
    print(f"Prediction directory: {prediction_dir}")
    print(f"Base path: {base_path}")
    
    results = evaluate_global_results()
    write_results_to_csv(results)
    
    # # Visualize global best trajectories for each case
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
        
    #     # Load and visualize global best trajectory
    #     vertices, _, _, _ = load_global_trajectory(dir_name)
    #     if vertices is not None:
    #         visualize_global_trajectory(case_name, vertices, gt_object_points, gt_visibilities)
    
    # print("\nEvaluation and visualization completed!")