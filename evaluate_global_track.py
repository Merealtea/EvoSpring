import pickle
import glob
import csv
import json
import numpy as np
from scipy.spatial import KDTree
import os

method_case_name = 'End2End_Reduction'  # 'neural_spring_field'  # 'evospring' # 'End2End' # End2End_Reduction
prediction_path = f"./res/{method_case_name}/"
base_path = "./data/different_types/"
output_file = "results/final_global_track.csv"


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


def evaluate_prediction(start_frame, end_frame, vertices, gt_track_3d, idx, mask):
    """
    评估预测轨迹与真值 3D 跟踪点之间的误差
    
    Args:
        start_frame: 起始帧
        end_frame: 结束帧
        vertices: 预测的顶点轨迹 [T, N, 3]
        gt_track_3d: 真值 3D 跟踪点 [T, M, 3]
        idx: 预测顶点中与真值跟踪点对应的索引
        mask: 有效跟踪点的掩码
    
    Returns:
        mean_track_error: 平均跟踪误差
    """
    track_errors = []
    for frame_idx in range(start_frame, end_frame):
        # Get the new mask and see
        new_mask = ~np.isnan(gt_track_3d[frame_idx][mask]).any(axis=1)
        gt_track_points = gt_track_3d[frame_idx][mask][new_mask]
        pred_x = vertices[frame_idx][idx][new_mask]
        if len(pred_x) == 0:
            track_error = 0
        else:
            track_error = np.mean(np.linalg.norm(pred_x - gt_track_points, axis=1))
        
        track_errors.append(track_error)
    return np.mean(track_errors)


def evaluate_global_track():
    """
    全局最优轨迹的跟踪误差评估主函数
    评估每个 level 的 Track Error
    """
    results = []
    
    dir_names = glob.glob(f"{prediction_path}/*")
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
        
        # 加载全局最优轨迹
        vertices, masses, edges, gt_indices = load_global_trajectory(dir_name)
        
        if vertices is None:
            print(f"Skipping {case_name} due to missing trajectory data")
            continue
        
        # 加载真值 3D 跟踪数据
        with open(f"{base_path}/{case_name}/gt_track_3d.pkl", "rb") as f:
            gt_track_3d = pickle.load(f)
        
        # 定位有效跟踪点
        mask = ~np.isnan(gt_track_3d[0]).any(axis=1)
        
        case_results = {
            'case_name': case_name,
            'levels': []
        }
        
        # 评估每个 level
        num_levels = len(vertices)
        print(f"Global best trajectory has {num_levels} levels")
        
        for level_idx in range(num_levels):
            print(f"\n--- Evaluating Level {level_idx} ---")
            
            level_vertices = vertices[level_idx]
            
            # 使用 KDTree 找到与真值跟踪点对应的预测顶点索引
            kdtree = KDTree(level_vertices[0])
            dis, idx = kdtree.query(gt_track_3d[0][mask])
            
            # 评估训练集跟踪误差
            train_track_error = evaluate_prediction(
                1, train_frame, level_vertices, gt_track_3d, idx, mask
            )
            
            # 评估测试集跟踪误差
            test_track_error = evaluate_prediction(
                train_frame, test_frame, level_vertices, gt_track_3d, idx, mask
            )
            
            level_results = {
                'level': level_idx,
                'train_track_error': train_track_error,
                'test_track_error': test_track_error,
                'num_nodes': level_vertices.shape[1]
            }
            
            case_results['levels'].append(level_results)
            
            print(f"  Level {level_idx}:")
            print(f"    Num nodes: {level_vertices.shape[1]}")
            print(f"    Train Track Error: {train_track_error:.6f}")
            print(f"    Test Track Error: {test_track_error:.6f}")
        
        # 计算平均误差
        avg_train_error = np.mean([r['train_track_error'] for r in case_results['levels']])
        avg_test_error = np.mean([r['test_track_error'] for r in case_results['levels']])
        
        case_results['avg_train_error'] = avg_train_error
        case_results['avg_test_error'] = avg_test_error
        case_results['num_levels'] = num_levels
        
        print(f"\n{'='*60}")
        print(f"Global Average Train Track Error: {avg_train_error:.6f}")
        print(f"Global Average Test Track Error: {avg_test_error:.6f}")
        print(f"{'='*60}\n")
        
        results.append(case_results)
    
    return results


def write_results_to_csv(results):
    """
    将全局最优跟踪误差结果写入 CSV 文件
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
            "Train Track Error",
            "Test Track Error"
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
                    level_result.get('train_track_error', 'N/A'),
                    level_result.get('test_track_error', 'N/A')
                ]
                writer.writerow(row)
            
            # 添加每个 case 内部所有 level 的平均行
            avg_row = [
                case_name,
                'AVG',
                'N/A',
                case_result.get('avg_train_error', 'N/A'),
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
                    train_errors.append(level_result.get('train_track_error', 0))
                    test_errors.append(level_result.get('test_track_error', 0))
                    num_nodes_list.append(level_result.get('num_nodes', 0))
            
            avg_train_error = np.mean(train_errors) if train_errors else 0
            avg_test_error = np.mean(test_errors) if test_errors else 0
            avg_num_nodes = np.mean(num_nodes_list) if num_nodes_list else 0
            
            row = [
                "Average",
                level_idx,
                f"{avg_num_nodes:.2f}",
                f"{avg_train_error:.6f}",
                f"{avg_test_error:.6f}"
            ]
            writer.writerow(row)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    print(f"Starting global best Track evaluation for {method_case_name}...")
    print(f"Prediction directory: {prediction_path}")
    print(f"Base path: {base_path}")
    
    results = evaluate_global_track()
    write_results_to_csv(results)
    
    print("\nGlobal track evaluation completed!")