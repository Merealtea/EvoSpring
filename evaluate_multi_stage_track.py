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
output_file = "results/final_track.csv"

# 多阶段评估配置
NUM_STAGES = 2  # 假设有 5 个下采样阶段


def evaluate_prediction(start_frame, end_frame, vertices, gt_track_3d, idx, mask):
    """
    评估预测轨迹与真值轨迹之间的误差
    
    Args:
        start_frame: 起始帧
        end_frame: 结束帧
        vertices: 预测轨迹 [T, N, 3]
        gt_track_3d: 真值 3D 轨迹 [T, M, 3]
        idx: GT 点对应的预测点索引（通过 KDTree 匹配得到）
        mask: 有效 GT 点掩码
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


def load_stage_data(dir_name, stage_idx):
    """
    从指定阶段加载轨迹数据
    返回：vertices (所有 level 的列表)
    
    注意：不需要加载 gt_indices，因为 Track Error 评估使用 KDTree 自动匹配
    """
    traj_path = f"{dir_name}/trajectories/stage_{stage_idx}_best_trajectory.pkl"
    
    if not os.path.exists(traj_path):
        print(f"Warning: Trajectory file not found: {traj_path}")
        return None
    
    with open(traj_path, "rb") as f:
        data = pickle.load(f)
    
    # vertices 现在是一个列表，包含所有 level 的轨迹
    vertices = data.get('vertices', None)  # List of [T, N_level, 3] for each level
    
    return vertices


def evaluate_multi_stage():
    """
    多阶段 Track 评估主函数
    
    【核心逻辑】使用 KDTree 最近邻匹配，不需要对 GT 进行下采样
    - KDTree 会自动为每个 GT 点找到最近的预测点
    - 无论预测点云密度如何，都能正确计算 Track Error
    """
    results = []
    
    dir_names = glob.glob(f"{prediction_path}/*")
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
        
        # 加载真值轨迹
        with open(f"{base_path}/{case_name}/gt_track_3d.pkl", "rb") as f:
            gt_track_3d = pickle.load(f)
        
        # 定位跟踪点索引（基于第 0 帧）
        mask = ~np.isnan(gt_track_3d[0]).any(axis=1)
        
        case_results = {
            'case_name': case_name,
            'stages': []
        }
        
        # 遍历所有阶段
        for stage_idx in range(NUM_STAGES):
            print(f"\n--- Evaluating Stage {stage_idx} ---")
            
            # 加载当前阶段的预测轨迹
            vertices = load_stage_data(dir_name, stage_idx)
            
            if vertices is None:
                print(f"Skipping stage {stage_idx} due to missing trajectory data")
                case_results['stages'].append({
                    'stage': stage_idx,
                    'train_track_error': None,
                    'test_track_error': None
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
                    
                    # 【关键】使用 KDTree 匹配 GT 点到预测点
                    # 不需要对 GT 进行下采样，KDTree 会自动找到最近的预测点
                    kdtree = KDTree(level_vertices[0])
                    _, gt_indices = kdtree.query(gt_track_3d[0][mask])
                    
                    # 评估 Track Error
                    train_track_error = evaluate_prediction(
                        1, train_frame, level_vertices, gt_track_3d, gt_indices, mask
                    )
                    test_track_error = evaluate_prediction(
                        train_frame, test_frame, level_vertices, gt_track_3d, gt_indices, mask
                    )
                    
                    level_results.append({
                        'level': level_idx,
                        'train_track_error': train_track_error,
                        'test_track_error': test_track_error
                    })
                    
                    print(f"    Level {level_idx}: Train Track Error: {train_track_error:.6f}, Test Track Error: {test_track_error:.6f}")
                
                # 计算所有 level 的平均误差
                avg_train_track = np.mean([r['train_track_error'] for r in level_results])
                avg_test_track = np.mean([r['test_track_error'] for r in level_results])
                
                print(f"Stage {stage_idx}:")
                print(f"  Average Train Track Error: {avg_train_track:.6f}")
                print(f"  Average Test Track Error: {avg_test_track:.6f}")
                
                # 保存每个 level 的结果
                case_results['stages'].append({
                    'stage': stage_idx,
                    'num_levels': num_levels,
                    'level_results': level_results,
                    'train_track_error': avg_train_track,
                    'test_track_error': avg_test_track
                })
            else:
                # 向后兼容：如果 vertices 不是列表，按原方式处理
                # 使用 KDTree 匹配 GT 点到预测点
                kdtree = KDTree(vertices[0])
                _, gt_indices = kdtree.query(gt_track_3d[0][mask])
                
                # 评估 Track Error
                train_track_error = evaluate_prediction(
                    1, train_frame, vertices, gt_track_3d, gt_indices, mask
                )
                test_track_error = evaluate_prediction(
                    train_frame, test_frame, vertices, gt_track_3d, gt_indices, mask
                )
                
                print(f"Stage {stage_idx}:")
                print(f"  Train Track Error: {train_track_error:.6f}")
                print(f"  Test Track Error: {test_track_error:.6f}")
                
                case_results['stages'].append({
                    'stage': stage_idx,
                    'train_track_error': train_track_error,
                    'test_track_error': test_track_error
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
            "Train Track Error",
            "Test Track Error"
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
                            level_result.get('train_track_error', 'N/A'),
                            level_result.get('test_track_error', 'N/A')
                        ]
                        writer.writerow(row)
                else:
                    # 向后兼容：没有 level 区分的旧格式
                    row = [
                        case_name,
                        stage_idx,
                        0,  # 默认 level 为 0
                        stage_result.get('train_track_error', 'N/A'),
                        stage_result.get('test_track_error', 'N/A')
                    ]
                    writer.writerow(row)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    print(f"Starting multi-stage evaluation for {method_case_name}...")
    print(f"Prediction path: {prediction_path}")
    print(f"Base path: {base_path}")
    print(f"Number of stages: {NUM_STAGES}")
    
    results = evaluate_multi_stage()
    write_results_to_csv(results)
    
    print("\nEvaluation completed!")
