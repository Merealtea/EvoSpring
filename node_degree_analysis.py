import pickle
import glob
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from tqdm import tqdm
import torch
import json
import warp as wp
from matplotlib import cm
from matplotlib.colors import Normalize

# 从 End2End_Reduction_Dataset 导入相关类
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from End2End_Reduction_Dataset import End2EndReductionDataset
from qqtt.utils import cfg

# 初始化 warp
wp.init()
wp.set_device("cuda:0")

prediction_path = "/mnt/pool1/cxy/phystwin-v2/real2sim-eval/log/Real2sim_Reduction"
base_path = "./data/different_types/"
output_file = "results/node_degree_analysis.csv"
visualization_output_dir = "./visualization/node_degree/"

# Node type constants
NODE_TYPE_OBJECT = 0      # 物体点
NODE_TYPE_SURFACE = 1     # 表面点
NODE_TYPE_INTERIOR = 2    # 内部点
NODE_TYPE_CONTROLLER = 3  # 控制点


def visualize_prediction_vs_gt(pred_vertices, gt_vertices, edges, output_dir, 
                                level_idx, node_types=None, num_original_points=None, fps=5):
    """
    可视化预测结果和真值的对比（只绘制 object point）
    
    Args:
        pred_vertices: 预测的顶点轨迹 [T, N, 3]
        gt_vertices: 真值顶点轨迹 [T, N_gt, 3]
        edges: 边连接 [2, num_edges]
        output_dir: 输出目录
        level_idx: 层级索引
        node_types: 节点类型 [N]
        num_original_points: object point 的数量（只绘制这些点）
        fps: 帧率
    """
    os.makedirs(output_dir, exist_ok=True)
    
    T = pred_vertices.shape[0]
    
    # 创建临时目录保存帧
    temp_dir = os.path.join(output_dir, f'temp_comparison_level_{level_idx}')
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Creating comparison video for Level {level_idx} with {T} frames...")
    
    # 如果没有指定 num_original_points，使用 gt_vertices 的数量
    if num_original_points is None:
        num_original_points = gt_vertices.shape[1]
    
    # 只取 object point 部分
    pred_object_points = pred_vertices[:, :num_original_points, :]  # [T, N_obj, 3]
    
    # 计算误差（只计算 object point）
    errors = np.linalg.norm(pred_object_points - gt_vertices, axis=-1)  # [T, N_obj]
    max_error = np.max(errors)
    
    # 预先计算统一的坐标轴范围（使用所有帧的所有顶点）
    all_gt_verts = gt_vertices.reshape(-1, 3)  # [T*N, 3]
    all_pred_verts = pred_object_points.reshape(-1, 3)  # [T*N, 3]
    all_verts_combined = np.concatenate([all_gt_verts, all_pred_verts], axis=0)
    
    margin = 0.15  # 增加 margin 确保所有点都可见
    x_min = all_verts_combined[:, 0].min() - margin
    x_max = all_verts_combined[:, 0].max() + margin
    y_min = all_verts_combined[:, 1].min() - margin
    y_max = all_verts_combined[:, 1].max() + margin
    z_min = all_verts_combined[:, 2].min() - margin
    z_max = all_verts_combined[:, 2].max() + margin
    
    print(f"Unified axis limits: X[{x_min:.3f}, {x_max:.3f}], Y[{y_min:.3f}, {y_max:.3f}], Z[{z_min:.3f}, {z_max:.3f}]")
    
    # 渲染每一帧
    for frame_idx in tqdm(range(T), desc=f"Rendering Level {level_idx}"):
        fig = plt.figure(figsize=(16, 6))
        
        # GT 子图
        ax1 = fig.add_subplot(121, projection='3d')
        gt_verts = gt_vertices[frame_idx]
        ax1.scatter(gt_verts[:, 0], gt_verts[:, 1], gt_verts[:, 2], c='green', s=30, alpha=0.8, label='GT')
        
        ax1.set_title(f'Ground Truth (Frame {frame_idx})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Prediction 子图
        ax2 = fig.add_subplot(122, projection='3d')
        pred_verts = pred_object_points[frame_idx]
        
        # 根据误差着色预测点
        frame_errors = errors[frame_idx]
        norm = Normalize(vmin=0, vmax=max_error)
        colors = cm.RdYlGn_r(norm(frame_errors))
        ax2.scatter(pred_verts[:, 0], pred_verts[:, 1], pred_verts[:, 2], 
                   c=colors, s=30, alpha=0.8, label='Pred')
        
        ax2.set_title(f'Prediction (Frame {frame_idx})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # 使用统一的坐标轴范围
        ax1.set_xlim(x_min, x_max)
        ax2.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
        ax1.set_zlim(z_min, z_max)
        ax2.set_zlim(z_min, z_max)
        
        # 保存帧
        frame_path = os.path.join(temp_dir, f'frame_{frame_idx:05d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    # 将图像转换为视频
    video_path = os.path.join(output_dir, f'level_{level_idx}_comparison.mp4')
    
    first_frame_path = os.path.join(temp_dir, 'frame_00000.png')
    if os.path.exists(first_frame_path):
        first_frame = cv2.imread(first_frame_path)
        height, width = first_frame.shape[:2]
        
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        for frame_idx in tqdm(range(T), desc="Creating video"):
            frame_path = os.path.join(temp_dir, f'frame_{frame_idx:05d}.png')
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                if frame.shape != first_frame.shape:
                    import pdb; pdb.set_trace()
                writer.write(frame)
        
        writer.release()
        print(f"Comparison video saved: {video_path}")
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)
    
    return video_path


def visualize_trajectory_overlay(pred_vertices, gt_vertices, output_dir, level_idx, 
                                  frame_indices=None, save_individual_frames=True):
    """
    可视化预测和真值的叠加对比（在同一 3D 空间中）
    
    Args:
        pred_vertices: 预测的顶点轨迹 [T, N, 3]
        gt_vertices: 真值顶点轨迹 [T, N_gt, 3]
        output_dir: 输出目录
        level_idx: 层级索引
        frame_indices: 要可视化的帧索引列表（可选）
        save_individual_frames: 是否保存单个帧图像
    """
    os.makedirs(output_dir, exist_ok=True)
    
    T = pred_vertices.shape[0]
    if frame_indices is None:
        frame_indices = [0, T//4, T//2, 3*T//4, T-1]
    frame_indices = [i for i in frame_indices if i < T]
    
    print(f"Creating trajectory overlay visualization for Level {level_idx}...")
    
    for frame_idx in tqdm(frame_indices, desc="Creating overlay visualizations"):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        gt_verts = gt_vertices[frame_idx]
        pred_verts = pred_vertices[frame_idx]
        
        # 绘制真值点（绿色）
        ax.scatter(gt_verts[:, 0], gt_verts[:, 1], gt_verts[:, 2], 
                  c='green', s=30, alpha=0.6, label='Ground Truth')
        
        # 绘制预测点（蓝色）
        ax.scatter(pred_verts[:, 0], pred_verts[:, 1], pred_verts[:, 2], 
                  c='blue', s=30, alpha=0.6, label='Prediction')
        
        # 设置坐标轴范围
        all_verts = np.concatenate([gt_verts, pred_verts], axis=0)
        margin = 0.1
        x_min, x_max = all_verts[:, 0].min() - margin, all_verts[:, 0].max() + margin
        y_min, y_max = all_verts[:, 1].min() - margin, all_verts[:, 1].max() + margin
        z_min, z_max = all_verts[:, 2].min() - margin, all_verts[:, 2].max() + margin
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        ax.set_title(f'Trajectory Overlay - Level {level_idx}, Frame {frame_idx}', fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='best')
        
        if save_individual_frames:
            output_path = os.path.join(output_dir, f'level_{level_idx}_frame_{frame_idx:03d}_overlay.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved overlay image: {output_path}")
        
        plt.close(fig)


def compute_trajectory_errors(pred_vertices, gt_vertices):
    """
    计算轨迹误差统计
    
    Args:
        pred_vertices: 预测的顶点轨迹 [T, N, 3]
        gt_vertices: 真值顶点轨迹 [T, N, 3]
    
    Returns:
        error_stats: 误差统计字典
    """
    if pred_vertices.shape != gt_vertices.shape:
        print(f"Warning: Shape mismatch. Pred: {pred_vertices.shape}, GT: {gt_vertices.shape}")
        # 尝试只比较共同的部分
        min_T = min(pred_vertices.shape[0], gt_vertices.shape[0])
        min_N = min(pred_vertices.shape[1], gt_vertices.shape[1])
        pred_vertices = pred_vertices[:min_T, :min_N]
        gt_vertices = gt_vertices[:min_T, :min_N]
    
    # 计算每帧每点的误差
    errors = np.linalg.norm(pred_vertices - gt_vertices, axis=-1)  # [T, N]
    
    # 计算统计
    error_stats = {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'min_error': np.min(errors),
        'max_error': np.max(errors),
        'per_frame_mean': np.mean(errors, axis=1),  # 每帧的平均误差
        'per_point_mean': np.mean(errors, axis=0),  # 每点的平均误差
    }
    
    return error_stats


def run_analysis_with_warp_simulator(node_info_path, mech_info_path, output_dir, device="cuda:0"):
    """
    使用 warp simulator 运行分析并可视化
    
    Args:
        data_dir: 数据目录
        object_case: 对象案例名称
        mech_info_path: 力学参数路径
        output_dir: 输出目录
        device: 设备
    """
    print(f"{'='*60}")
    print(f"Running analysis with Warp Simulator")
    print(f"Node info path: {node_info_path}")
    print(f"Mech info path: {mech_info_path}")
    print(f"{'='*60}\n")
    
    # 创建 simulator runner
    runner = WarpSimulatorRunner(node_info_path, mech_info_path, device)

    # 运行仿真
    pred_vertices = runner.run_simulation()
    
    # 获取真值
    gt_vertices = runner.get_gt_vertices()
    
   
    
    return pred_vertices, gt_vertices

class WarpSimulatorRunner:
    """
    用于加载训练好的 warp 模型参数并运行 simulator 的类
    使用 End2EndReductionDataset 进行数据加载和 simulator 创建
    """
    def __init__(self, node_file_path=None, mech_info_path=None, device="cuda:0"):
        """
        初始化 simulator runner
        
        Args:
            data_dir: 数据目录路径
            object_case: 对象案例名称
            mech_info_path: 力学参数路径（可选，如果不提供则使用默认参数）
            device: 设备
        """
        self.node_file_path = node_file_path
        self.mech_info_path = mech_info_path
        self.device = device
        self.simulator = None
        self.dataset = None

        self._load_mech_info()
        self._create_dataset()
        self._create_simulator()
    
    def _load_mech_info(self):
        """加载力学参数"""
        self.mech_info = None  # 初始化 mech_info
        self.mech_info_list = None  # 多层级 mech_info 列表
        
        if self.mech_info_path is not None and os.path.exists(self.mech_info_path):
            print(f"Loading mechanical parameters from: {self.mech_info_path}")
            loaded_data = torch.load(self.mech_info_path, map_location=self.device)
            
            # 处理保存的数据格式：{'mech': [...]} 或直接是 mech_info 字典
            if isinstance(loaded_data, dict) and 'mech' in loaded_data:
                # 新格式：包含完整 stage 信息的列表
                self.mech_info_list = loaded_data['mech']
                print(f"Loaded {len(self.mech_info_list)} levels of mechanical info")
                # 使用第 0 层（最上层）的数据进行仿真
                self.mech_info = self.mech_info_list[0] if len(self.mech_info_list) > 0 else None
            else:
                # 直接是 mech_info 字典
                self.mech_info = loaded_data
                self.mech_info_list = [loaded_data]
                print(f"Loaded mechanical info (direct format)")
        else:
            print(f"Warning: mech_info_path is None or file does not exist: {self.mech_info_path}")

    def _create_dataset(self):
        """创建 End2EndReductionDataset 实例"""
        print("Creating End2EndReductionDataset...")
        
        # 从 mech_info 中提取 object_case
        # 假设 mech_info_path 类似于 "./res/End2End_Reduction/double_lift_zebra/..."
        if self.mech_info_path is not None:
            parts = self.mech_info_path.split('/')
            for i, part in enumerate(parts):
                if part == 'End2End_Reduction' and i + 1 < len(parts):
                    object_case = parts[i + 1]
                    break
            else:
                object_case = "double_lift_zebra"  # 默认值
        else:
            object_case = "double_lift_zebra"
        
        # 创建 args 对象（模拟 argparse.Namespace）
        class Args:
            def __init__(self):
                self.object_case = object_case
                self.multi_mesh_layer = 1
                self.recal_mesh = False
                self.consist_mesh = False
        
        args = Args()
        
        # 创建 dataset
        self.dataset = End2EndReductionDataset(
            root="./data/different_types/",
            layer_num=args.multi_mesh_layer,
            stride=1,
            mode='train',
            recal_mesh=args.recal_mesh,
            consist_mesh=args.consist_mesh,
            object_case=args.object_case,
            args=args,
            device=self.device
        )
        print(f"Dataset created for {object_case}")
    
    def _create_simulator(self):
        """使用 End2EndReductionDataset.create_spring_mass_sim 创建 simulator"""
        print("Creating SpringMassSystemWarp simulator using dataset.create_spring_mass_sim...")
    
        # 检查 mech_info 是否加载成功
        if self.mech_info is None:
            print("Error: mech_info not loaded, cannot create simulator")
            return
        
        # 从 mech_info 中提取 simulator 所需的参数
        self.vertices = self.mech_info['vertices']  # 下采样后的节点位置
        self.edges = self.mech_info['edges']  # 下采样后的连接关系
        self.node_type = self.mech_info['node_type']  # 下采样后的节点类型
        self.masses = self.mech_info['masses']  # 下采样后的节点质量
        self.gt_vertices = self.mech_info['gt_vertices']  # 下采样后的 GT vertices
        self.gt_visibility = self.mech_info['gt_visibility']  # 下采样后的 GT visibility
        self.gt_motions_valid = self.mech_info.get('gt_motions_valid', None)  # 下采样后的 GT motions valid
        self.node_ids = self.mech_info.get('node_ids', None)  # 节点 ID 映射
        
        # 计算 rest_lengths
        if isinstance(self.edges, torch.Tensor):
            spring_graph = self.edges.int().T.contiguous()
        else:
            spring_graph = torch.tensor(self.edges, dtype=torch.long, device=self.device).T.contiguous()
        
        rest_lengths = torch.norm(
            (self.vertices[spring_graph[:, 0]] - self.vertices[spring_graph[:, 1]]), 
            dim=1
        )
        
        # 使用 dataset.create_spring_mass_sim 创建 simulator
        self.simulator = self.dataset.create_spring_mass_sim(
            init_vertices=self.vertices,
            init_springs=spring_graph,
            init_rest_lengths=rest_lengths,
            init_masses=self.masses,
            node_type=self.node_type,
            gt_object_points=self.gt_vertices,
            gt_object_visibilities=self.gt_visibility,
            gt_object_motions_valid=self.gt_motions_valid,
        )
        
        # 如果加载了力学参数，设置到 simulator 中
        if self.mech_info is not None:
            self._set_mech_info_to_simulator()
    
    def _set_mech_info_to_simulator(self):
        """将力学参数设置到 simulator 中"""
        if 'log_spring_Y' in self.mech_info:
            wp_spring_Y = wp.from_torch(self.mech_info['log_spring_Y'].contiguous(), dtype=wp.float32, requires_grad=False)
            self.simulator.set_spring_Y(wp_spring_Y)
            print("Set spring_Y from loaded mech_info")
        
        if 'drag_damping' in self.mech_info:
            wp_drag = wp.from_torch(self.mech_info['drag_damping'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
            self.simulator.set_drag_damping(wp_drag)
            print("Set drag_damping from loaded mech_info")
        
        if 'dashpot_damping' in self.mech_info:
            wp_dash = wp.from_torch(self.mech_info['dashpot_damping'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
            self.simulator.set_dashpot_damping(wp_dash)
            print("Set dashpot_damping from loaded mech_info")
        
        if 'collision_elas' in self.mech_info:
            wp_elas = wp.from_torch(self.mech_info['collision_elas'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
            self.simulator.set_collision_elas(wp_elas)
        
        if 'collision_fric' in self.mech_info:
            wp_fric = wp.from_torch(self.mech_info['collision_fric'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
            self.simulator.set_collision_fric(wp_fric)
        
        if 'collision_object_elas' in self.mech_info:
            wp_obj_elas = wp.from_torch(self.mech_info['collision_object_elas'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
            self.simulator.set_collision_object_elas(wp_obj_elas)
        
        if 'collision_object_fric' in self.mech_info:
            wp_obj_fric = wp.from_torch(self.mech_info['collision_object_fric'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
            self.simulator.set_collision_object_fric(wp_obj_fric)
    
    def run_simulation(self, num_frames=None):
        """
        运行 simulator 进行前向传播
        
        Returns:
            pred_vertices: 预测的顶点轨迹 [T, N, 3]
        """
        if num_frames is None:
            num_frames = cfg.train_frame + cfg.test_frame
        
        print(f"Running simulation for {num_frames} frames...")
        
        # 初始化状态
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices,
            self.simulator.wp_init_velocities
        )
        
        pred_vertices = []
        
        for frame_idx in tqdm(range(num_frames), desc="Running simulation"):
            # 获取当前帧的预测结果
            x = wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False).cpu().numpy()
            pred_vertices.append(x.copy())
            
            # 设置控制器目标（如果是 real 数据）
            if cfg.data_type == "real":
                self.simulator.set_controller_target(frame_idx, pure_inference=False)
            
            # 更新碰撞图
            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()
            
            # 运行一步仿真
            if cfg.use_graph:
                wp.capture_launch(self.simulator.forward_graph)
            else:
                self.simulator.step()
            
            # 更新状态
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v
            )
        
        pred_vertices = np.stack(pred_vertices, axis=0)  # [T, N, 3]
        print(f"Simulation completed. Output shape: {pred_vertices.shape}")
        
        return pred_vertices
    
    def get_gt_vertices(self, num_frames=None):
        """
        获取真值顶点
        直接从加载的 mech_info 中获取已下采样的 GT 数据
        
        注意：mech_info 中的 gt_vertices 可能帧数不足，需要重复最后一帧
        """
        if num_frames is None:
            num_frames = cfg.train_frame + cfg.test_frame
        
        # 从加载的 mech_info 中获取 GT 数据（已经是下采样后的）
        if hasattr(self, 'gt_vertices') and self.gt_vertices is not None:
            # gt_vertices shape: [T, N, 3]
            if isinstance(self.gt_vertices, torch.Tensor):
                gt_vertices = self.gt_vertices.cpu().numpy()
            else:
                gt_vertices = self.gt_vertices
            
            # 如果 GT 数据帧数不足，重复最后一帧
            if gt_vertices.shape[0] < num_frames:
                last_frame = gt_vertices[-1:]
                num_repeat = num_frames - gt_vertices.shape[0]
                gt_vertices = np.concatenate([gt_vertices] + [last_frame] * num_repeat, axis=0)
            
            return gt_vertices[:num_frames]
        else:
            # 如果没有加载的 GT 数据，返回 None
            print("Warning: No GT vertices available in mech_info")
            return None
    
    def get_edges(self):
        """获取边连接"""
        # 从加载的 mech_info 中获取边
        if hasattr(self, 'edges') and self.edges is not None:
            return self.edges
        return None
    
    def get_node_types(self):
        """获取节点类型"""
        # 从加载的 mech_info 中获取节点类型
        if hasattr(self, 'node_type') and self.node_type is not None:
            if isinstance(self.node_type, torch.Tensor):
                return self.node_type
            else:
                return torch.tensor(self.node_type, dtype=torch.long, device=self.device)
        return None


def load_all_data_from_mech_info(dir_name):
    """
    完全从 mech_info 文件中加载所有可视化所需的数据
    不再依赖 global_best_trajectory.pkl
    
    mech_info 文件格式：
    - 直接是一个 dict，包含：
      - vertices: 节点位置 [N, 3]
      - edges: 边连接 [2, num_edges]
      - node_type: 节点类型 [N]
      - masses: 节点质量 [N]
      - gt_vertices: GT 顶点轨迹 [T, N, 3]
      - gt_visibility: GT 可见性 [T, N]
      - gt_motions_valid: GT 运动有效性 [T]
      - log_spring_Y: 弹簧杨氏模量（log）[num_edges//2]
      - drag_damping, dashpot_damping: 阻尼参数
      - collision_elas, collision_fric, collision_object_elas, collision_object_fric: 碰撞参数
    
    返回：dict 包含所有数据（单个 level，不是 list）
    """
    # 使用 stage1_epoch0_iter0_best_mech_info.pth
    mech_info_path = os.path.join(dir_name, 'spring_mech_info', 'global_best_mech_info.pth')
    
    if not os.path.exists(mech_info_path):
        print(f"Warning: mech_info file not found: {mech_info_path}")
        # 尝试备选路径
        alt_mech_info_path = os.path.join(dir_name,'spring_mech_info', 'global_best_mech_info.pth')
        if os.path.exists(alt_mech_info_path):
            mech_info_path = alt_mech_info_path
            print(f"Trying alternative: {mech_info_path}")
        else:
            return None
    
    print(f"Loading all data from mech_info: {mech_info_path}")
    mech_data = torch.load(mech_info_path, map_location='cpu')['mech'][0]
    import pdb; pdb.set_trace()
    # mech_data 直接就是 dict，不需要处理 'mech' 键
    if not isinstance(mech_data, dict):
        print(f"Error: Unexpected mech_data format: {type(mech_data)}, expected dict")
        return None
    
    # 提取所有数据（单个 level）
    all_data = {
        'vertices': None,      # [N, 3]
        'edges': None,         # [2, num_edges]
        'node_types': None,    # [N]
        'masses': None,        # [N]
        'gt_vertices': None,   # [T, N, 3]
        'gt_visibility': None, # [T, N]
        'gt_motions_valid': None, # [T]
        'spring_Y': None,      # spring Y
    }
    
    # 提取顶点
    if 'vertices' in mech_data:
        verts = mech_data['vertices']
        if isinstance(verts, torch.Tensor):
            all_data['vertices'] = verts.cpu().numpy()
        else:
            all_data['vertices'] = verts
        print(f"  Loaded vertices: {all_data['vertices'].shape}")

    # 提取边
    if 'edges' in mech_data:
        edges = mech_data['edges']
        if isinstance(edges, torch.Tensor):
            all_data['edges'] = edges.cpu().numpy()
        else:
            all_data['edges'] = edges
        print(f"  Loaded edges: {all_data['edges'].shape if hasattr(all_data['edges'], 'shape') else len(all_data['edges'])}")
    
    # 提取节点类型
    if 'node_type' in mech_data:
        nt = mech_data['node_type']
        if isinstance(nt, torch.Tensor):
            all_data['node_types'] = nt.cpu().numpy()
        else:
            all_data['node_types'] = nt
        print(f"  Loaded node_types: {all_data['node_types'].shape}")
    
    # 提取质量
    if 'masses' in mech_data:
        m = mech_data['masses']
        if isinstance(m, torch.Tensor):
            all_data['masses'] = m.cpu().numpy()
        else:
            all_data['masses'] = m
        print(f"  Loaded masses: {all_data['masses'].shape}")
    
    # 提取 GT 顶点轨迹
    if 'gt_vertices' in mech_data:
        gt_verts = mech_data['gt_vertices']
        if isinstance(gt_verts, torch.Tensor):
            all_data['gt_vertices'] = gt_verts.cpu().numpy()
        else:
            all_data['gt_vertices'] = gt_verts
        print(f"  Loaded gt_vertices: {all_data['gt_vertices'].shape if hasattr(all_data['gt_vertices'], 'shape') else 'N/A'}")
    
    # 提取 GT 可见性
    if 'gt_visibility' in mech_data:
        gt_vis = mech_data['gt_visibility']
        if isinstance(gt_vis, torch.Tensor):
            all_data['gt_visibility'] = gt_vis.cpu().numpy()
        else:
            all_data['gt_visibility'] = gt_vis
        print(f"  Loaded gt_visibility: {all_data['gt_visibility'].shape if hasattr(all_data['gt_visibility'], 'shape') else 'N/A'}")
    
    # 提取 GT 运动有效性
    if 'gt_motions_valid' in mech_data:
        gt_valid = mech_data['gt_motions_valid']
        if isinstance(gt_valid, torch.Tensor):
            all_data['gt_motions_valid'] = gt_valid.cpu().numpy()
        else:
            all_data['gt_motions_valid'] = gt_valid
        print(f"  Loaded gt_motions_valid: {all_data['gt_motions_valid'].shape if hasattr(all_data['gt_motions_valid'], 'shape') else 'N/A'}")
    
    # 提取弹簧杨氏模量
    if 'log_spring_Y' in mech_data:
        sy = mech_data['log_spring_Y']
        if isinstance(sy, torch.Tensor):
            all_data['spring_Y'] = sy.cpu().numpy()
        else:
            all_data['spring_Y'] = sy
        print(f"  Loaded log_spring_Y: {all_data['spring_Y'].shape if hasattr(all_data['spring_Y'], 'shape') else len(all_data['spring_Y'])}")
    
    print(f"Successfully loaded mech_info (single level)")
    return all_data


def visualize_node_connections(vertices, edges, node_types, output_dir, level_idx, frame_idx=0, spring_Y=None, view_angle='front'):
    """
    可视化节点连接关系（边），根据弹簧杨氏模量大小着色
    
    Args:
        vertices: 顶点位置 [N, 3]
        edges: 边连接 [2, num_edges] 或 [num_edges, 2]
        node_types: 节点类型 [N]
        output_dir: 输出目录
        level_idx: 层级索引
        frame_idx: 帧索引
        spring_Y: 弹簧杨氏模量 [num_edges] 或 [num_edges//2]，用于着色边
        view_angle: 视角角度，可选 'front', 'side', 'top', 'iso'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理顶点数据
    if isinstance(vertices, np.ndarray):
        verts = vertices
    else:
        verts = vertices.cpu().numpy() if hasattr(vertices, 'cpu') else np.array(vertices)
    
    # 处理边数据
    edge_array = edges.T 
    
    # edge_array 形状可能是 [2, num_edges] 或 [num_edges, 2]
    # 由于是有向图构建的边（每条边存储了正向和反向），无向图需要除以 2

    # 处理节点类型
    if node_types is not None:
        if isinstance(node_types, np.ndarray):
            types = node_types.flatten()
        else:
            types = node_types.cpu().numpy().flatten() if hasattr(node_types, 'cpu') else np.array(node_types).flatten()
    else:
        types = None
    
    # 处理弹簧杨氏模量
    original_spring_Y = None
    if spring_Y is not None:
        if isinstance(spring_Y, torch.Tensor):
            spring_Y = spring_Y.cpu().numpy()
        # spring_Y 可能是 log_spring_Y，需要取 exp
        if spring_Y.max() < 10:  # 可能是 log 值
            original_spring_Y = spring_Y.copy()  # 保存原始 log 值用于统计
            spring_Y = np.exp(spring_Y)
        

    # 定义视角 (elev, azim)
    view_angles = {
        'front': (0, 0),      # 前视图
        'side': (0, 90),      # 侧视图
        'top': (90, 0),       # 俯视图
        'iso': (20, 45),      # 等轴测视图
    }
    elev, azim = view_angles.get(view_angle, (20, 45))
    
    # 创建 3D 图形
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    
    # 定义节点类型颜色
    color_map = {
        NODE_TYPE_OBJECT: 'green',      # 物体点 - 绿色
        NODE_TYPE_SURFACE: 'blue',      # 表面点 - 蓝色
        NODE_TYPE_INTERIOR: 'orange',   # 内部点 - 橙色
        NODE_TYPE_CONTROLLER: 'red',    # 控制点 - 红色
    }

  
    # 绘制节点（缩小点的大小）
    if types is not None:
        for node_type, color in color_map.items():
            mask = (types == node_type)
            if np.sum(mask) > 0:
                ax.scatter(verts[mask, 0], verts[mask, 1], verts[mask, 2], 
                          c=color, s=5, alpha=0.8, label=f'Node Type {node_type}')
    else:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='green', s=5, alpha=0.8)
    
    # 绘制边（连接线），根据弹簧杨氏模量着色
    if spring_Y is not None and len(spring_Y) > 0:
        # 使用 colormap 来着色边
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
       
        norm = Normalize(vmin=np.min(spring_Y), vmax=np.max(spring_Y))
        cmap = plt.cm.viridis

        for idx, edge in enumerate(edge_array):
            if len(edge) >= 2:
                i, j = int(edge[0]), int(edge[1])
                if 0 <= i < len(verts) and 0 <= j < len(verts):
                    color = cmap(norm(spring_Y[idx]))
                    ax.plot(
                        [verts[i, 0], verts[j, 0]],
                        [verts[i, 1], verts[j, 1]],
                        [verts[i, 2], verts[j, 2]],
                        c=color, alpha=0.9, linewidth=3
                    )
        
        # 添加 colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.5)
        cbar.set_label('Spring Young\'s Modulus (Y)', fontsize=12)
    else:
        # 没有 spring_Y 时使用默认灰色
        for edge in edge_array:
            if len(edge) >= 2:
                i, j = int(edge[0]), int(edge[1])
                if 0 <= i < len(verts) and 0 <= j < len(verts):
                    ax.plot(
                        [verts[i, 0], verts[j, 0]],
                        [verts[i, 1], verts[j, 1]],
                        [verts[i, 2], verts[j, 2]],
                        c='gray', alpha=0.7, linewidth=2.5
                    )
    
    # 设置标题和标签
    view_title = {'front': 'Front', 'side': 'Side', 'top': 'Top', 'iso': 'Isometric'}
    view_name = view_title.get(view_angle, view_angle.capitalize())
    ax.set_title(f'Level {level_idx} - Node Connections (Frame {frame_idx}, {view_name} View)', fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if types is not None:
        ax.legend(loc='best')
    
    # 保存图像
    output_path = os.path.join(output_dir, f'level_{level_idx}_frame_{frame_idx:03d}_connections_{view_angle}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def visualize_node_masses(vertices, masses, node_types, output_dir, level_idx, frame_idx=0):
    """
    可视化节点质量分布（静态图）
    根据质量大小对节点进行着色
    
    Args:
        vertices: 顶点位置 [N, 3]
        masses: 节点质量 [N] 或标量
        node_types: 节点类型 [N]
        output_dir: 输出目录
        level_idx: 层级索引
        frame_idx: 帧索引
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理顶点数据
    if isinstance(vertices, np.ndarray):
        verts = vertices
    else:
        verts = vertices.cpu().numpy() if hasattr(vertices, 'cpu') else np.array(vertices)
    
    # 处理质量数据
    if isinstance(masses, np.ndarray):
        mass_array = masses.flatten()
    elif hasattr(masses, 'cpu'):
        mass_array = masses.cpu().numpy().flatten()
    else:
        # 如果 masses 是标量或列表
        mass_array = np.array(masses).flatten() if not np.isscalar(masses) else np.array([masses] * len(verts))
    
    # 处理节点类型
    if node_types is not None:
        if isinstance(node_types, np.ndarray):
            types = node_types.flatten()
        else:
            types = node_types.cpu().numpy().flatten() if hasattr(node_types, 'cpu') else np.array(node_types).flatten()
    else:
        types = None
    
    # 创建 3D 图形
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 根据质量着色节点
    norm = Normalize(vmin=np.min(mass_array), vmax=np.max(mass_array))
    cmap = cm.viridis
    
    # 绘制节点，根据质量着色
    colors = cmap(norm(mass_array))

    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], 
               c=colors, s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # 添加 colorbar
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.5)
    cbar.set_label('Node Mass', fontsize=12)
    
    # 设置标题和标签
    ax.set_title(f'Level {level_idx} - Node Mass Distribution (Frame {frame_idx})', fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置坐标轴范围
    margin = 0.1
    x_min, x_max = verts[:, 0].min() - margin, verts[:, 0].max() + margin
    y_min, y_max = verts[:, 1].min() - margin, verts[:, 1].max() + margin
    z_min, z_max = verts[:, 2].min() - margin, verts[:, 2].max() + margin
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # 保存图像
    output_path = os.path.join(output_dir, f'level_{level_idx}_frame_{frame_idx:03d}_mass.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 打印质量统计信息
    print(f"  Level {level_idx} Mass Statistics:")
    print(f"    Min: {np.min(mass_array):.6f}")
    print(f"    Max: {np.max(mass_array):.6f}")
    print(f"    Mean: {np.mean(mass_array):.6f}")
    print(f"    Std: {np.std(mass_array):.6f}")
    
    return output_path


def visualize_node_connections_video(vertices_seq, edges, node_types, output_dir, level_idx, fps=5):
    """
    创建节点连接动画视频
    
    Args:
        vertices_seq: 顶点序列 [T, N, 3]
        edges: 边连接 [2, num_edges] 或 [num_edges, 2]
        node_types: 节点类型 [N]
        output_dir: 输出目录
        level_idx: 层级索引
        fps: 帧率
    """
    os.makedirs(output_dir, exist_ok=True)
    
    temp_dir = os.path.join(output_dir, f'temp_level_{level_idx}')
    os.makedirs(temp_dir, exist_ok=True)
    
    T = vertices_seq.shape[0] if hasattr(vertices_seq, 'shape') else len(vertices_seq)
    
    print(f"Creating node connection video for Level {level_idx} with {T} frames...")

    
    visualize_node_connections(
        vertices=vertices_seq,
        edges=edges,
        node_types=node_types,
        output_dir=temp_dir,
        level_idx=level_idx,
        frame_idx=frame_idx
    )
    
    # 将图像转换为视频
    video_path = os.path.join(output_dir, f'level_{level_idx}_connections.mp4')
    
    # 获取第一帧的尺寸
    first_frame_path = os.path.join(temp_dir, 'level_{}_frame_000_connections.png'.format(level_idx))
    if os.path.exists(first_frame_path):
        first_frame = cv2.imread(first_frame_path)
        height, width = first_frame.shape[:2]
        
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        for frame_idx in tqdm(range(T), desc="Creating video"):
            frame_path = os.path.join(temp_dir, f'level_{level_idx}_frame_{frame_idx:03d}_connections.png')
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                writer.write(frame)
        
        writer.release()
        print(f"Video saved: {video_path}")
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)
    
    return video_path


def compute_node_degrees_with_types(edges_list, node_types_list):
    """
    计算每个 level 中每个节点的 degree（连接边数），并区分节点类型
    
    Args:
        edges_list: List of edge arrays, each edge array is [2, num_edges]
        node_types_list: List of node type arrays for each level
    
    Returns:
        degree_info: List of dicts with separated stats for controller vs non-controller nodes
    """
    degree_info = []

    for level_idx, edges in enumerate(edges_list):
        if edges is None:
            continue
        
        # edges shape: [2, num_edges]
        if isinstance(edges, np.ndarray):
            if edges.shape[0] == 2:
                src_nodes = edges[0]
                dst_nodes = edges[1]
            else:
                # Handle different edge format
                src_nodes = edges[:, 0]
                dst_nodes = edges[:, 1]
        else:
            # Convert to numpy if tensor
            edges_np = edges.cpu().numpy() if hasattr(edges, 'cpu') else np.array(edges)
            src_nodes = edges_np[0]
            dst_nodes = edges_np[1]
        
        # Count degree for each node
        all_nodes = np.concatenate([src_nodes, dst_nodes])
        unique_nodes, counts = np.unique(all_nodes, return_counts=True)
        
        # Create degree array for all nodes in this level
        node_type = node_types_list[level_idx]
        num_nodes = len(node_type)

        degrees = np.zeros_like(node_type)
        degrees[unique_nodes] = counts // 2

        # Get node types for this level
        has_node_types = False
        if node_types_list and level_idx < len(node_types_list):
            current_node_types = node_types_list[level_idx]
            if current_node_types is not None:
                if isinstance(current_node_types, np.ndarray):
                    node_types = current_node_types.flatten()
                elif hasattr(current_node_types, 'cpu'):
                    node_types = current_node_types.cpu().numpy().flatten()
                else:
                    node_types = np.array(current_node_types).flatten()
                has_node_types = True
            else:
                node_types = None
        else:
            node_types = None
        
        # Check if node_types is valid (not all zeros)
        if node_types is not None and len(node_types) > 0:
            unique_types = np.unique(node_types)
            if len(unique_types) == 1 and unique_types[0] == 0:
                # All zeros means no valid node type info
                has_node_types = False
                node_types = None
        
        # Separate degree statistics by node type
        if node_types is not None:
            # Controller nodes (type = 3)
            controller_mask = (node_types == NODE_TYPE_CONTROLLER)
            controller_indices = np.where(controller_mask)[0]

            controller_degrees = degrees[controller_indices] if len(controller_indices) > 0 else np.array([])
            
            # Non-controller nodes (type != 3)
            non_controller_mask = (node_types != NODE_TYPE_CONTROLLER)
            non_controller_indices = np.where(non_controller_mask)[0]
            non_controller_degrees = degrees[non_controller_indices] if len(non_controller_indices) > 0 else np.array([])
            
            # 计算每个非控制节点与控制节点之间的连接 degree
            # 遍历所有边，统计非控制节点连接到控制节点的边数
            non_controller_to_controller_degrees = np.zeros(num_nodes, dtype=int)
            for i in range(len(src_nodes)):
                src, dst = src_nodes[i], dst_nodes[i]
                if src < len(node_types) and dst < len(node_types):
                    # 如果 src 是非控制节点，dst 是控制节点
                    if node_types[src] != NODE_TYPE_CONTROLLER and node_types[dst] == NODE_TYPE_CONTROLLER:
                        non_controller_to_controller_degrees[src] += 1
                    # 如果 dst 是非控制节点，src 是控制节点
                    elif node_types[dst] != NODE_TYPE_CONTROLLER and node_types[src] == NODE_TYPE_CONTROLLER:
                        non_controller_to_controller_degrees[dst] += 1
            
            # 提取非控制节点的控制连接 degree
            non_controller_controller_degrees = non_controller_to_controller_degrees[non_controller_indices] if len(non_controller_indices) > 0 else np.array([])
            
            # Further break down non-controller by type
            object_degrees = degrees[node_types == NODE_TYPE_OBJECT] if len(node_types) > 0 else np.array([])
            surface_degrees = degrees[node_types == NODE_TYPE_SURFACE] if len(node_types) > 0 else np.array([])
            interior_degrees = degrees[node_types == NODE_TYPE_INTERIOR] if len(node_types) > 0 else np.array([])
        else:
            controller_degrees = np.array([])
            non_controller_degrees = degrees
            non_controller_controller_degrees = np.array([])
            object_degrees = np.array([])
            surface_degrees = np.array([])
            interior_degrees = np.array([])
        degree_info.append({
            'level_idx': level_idx,
            'num_nodes': num_nodes,
            'num_edges': len(src_nodes),
            'degrees': degrees,
            'non_controller_controller_degrees': non_controller_controller_degrees,  # 保存每个非控制节点到控制节点的连接 degree
            'all_mean_degree': np.mean(degrees) if len(degrees) > 0 else 0,
            'all_std_degree': np.std(degrees) if len(degrees) > 0 else 0,
            'all_min_degree': np.min(degrees) if len(degrees) > 0 else 0,
            'all_max_degree': np.max(degrees) if len(degrees) > 0 else 0,
            
            # Non-controller statistics
            'num_non_controller': len(non_controller_degrees),
            'non_controller_mean_degree': np.mean(non_controller_degrees) if len(non_controller_degrees) > 0 else 0,
            'non_controller_std_degree': np.std(non_controller_degrees) if len(non_controller_degrees) > 0 else 0,
            'non_controller_min_degree': np.min(non_controller_degrees) if len(non_controller_degrees) > 0 else 0,
            'non_controller_max_degree': np.max(non_controller_degrees) if len(non_controller_degrees) > 0 else 0,
            
            # 非控制节点与控制节点的连接统计
            'non_controller_to_controller_mean_degree': np.mean(non_controller_controller_degrees) if len(non_controller_controller_degrees) > 0 else 0,
            'non_controller_to_controller_std_degree': np.std(non_controller_controller_degrees) if len(non_controller_controller_degrees) > 0 else 0,
            'non_controller_to_controller_min_degree': np.min(non_controller_controller_degrees) if len(non_controller_controller_degrees) > 0 else 0,
            'non_controller_to_controller_max_degree': np.max(non_controller_controller_degrees) if len(non_controller_controller_degrees) > 0 else 0,
            'num_non_controller_with_controller_connection': np.sum(non_controller_controller_degrees > 0) if len(non_controller_controller_degrees) > 0 else 0,
            
            # Controller statistics
            'num_controller': len(controller_degrees),
            'controller_mean_degree': np.mean(controller_degrees) if len(controller_degrees) > 0 else 0,
            'controller_std_degree': np.std(controller_degrees) if len(controller_degrees) > 0 else 0,
            
            # By type statistics
            'object_mean_degree': np.mean(object_degrees) if len(object_degrees) > 0 else 0,
            'surface_mean_degree': np.mean(surface_degrees) if len(surface_degrees) > 0 else 0,
            'interior_mean_degree': np.mean(interior_degrees) if len(interior_degrees) > 0 else 0,
        })
        
        print(f"Level {level_idx}: {num_nodes} nodes, {len(src_nodes)} edges")
        print(f"  All nodes: mean={np.mean(degrees) if len(degrees) > 0 else 0:.2f}, "
              f"std={np.std(degrees) if len(degrees) > 0 else 0:.2f}, "
              f"min={np.min(degrees) if len(degrees) > 0 else 0}, "
              f"max={np.max(degrees) if len(degrees) > 0 else 0}")
        
        if has_node_types and node_types is not None:
            print(f"  Non-controller ({len(non_controller_degrees)} nodes): "
                  f"mean={np.mean(non_controller_degrees) if len(non_controller_degrees) > 0 else 0:.2f}, "
                  f"std={np.std(non_controller_degrees) if len(non_controller_degrees) > 0 else 0:.2f}, "
                  f"min={np.min(non_controller_degrees) if len(non_controller_degrees) > 0 else 0}, "
                  f"max={np.max(non_controller_degrees) if len(non_controller_degrees) > 0 else 0}")
            if len(controller_degrees) > 0:
                print(f"  Controller ({len(controller_degrees)} nodes): "
                      f"mean={np.mean(controller_degrees):.2f}, "
                      f"std={np.std(controller_degrees):.2f}")
            
            # 打印非控制节点与控制节点的连接统计
            if len(non_controller_controller_degrees) > 0:
                num_connected = np.sum(non_controller_controller_degrees > 0)
                print(f"  Non-controller to Controller connection:")
                print(f"    Nodes with controller connection: {num_connected} / {len(non_controller_controller_degrees)}")
                print(f"    Mean degree to controller: {np.mean(non_controller_controller_degrees):.2f}")
                print(f"    Std degree to controller: {np.std(non_controller_controller_degrees):.2f}")
                print(f"    Min degree to controller: {np.min(non_controller_controller_degrees)}")
                print(f"    Max degree to controller: {np.max(non_controller_controller_degrees)}")
            
            if len(object_degrees) > 0:
                print(f"  Object ({len(object_degrees)} nodes): mean={np.mean(object_degrees):.2f}")
            if len(surface_degrees) > 0:
                print(f"  Surface ({len(surface_degrees)} nodes): mean={np.mean(surface_degrees):.2f}")
            if len(interior_degrees) > 0:
                print(f"  Interior ({len(interior_degrees)} nodes): mean={np.mean(interior_degrees):.2f}")
        else:
            print(f"  [Note: No node type information available for this level]")
    
    return degree_info


def analyze_node_degree():
    """
    分析所有 case 的节点连接度，区分 controller 和 non-controller 节点
    完全从 mech_info 文件加载数据，不再依赖 global_best_trajectory.pkl
    
    mech_info 是 dict 格式（单个 level），不是 list
    """
    all_results = []
    
    dir_names = glob.glob(f"{prediction_path}/*")
    for dir_name in dir_names:
        
        if 'rope_0001' not in dir_name:
            continue
        
        case_name = dir_name.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"Analyzing {case_name}")
        print(f"{'='*60}\n")
        
        # Load all data from mech_info (returns dict for single level)
        all_data = load_all_data_from_mech_info(dir_name)
        import pdb; pdb.set_trace()
        if all_data is None or all_data.get('edges') is None:
            print(f"Skipping {case_name} due to missing edge data")
            continue
        
        # Compute node degrees with type separation (wrap in list for the function)
        degree_info = compute_node_degrees_with_types([all_data['edges']], [all_data['node_types']])
        
        # Store results - use gt_vertices for trajectory visualization
        case_results = {
            'case_name': case_name,
            'levels': degree_info,
            'vertices': [all_data['vertices']],  # Wrap in list: [[T, N, 3]]
            'edges': [all_data['edges']],           # Wrap in list: [[2, num_edges]]
            'node_types': [all_data['node_types']], # Wrap in list: [[N]]
            'masses': [all_data['masses']],         # Wrap in list: [[N]]
            'spring_Y_list': [all_data['spring_Y']] if all_data['spring_Y'] is not None else None,
            'gt_vertices': all_data['gt_vertices'],
        }
        all_results.append(case_results)
    
    return all_results


def visualize_all_cases(results, output_dir=None, spring_Y_list=None, visualize_mass=False, frame_interval=10):
    """
    可视化所有 case 的节点连接和质点质量分布
    每 frame_interval 帧生成一次静态可视化
    
    Args:
        results: analyze_node_degree 返回的结果
        output_dir: 输出目录
        spring_Y_list: 每个 level 的弹簧杨氏模量列表 [num_edges] 或 [num_edges//2]
        visualize_mass: 是否可视化质点质量分布
        frame_interval: 每多少帧生成一次静态可视化（默认每 10 帧）
    """
    if output_dir is None:
        output_dir = visualization_output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Starting node connection visualization...")
    if visualize_mass:
        print("Also visualizing node mass distribution...")
    print(f"Generating static visualization every {frame_interval} frames...")
    print(f"{'='*60}\n")

    for case_result in results:
        case_name = case_result['case_name']
        vertices = case_result.get('vertices')  # List of [T, N, 3] for each level
        edges = case_result.get('edges')        # List of [2, num_edges] for each level
        node_types = case_result.get('node_types')  # List of [N] for each level
        masses = case_result.get('masses')      # List of [N] for each level
        spring_Y_list = case_result.get('spring_Y_list')  # List of spring_Y for each level
        
        if vertices is None or edges is None:
            print(f"Skipping {case_name}: missing vertices or edges")
            continue
        
        case_output_dir = os.path.join(output_dir, case_name)
        os.makedirs(case_output_dir, exist_ok=True)
        
        print(f"\nVisualizing {case_name}...")
        
        # 处理所有 level
        for level_idx in range(len(vertices)):
            level_verts_seq = vertices[level_idx]  # [T, N, 3] 轨迹数据

            level_edges = edges[level_idx] if level_idx < len(edges) else None
            level_node_types = node_types[level_idx] if node_types and level_idx < len(node_types) else None
            level_masses = masses[level_idx] if masses and level_idx < len(masses) else None
            
            if level_verts_seq is None or level_edges is None:
                print(f"  Level {level_idx}: skipping due to missing data")
                continue
            
            # 获取轨迹帧数
            if isinstance(level_verts_seq, np.ndarray):
                num_frames = level_verts_seq.shape[0]
            elif hasattr(level_verts_seq, '__len__'):
                num_frames = len(level_verts_seq)
            else:
                print(f"  Level {level_idx}: skipping due to invalid vertex sequence format")
                continue
            
            print(f"  Level {level_idx}: Processing {num_frames} frames, generating visualization every {frame_interval} frames...")
            
            # 获取弹簧杨氏模量（如果有）
            spring_Y = None
            if spring_Y_list is not None and level_idx < len(spring_Y_list):
                spring_Y = spring_Y_list[level_idx]
            
            # 定义 3 个视角：前视图、侧视图、俯视图
            view_angles = ['front', 'side', 'top']
            
            # 生成节点连接可视化（3 个不同视角）
            for view_angle in view_angles:
                conn_path = visualize_node_connections(
                    vertices=level_verts_seq,
                    edges=level_edges,
                    node_types=level_node_types,
                    output_dir=case_output_dir,
                    level_idx=level_idx,
                    frame_idx=0,
                    spring_Y=spring_Y,
                    view_angle=view_angle
                )
                print(f"    Frame {0} ({view_angle} view): Connection image saved to {conn_path}")
            
            # 生成质点质量可视化
            if visualize_mass and level_masses is not None:
                mass_path = visualize_node_masses(
                    vertices=level_verts_seq,
                    masses=level_masses,
                    node_types=level_node_types,
                    output_dir=case_output_dir,
                    level_idx=level_idx,
                    frame_idx=0
                )
                print(f"    Frame {0}: Mass image saved to {mass_path}")
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*60}\n")


def write_results_to_csv(results):
    """
    将节点连接度分析结果写入 CSV 文件
    包含所有节点类型和 non-controller 的详细统计
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 写入表头
        header = [
            "Case Name",
            "Level",
            "Num Nodes",
            "Num Edges",
            "All Mean Degree",
            "All Std Degree",
            "All Min Degree",
            "All Max Degree",
            "Num Non-Controller",
            "Non-Controller Mean Degree",
            "Non-Controller Std Degree",
            "Non-Controller Min Degree",
            "Non-Controller Max Degree",
            "Num Controller",
            "Controller Mean Degree",
            "Controller Std Degree",
            "Num Non-Controller with Controller Connection",
            "Non-Controller to Controller Mean Degree",
            "Non-Controller to Controller Std Degree",
            "Non-Controller to Controller Min Degree",
            "Non-Controller to Controller Max Degree",
            "Object Mean Degree",
            "Surface Mean Degree",
            "Interior Mean Degree"
        ]
        writer.writerow(header)
        
        # 写入数据 - 每个 level 一行汇总统计
        for case_result in results:
            case_name = case_result['case_name']
            for level_info in case_result['levels']:
                row = [
                    case_name,
                    level_info['level_idx'],
                    level_info['num_nodes'],
                    level_info['num_edges'],
                    f"{level_info['all_mean_degree']:.4f}",
                    f"{level_info['all_std_degree']:.4f}",
                    level_info['all_min_degree'],
                    level_info['all_max_degree'],
                    level_info['num_non_controller'],
                    f"{level_info['non_controller_mean_degree']:.4f}",
                    f"{level_info['non_controller_std_degree']:.4f}",
                    level_info['non_controller_min_degree'],
                    level_info['non_controller_max_degree'],
                    level_info['num_controller'],
                    f"{level_info['controller_mean_degree']:.4f}",
                    f"{level_info['controller_std_degree']:.4f}",
                    level_info['num_non_controller_with_controller_connection'],
                    f"{level_info['non_controller_to_controller_mean_degree']:.4f}",
                    f"{level_info['non_controller_to_controller_std_degree']:.4f}",
                    level_info['non_controller_to_controller_min_degree'],
                    level_info['non_controller_to_controller_max_degree'],
                    f"{level_info['object_mean_degree']:.4f}",
                    f"{level_info['surface_mean_degree']:.4f}",
                    f"{level_info['interior_mean_degree']:.4f}"
                ]
                writer.writerow(row)
    
    print(f"\nResults saved to {output_file}")


def write_detailed_results_to_csv(results):
    """
    将详细的节点连接度结果写入 CSV 文件
    每个 node 单独列出一行
    """
    detailed_output = output_file.replace('.csv', '_detailed.csv')
    os.makedirs(os.path.dirname(detailed_output), exist_ok=True)
    
    with open(detailed_output, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 写入表头
        header = [
            "Case Name",
            "Level",
            "Node Index",
            "Node Degree"
        ]
        writer.writerow(header)
        
        # 写入数据 - 每个 node 一行
        for case_result in results:
            case_name = case_result['case_name']
            for level_info in case_result['levels']:
                for node_idx, degree in enumerate(level_info['degrees']):
                    row = [
                        case_name,
                        level_info['level_idx'],
                        node_idx,
                        degree
                    ]
                    writer.writerow(row)
    
    print(f"Detailed results saved to {detailed_output}")


def find_common_nodes_across_all_levels(mech_info_list):
    """
    找到在所有层中都出现的点（从第 0 层的视角）
    
    由于 node_ids 是对上一层结果的下采样，所以需要在上一层的 node_ids 的结果上找到对应的 node_ids，
    逐层映射才能得到对应最上层（第 0 层）的 node_ids。
    
    Args:
        mech_info_list: 包含多个层级信息的列表，每个层级包含 'node_ids' 列表
                       node_ids[i] 表示第 i 层的节点在第 i-1 层的索引
    
    Returns:
        common_node_ids: 在所有层中都出现的节点 ID 列表（第 0 层的索引）
    """
    if not isinstance(mech_info_list, list) or len(mech_info_list) < 1:
        print("Warning: mech_info_list is not a list or is empty")
        return []
    
    current_level_node_ids = mech_info_list[0]['node_ids']
    if isinstance(current_level_node_ids, torch.Tensor):
        current_level_node_ids = current_level_node_ids.cpu().numpy().tolist()

    print(f"Level 0: {len(current_level_node_ids)} nodes")
    
    # 逐层映射回第 0 层
    # 对于每一层 L，我们需要找到 L 层的节点在 0 层的索引
    # 方法：从第 1 层开始，逐层向上映射
    all_levels_node_ids_in_level0 = [current_level_node_ids]  # 第 0 层的所有节点
    
    for level_idx in range(1, len(mech_info_list)):
        # 获取当前层在上一层的索引
        current_to_prev_ids = mech_info_list[level_idx]['node_ids']
        
        # 将当前层的索引映射回第 0 层
        # current_to_prev_ids[i] 表示当前层第 i 个节点对应上一层的节点索引
        # 我们需要找到上一层这些节点在第 0 层的索引
        current_level_ids_in_level0 = []
        for prev_level_idx in current_to_prev_ids:
            current_level_ids_in_level0.append(all_levels_node_ids_in_level0[-1][prev_level_idx])
        
        all_levels_node_ids_in_level0.append(current_level_ids_in_level0)
        print(f"Level {level_idx}: {len(current_to_prev_ids)} nodes, mapped to {len(current_level_ids_in_level0)} nodes in level 0")

    
    # 找到在所有层中都出现的点（求交集）
    common_ids = all_levels_node_ids_in_level0[-1]
    
    print(f"\nTotal common nodes across all levels (in level 0 indices): {len(common_ids)}")
    
    return common_ids


def select_scattered_nodes(common_node_ids, vertices, num_nodes=10):
    """
    从共同节点中选择空间上分散的节点
    
    使用贪心算法：每次选择距离已选节点最远的节点
    
    Args:
        common_node_ids: 共同节点 ID 列表（第 0 层的索引）
        vertices: 顶点位置 [N, 3]
        num_nodes: 要选择的节点数量（默认 10）
    
    Returns:
        selected_nodes: 选中的节点 ID 列表
    """
    if len(common_node_ids) <= num_nodes:
        return common_node_ids
    
    # 转换为 numpy 数组方便计算
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    
    # 获取共同节点的坐标
    common_coords = vertices[common_node_ids]
    
    # 初始化：随机选择第一个节点（选择中间位置的节点）
    mid_idx = len(common_node_ids) // 2
    selected_indices = [mid_idx]
    selected_coords = [common_coords[mid_idx]]
    
    # 贪心选择：每次选择距离已选节点最远的节点
    while len(selected_indices) < num_nodes:
        max_min_dist = -1
        best_idx = -1
        
        for i, node_id in enumerate(common_node_ids):
            if i in selected_indices:
                continue
            
            # 计算该节点到所有已选节点的最小距离
            coord = common_coords[i]
            min_dist = float('inf')
            for sel_coord in selected_coords:
                dist = np.linalg.norm(coord - sel_coord)
                min_dist = min(min_dist, dist)
            
            # 更新最佳选择
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = i
        
        if best_idx >= 0:
            selected_indices.append(best_idx)
            selected_coords.append(common_coords[best_idx])
        else:
            break
    
    # 返回选中的节点 ID
    selected_nodes = [common_node_ids[i] for i in selected_indices]
    print(f"Selected {len(selected_nodes)} scattered nodes from {len(common_node_ids)} common nodes")
    
    return selected_nodes


def visualize_selected_nodes_with_forces_per_frame(
    mech_info_list,
    selected_node_ids,
    pred_vertices_list,
    output_dir,
    frame_idx,
    global_norm=None,
    node_colors=None,
    axis_limits=None
):
    """
    可视化同一 frame 下不同 level 的选中节点的受力和 topo 对比
    
    Args:
        mech_info_list: 多层级 mech_info 列表
        selected_node_ids: 第 0 层选定的节点 ID 列表
        pred_vertices_list: 各层的预测顶点轨迹列表
        output_dir: 输出目录
        frame_idx: 帧索引
        global_norm: 全局归一化器（用于统一的力的大小着色）
        node_colors: 每个选中节点的固定颜色（用于跨层识别）
        axis_limits: 统一的坐标轴范围字典 {'x_min': ..., 'x_max': ..., 'y_min': ..., 'y_max': ..., 'z_min': ..., 'z_max': ...}
    
    Returns:
        global_norm: 全局归一化器
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_levels = len(mech_info_list)
    
    # 为每一层构建邻接表
    adjacency_list = []
    for level_idx in range(num_levels):
        edges = mech_info_list[level_idx].get('edges')
        if edges is None:
            adjacency_list.append({})
            continue
        
        if isinstance(edges, torch.Tensor):
            edge_array = edges.T.cpu().numpy()
        else:
            edge_array = np.array(edges).T
        
        num_nodes = len(mech_info_list[level_idx].get('vertices', []))
        adjacency = {i: [] for i in range(num_nodes)}
        for edge in edge_array:
            if len(edge) >= 2:
                i, j = int(edge[0]), int(edge[1])
                if i < num_nodes and j < num_nodes:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
        adjacency_list.append(adjacency)
    
    # 获取所有层的顶点用于统一坐标轴范围（如果没有提供）
    if axis_limits is None:
        all_verts = []
        for level_idx, vertices_seq in enumerate(pred_vertices_list):
            if vertices_seq is not None:
                all_verts.append(vertices_seq.reshape(-1, 3))
        all_verts = np.concatenate(all_verts, axis=0)
        margin = 0.15
        x_min, x_max = all_verts[:, 0].min() - margin, all_verts[:, 0].max() + margin
        y_min, y_max = all_verts[:, 1].min() - margin, all_verts[:, 1].max() + margin
        z_min, z_max = all_verts[:, 2].min() - margin, all_verts[:, 2].max() + margin
    else:
        x_min = axis_limits['x_min']
        x_max = axis_limits['x_max']
        y_min = axis_limits['y_min']
        y_max = axis_limits['y_max']
        z_min = axis_limits['z_min']
        z_max = axis_limits['z_max']
    
    # 计算每一层的力
    T = pred_vertices_list[0].shape[0]
    forces_list = []
    accelerations_list = []
    
    for level_idx in range(num_levels):
        vertices_seq = pred_vertices_list[level_idx]
        frame_verts = vertices_seq[frame_idx]
        
        if frame_idx > 0 and frame_idx < T - 1:
            prev_verts = vertices_seq[frame_idx - 1]
            next_verts = vertices_seq[frame_idx + 1]
            accelerations = (next_verts - prev_verts) / (2 * 0.01)
        elif frame_idx == 0 and T > 1:
            next_verts = vertices_seq[frame_idx + 1]
            accelerations = (next_verts - frame_verts) / 0.01
        elif frame_idx == T - 1 and T > 1:
            prev_verts = vertices_seq[frame_idx - 1]
            accelerations = (frame_verts - prev_verts) / 0.01
        else:
            accelerations = np.zeros((len(frame_verts), 3))
        
        forces = np.linalg.norm(accelerations, axis=1)
        forces_list.append(forces)
        accelerations_list.append(accelerations)
    
    # 计算全局力的范围用于统一着色
    if global_norm is None:
        all_forces = np.concatenate(forces_list, axis=0)
        max_force = np.max(all_forces) if np.max(all_forces) > 0 else 1.0
        global_norm = Normalize(vmin=0, vmax=max_force)
    
    # 创建图形：水平排列所有层
    fig = plt.figure(figsize=(6 * num_levels, 5))
    
    for level_idx in range(num_levels):
        ax = fig.add_subplot(1, num_levels, level_idx + 1, projection='3d')
        
        vertices_seq = pred_vertices_list[level_idx]
        frame_verts = vertices_seq[frame_idx]
        forces = forces_list[level_idx]
        accelerations = accelerations_list[level_idx]
        adjacency = adjacency_list[level_idx]
        
        # 映射节点 ID 到当前层，并保留原始索引用于颜色查找
        mapped_node_ids = map_node_ids_to_level(selected_node_ids, mech_info_list, level_idx)
        
        # 只绘制选中节点的 topo 连接和受力
        for orig_idx, node_id in enumerate(mapped_node_ids):
            if node_id is None or node_id >= len(frame_verts):
                continue
            
            node_pos = frame_verts[node_id]
            node_force = forces[node_id]
            
            # 获取该节点的邻居
            neighbors = adjacency.get(node_id, [])
            
            # 绘制该节点到邻居的连接（彩色粗线，根据力的大小着色）
            for neighbor_id in neighbors:
                if neighbor_id < len(frame_verts):
                    neighbor_pos = frame_verts[neighbor_id]
                    # 使用力的大小着色连接
                    edge_color = cm.RdYlBu_r(global_norm(node_force))
                    ax.plot(
                        [node_pos[0], neighbor_pos[0]],
                        [node_pos[1], neighbor_pos[1]],
                        [node_pos[2], neighbor_pos[2]],
                        c=edge_color, alpha=0.9, linewidth=2.5
                    )
            
            # 绘制节点本身 - 使用固定颜色（跨层识别同一个点）
            # orig_idx 是 selected_node_ids 中的原始索引，直接用于查找颜色
            if node_colors is not None and orig_idx < len(node_colors):
                node_color = node_colors[orig_idx]
            else:
                node_color = cm.RdYlBu_r(global_norm(node_force))
            
            # 为每个节点添加图例（只显示一次）
            label = f'Node {selected_node_ids[orig_idx]}' if level_idx == 0 else None
            
            ax.scatter(node_pos[0], node_pos[1], node_pos[2], 
                      c=[node_color], s=150, alpha=1.0, edgecolors='black', linewidth=1.5,
                      label=label)
            
            # 绘制力的方向箭头
            if frame_idx < T - 1:
                accel = accelerations[node_id]
                arrow_scale = 0.02
                ax.quiver(node_pos[0], node_pos[1], node_pos[2],
                         accel[0] * arrow_scale, accel[1] * arrow_scale, accel[2] * arrow_scale,
                         color='red', alpha=0.8, arrow_length_ratio=0.3, linewidth=2)
        
        # 设置坐标轴范围
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        ax.set_title(f'Level {level_idx}\n({len(mapped_node_ids)}/{len(selected_node_ids)} nodes)', fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if level_idx == 0:
            ax.legend(loc='best')
    
    # 添加总标题
    fig.suptitle(f'Selected Nodes with Forces - Frame {frame_idx}', fontsize=14, y=1.05)
    
    # 保存帧
    frame_path = os.path.join(output_dir, f'frame_{frame_idx:05d}_comparison.png')
    plt.savefig(frame_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return global_norm, frame_path


def visualize_selected_nodes_comparison_video(
    mech_info_list,
    selected_node_ids,
    pred_vertices_list,
    output_dir,
    frame_indices=None,
    fps=5,
    node_colors=None
):
    """
    创建同一 frame 下不同 level 的选中节点受力对比视频
    
    Args:
        mech_info_list: 多层级 mech_info 列表
        selected_node_ids: 第 0 层选定的节点 ID 列表
        pred_vertices_list: 各层的预测顶点轨迹列表
        output_dir: 输出目录
        frame_indices: 要可视化的帧索引列表
        fps: 帧率
        node_colors: 每个选中节点的固定颜色（用于跨层识别）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_levels = len(mech_info_list)
    T = pred_vertices_list[0].shape[0] if pred_vertices_list else 0
    
    if frame_indices is None:
        frame_indices = list(range(0, T, 10))
    frame_indices = [i for i in frame_indices if i < T]
    
    print(f"Creating per-frame comparison visualization for {num_levels} levels, {len(frame_indices)} frames...")
    
    # 创建临时目录保存帧
    temp_dir = os.path.join(output_dir, 'temp_per_frame_comparison')
    os.makedirs(temp_dir, exist_ok=True)
    
    # 计算所有层的统一坐标轴范围
    all_verts = []
    for level_idx, vertices_seq in enumerate(pred_vertices_list):
        if vertices_seq is not None:
            all_verts.append(vertices_seq.reshape(-1, 3))
    all_verts = np.concatenate(all_verts, axis=0)
    margin = 0.15
    axis_limits = {
        'x_min': all_verts[:, 0].min() - margin,
        'x_max': all_verts[:, 0].max() + margin,
        'y_min': all_verts[:, 1].min() - margin,
        'y_max': all_verts[:, 1].max() + margin,
        'z_min': all_verts[:, 2].min() - margin,
        'z_max': all_verts[:, 2].max() + margin
    }
    print(f"Unified 3D axis limits: X[{axis_limits['x_min']:.3f}, {axis_limits['x_max']:.3f}], "
          f"Y[{axis_limits['y_min']:.3f}, {axis_limits['y_max']:.3f}], "
          f"Z[{axis_limits['z_min']:.3f}, {axis_limits['z_max']:.3f}]")
    
    global_norm = None
    frame_paths = []
    
    # 渲染每一帧
    for frame_idx in tqdm(frame_indices, desc="Rendering per-frame comparison"):
        global_norm, frame_path = visualize_selected_nodes_with_forces_per_frame(
            mech_info_list=mech_info_list,
            selected_node_ids=selected_node_ids,
            pred_vertices_list=pred_vertices_list,
            output_dir=temp_dir,
            frame_idx=frame_idx,
            global_norm=global_norm,
            node_colors=node_colors,
            axis_limits=axis_limits
        )
        frame_paths.append(frame_path)
    
    # 将图像转换为视频
    video_path = os.path.join(output_dir, 'per_frame_comparison.mp4')
    
    if len(frame_paths) > 0 and os.path.exists(frame_paths[0]):
        first_frame = cv2.imread(frame_paths[0])
        height, width = first_frame.shape[:2]
        
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        for frame_path in tqdm(frame_paths, desc="Creating comparison video"):
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                writer.write(frame)
        
        writer.release()
        print(f"Per-frame comparison video saved: {video_path}")
        
        # # 清理临时文件
        # import shutil
        # shutil.rmtree(temp_dir)
    
    return video_path


def map_node_ids_to_level(selected_node_ids, mech_info_list, target_level):
    """
    将第 0 层的节点 ID 映射到目标层
    
    由于 node_ids 是下采样索引，需要找到第 0 层的节点在目标层的对应索引
    
    映射逻辑：
    - mech_info_list[level]['node_ids'] 是一个索引列表
    - 表示该层的节点在上一层的索引
    - 例如：level 1 的 node_ids = [0, 2, 5, ...] 表示 level 1 的第 0 个节点对应 level 0 的第 0 个节点，
      level 1 的第 1 个节点对应 level 0 的第 2 个节点，以此类推
    
    要从 level 0 映射到 level N，需要反向查找：
    - 对于 level 0 的节点 ID，找到它在 level 1 的索引（如果存在）
    - 然后找到它在 level 2 的索引，依此类推
    
    Args:
        selected_node_ids: 第 0 层的节点 ID 列表
        mech_info_list: 多层级 mech_info 列表
        target_level: 目标层级索引
    
    Returns:
        mapped_node_ids: 目标层中的节点 ID 列表（可能少于输入，因为有些节点可能被下采样掉了）
    """
    if target_level == 0:
        return selected_node_ids
    
    # 从第 0 层开始，逐层向下映射
    # current_ids_in_prev 表示当前层节点在上一层（最初是第 0 层）的索引
    current_ids_in_prev = np.array(mech_info_list[0]['node_ids'])
    
    for level_idx in range(1, target_level + 1):
        # 获取当前层在上一层的索引
        prev_to_current_ids = mech_info_list[level_idx]['node_ids']
        
        # 更新 current_ids_in_prev：现在表示当前层节点在第 0 层的索引
        current_ids_in_prev = current_ids_in_prev[prev_to_current_ids]
    
    # 现在 current_ids_in_prev 表示 target_level 层的节点在第 0 层的索引
    # 我们需要找到 selected_node_ids 中的节点在 target_level 层的索引
    mapped_ids = []
    for selected_id in selected_node_ids:
        # 在 current_ids_in_prev 中查找 selected_id
        idx_in_target = np.where(current_ids_in_prev == selected_id)[0]
        if len(idx_in_target) > 0:
            mapped_ids.append(int(idx_in_target[0]))
        else:
            mapped_ids.append(None)  # 该节点在目标层不存在
    
    return mapped_ids


def visualize_all_levels_comparison(
    mech_info_list,
    selected_node_ids,
    pred_vertices_list,
    output_dir,
    frame_indices=None,
    fps=5
):
    """
    可视化所有层的共同节点对比
    
    Args:
        mech_info_list: 多层级 mech_info 列表
        selected_node_ids: 第 0 层选定的节点 ID 列表
        pred_vertices_list: 各层的预测顶点轨迹列表
        output_dir: 输出目录
        frame_indices: 要可视化的帧索引列表
        fps: 帧率
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_levels = len(mech_info_list)
    T = pred_vertices_list[0].shape[0] if pred_vertices_list else 0
    
    if frame_indices is None:
        frame_indices = list(range(0, T, 10))
    frame_indices = [i for i in frame_indices if i < T]
    
    print(f"Creating comparison visualization for {num_levels} levels, {len(frame_indices)} frames...")
    
    # 创建临时目录保存帧
    temp_dir = os.path.join(output_dir, 'temp_comparison_all_levels')
    os.makedirs(temp_dir, exist_ok=True)
    
    # 获取所有层的统一坐标轴范围
    all_verts = []
    for level_idx, vertices_seq in enumerate(pred_vertices_list):
        if vertices_seq is not None:
            all_verts.append(vertices_seq.reshape(-1, 3))
    all_verts = np.concatenate(all_verts, axis=0)
    margin = 0.15
    x_min, x_max = all_verts[:, 0].min() - margin, all_verts[:, 0].max() + margin
    y_min, y_max = all_verts[:, 1].min() - margin, all_verts[:, 1].max() + margin
    z_min, z_max = all_verts[:, 2].min() - margin, all_verts[:, 2].max() + margin
    
    # 为每一层构建邻接表
    adjacency_list = []
    for level_idx in range(num_levels):
        edges = mech_info_list[level_idx].get('edges')
        if edges is None:
            adjacency_list.append({})
            continue
        
        if isinstance(edges, torch.Tensor):
            edge_array = edges.T.cpu().numpy()
        else:
            edge_array = np.array(edges).T
        
        num_nodes = len(mech_info_list[level_idx].get('vertices', []))
        adjacency = {i: [] for i in range(num_nodes)}
        for edge in edge_array:
            if len(edge) >= 2:
                i, j = int(edge[0]), int(edge[1])
                if i < num_nodes and j < num_nodes:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
        adjacency_list.append(adjacency)
    
    # 渲染每一帧
    for frame_idx in tqdm(frame_indices, desc="Rendering comparison"):
        fig = plt.figure(figsize=(20, 5 * num_levels))
        
        for level_idx in range(num_levels):
            ax = fig.add_subplot(num_levels, 1, level_idx + 1, projection='3d')
            
            # 获取当前层的顶点
            vertices_seq = pred_vertices_list[level_idx] if level_idx < len(pred_vertices_list) else None
            if vertices_seq is None:
                continue
            
            frame_verts = vertices_seq[frame_idx]
            
            # 计算力的大小
            if frame_idx > 0 and frame_idx < T - 1:
                prev_verts = vertices_seq[frame_idx - 1]
                next_verts = vertices_seq[frame_idx + 1]
                accelerations = (next_verts - prev_verts) / (2 * 0.01)
                forces = np.linalg.norm(accelerations, axis=1)
            elif frame_idx == 0 and T > 1:
                next_verts = vertices_seq[frame_idx + 1]
                accelerations = (next_verts - frame_verts) / 0.01
                forces = np.linalg.norm(accelerations, axis=1)
            elif frame_idx == T - 1 and T > 1:
                prev_verts = vertices_seq[frame_idx - 1]
                accelerations = (frame_verts - prev_verts) / 0.01
                forces = np.linalg.norm(accelerations, axis=1)
            else:
                forces = np.zeros(len(frame_verts))
                accelerations = np.zeros((len(frame_verts), 3))
            
            # 归一化力的大小
            max_force = np.max(forces) if np.max(forces) > 0 else 1.0
            norm = Normalize(vmin=0, vmax=max_force)
            
            # 映射节点 ID 到当前层
            mapped_node_ids = map_node_ids_to_level(selected_node_ids, mech_info_list, level_idx)
            
            # 只绘制选中节点的 topo 连接和受力
            for node_id in mapped_node_ids:
                if node_id >= len(frame_verts):
                    continue
                
                node_pos = frame_verts[node_id]
                node_force = forces[node_id]
                
                # 获取该节点的邻居
                adjacency = adjacency_list[level_idx]
                neighbors = adjacency.get(node_id, [])
                
                # 绘制该节点到邻居的连接
                for neighbor_id in neighbors:
                    if neighbor_id < len(frame_verts):
                        neighbor_pos = frame_verts[neighbor_id]
                        color = cm.RdYlBu_r(norm(node_force))
                        ax.plot(
                            [node_pos[0], neighbor_pos[0]],
                            [node_pos[1], neighbor_pos[1]],
                            [node_pos[2], neighbor_pos[2]],
                            c=color, alpha=0.9, linewidth=2.5
                        )
                
                # 绘制节点本身
                color = cm.RdYlBu_r(norm(node_force))
                ax.scatter(node_pos[0], node_pos[1], node_pos[2], 
                          c=[color], s=150, alpha=1.0, edgecolors='black', linewidth=1.5,
                          label=f'Node {node_id}' if node_id == mapped_node_ids[0] else None)
                
                # 绘制力的方向箭头
                if frame_idx < T - 1:
                    accel = accelerations[node_id]
                    arrow_scale = 0.02
                    ax.quiver(node_pos[0], node_pos[1], node_pos[2],
                             accel[0] * arrow_scale, accel[1] * arrow_scale, accel[2] * arrow_scale,
                             color='red', alpha=0.8, arrow_length_ratio=0.3, linewidth=2)
            
            # 设置坐标轴范围
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            
            ax.set_title(f'Level {level_idx} - Selected Nodes (mapped: {len(mapped_node_ids)}/{len(selected_node_ids)} nodes)', fontsize=12)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            if level_idx == 0:
                ax.legend(loc='best')
        
        # 添加总标题
        fig.suptitle(f'All Levels Comparison - Frame {frame_idx}', fontsize=16, y=1.02)
        
        # 保存帧
        frame_path = os.path.join(temp_dir, f'frame_{frame_idx:05d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    # 将图像转换为视频
    video_path = os.path.join(output_dir, 'all_levels_comparison.mp4')
    
    first_frame_path = os.path.join(temp_dir, 'frame_00000.png')
    if os.path.exists(first_frame_path):
        first_frame = cv2.imread(first_frame_path)
        height, width = first_frame.shape[:2]
        
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        for frame_idx in tqdm(frame_indices, desc="Creating comparison video"):
            frame_path = os.path.join(temp_dir, f'frame_{frame_idx:05d}.png')
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                writer.write(frame)
        
        writer.release()
        print(f"All levels comparison video saved: {video_path}")
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)
    
    return video_path


def run_simulation_for_level(mech_info, dataset, level_idx, num_frames=None):
    """
    为指定层级运行仿真
    
    Args:
        mech_info: 该层的力学参数（dict，包含 list 形式的多层级数据）
        dataset: 数据集实例
        level_idx: 层级索引
        num_frames: 仿真帧数
    
    Returns:
        pred_vertices: 预测的顶点轨迹 [T, N, 3]
    """
    if num_frames is None:
        num_frames = cfg.train_frame + cfg.test_frame
    
    # 提取参数 - 注意这些都是 list，存储着不同 level 的结果
    # mech_info 中的 vertices, edges 等都是 list，每个元素对应一个 level
    # 需要根据 level_idx 提取对应层的数据
    vertices = mech_info['vertices'][level_idx] if isinstance(mech_info['vertices'], list) else mech_info['vertices']
    edges = mech_info['edges'][level_idx] if isinstance(mech_info['edges'], list) else mech_info['edges']
    node_type = mech_info['node_type'][level_idx] if isinstance(mech_info['node_type'], list) else mech_info['node_type']
    masses = mech_info['masses'][level_idx] if isinstance(mech_info['masses'], list) else mech_info['masses']
    gt_vertices = mech_info['gt_vertices'][level_idx] if isinstance(mech_info['gt_vertices'], list) else mech_info['gt_vertices']
    gt_visibility = mech_info['gt_visibility'][level_idx] if isinstance(mech_info['gt_visibility'], list) else mech_info['gt_visibility']
    gt_motions_valid_list = mech_info.get('gt_motions_valid', None)
    gt_motions_valid = gt_motions_valid_list[level_idx] if (gt_motions_valid_list is not None and isinstance(gt_motions_valid_list, list)) else gt_motions_valid_list
    
    # 计算 rest_lengths
    if isinstance(edges, torch.Tensor):
        spring_graph = edges.int().T.contiguous()
    else:
        spring_graph = torch.tensor(edges, dtype=torch.long, device=cfg.device).T.contiguous()
    
    rest_lengths = torch.norm(
        (vertices[spring_graph[:, 0]] - vertices[spring_graph[:, 1]]), 
        dim=1
    )
    
    # 创建 simulator
   
    simulator = dataset.create_spring_mass_sim(
        init_vertices=vertices.cuda(),
        init_springs=spring_graph.cuda(),
        init_rest_lengths=rest_lengths.cuda(),
        init_masses=masses.cuda(),
        node_type=node_type.cuda(),
        gt_object_points=gt_vertices.cuda(),
        gt_object_visibilities=gt_visibility.cuda(),
        gt_object_motions_valid=gt_motions_valid.cuda(),
    )
    
    # 设置力学参数 - 根据 level_idx 提取对应层的数据
    if 'log_spring_Y' in mech_info:
        log_spring_Y = mech_info['log_spring_Y'][level_idx] if isinstance(mech_info['log_spring_Y'], list) else mech_info['log_spring_Y']
        wp_spring_Y = wp.from_torch(log_spring_Y.cuda().contiguous(), dtype=wp.float32, requires_grad=False)
        simulator.set_spring_Y(wp_spring_Y)
    
    if 'drag_damping' in mech_info:
        drag_damping = mech_info['drag_damping'][level_idx] if isinstance(mech_info['drag_damping'], list) else mech_info['drag_damping']
        wp_drag = wp.from_torch(drag_damping.cuda().reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
        simulator.set_drag_damping(wp_drag)
    
    if 'dashpot_damping' in mech_info:
        dashpot_damping = mech_info['dashpot_damping'][level_idx] if isinstance(mech_info['dashpot_damping'], list) else mech_info['dashpot_damping']
        wp_dash = wp.from_torch(dashpot_damping.cuda().reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
        simulator.set_dashpot_damping(wp_dash)
    
    if 'collision_elas' in mech_info:
        collision_elas = mech_info['collision_elas'][level_idx] if isinstance(mech_info['collision_elas'], list) else mech_info['collision_elas']
        wp_elas = wp.from_torch(collision_elas.cuda().reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
        simulator.set_collision_elas(wp_elas)
    
    if 'collision_fric' in mech_info:
        collision_fric = mech_info['collision_fric'][level_idx] if isinstance(mech_info['collision_fric'], list) else mech_info['collision_fric']
        wp_fric = wp.from_torch(collision_fric.cuda().reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
        simulator.set_collision_fric(wp_fric)
    
    if 'collision_object_elas' in mech_info:
        collision_object_elas = mech_info['collision_object_elas'][level_idx] if isinstance(mech_info['collision_object_elas'], list) else mech_info['collision_object_elas']
        wp_obj_elas = wp.from_torch(collision_object_elas.cuda().reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
        simulator.set_collision_object_elas(wp_obj_elas)
    
    if 'collision_object_fric' in mech_info:
        collision_object_fric = mech_info['collision_object_fric'][level_idx] if isinstance(mech_info['collision_object_fric'], list) else mech_info['collision_object_fric']
        wp_obj_fric = wp.from_torch(collision_object_fric.cuda().reshape(1).contiguous(), dtype=wp.float32, requires_grad=False)
        simulator.set_collision_object_fric(wp_obj_fric)
    
    # 运行仿真
    print(f"Running simulation for level {level_idx} with {vertices.shape[0]} nodes, {num_frames} frames...")
    
    # 初始化状态
    simulator.set_init_state(
        simulator.wp_init_vertices,
        simulator.wp_init_velocities
    )
    
    pred_vertices = []
    
    for frame_idx in tqdm(range(num_frames), desc=f"Running simulation level {level_idx}"):
        # 获取当前帧的预测结果
        x = wp.to_torch(simulator.wp_states[0].wp_x, requires_grad=False).cpu().numpy()
        pred_vertices.append(x.copy())
        
        # 设置控制器目标（如果是 real 数据）
        if cfg.data_type == "real":
            simulator.set_controller_target(frame_idx, pure_inference=False)
        
        # 更新碰撞图
        if simulator.object_collision_flag:
            simulator.update_collision_graph()
        
        # 运行一步仿真
        if cfg.use_graph:
            wp.capture_launch(simulator.forward_graph)
        else:
            simulator.step()
        
        # 更新状态
        simulator.set_init_state(
            simulator.wp_states[-1].wp_x,
            simulator.wp_states[-1].wp_v
        )
    
    pred_vertices = np.stack(pred_vertices, axis=0)  # [T, N, 3]
    print(f"Simulation completed for level {level_idx}. Output shape: {pred_vertices.shape}")
    
    return pred_vertices


def run_warp_simulator_analysis():
    """
    运行 warp simulator 分析并可视化预测结果和真值
    包含新的可视化：找到在所有层中都出现的点，可视化其连接关系和受力
    
    修改：
    1. 为每一层重新构建仿真器获取轨迹
    2. 对于不同层级来自于同一个点的点使用相同的颜色进行绘制
    """
    # 配置参数
    node_info_path = ""
    
    # 力学参数路径（训练好的模型参数）
    mech_info_path = "/mnt/pool1/cxy/phystwin-v2/real2sim-eval/log/Real2sim_Reduction/sloth_0001/E2E/spring_mech_info/global_best_mech_info.pth"
    
    # 输出目录
    output_dir = "./visualization/warp_simulator_results/"
    
    # ============================================================
    # 新增：加载多层级 mech_info 并可视化共同节点的连接和受力
    # ============================================================
    print("\n" + "="*60)
    print("Loading multi-level mech_info for common node analysis...")
    print("="*60)
    
    # 加载包含多层级信息的 mech_info
    # global_best_mech_info.pth 包含 {'mech': [level0_info, level1_info, ...]}
    if os.path.exists(mech_info_path):
        loaded_data = torch.load(mech_info_path, map_location='cpu')
        
        # 处理保存的数据格式：{'mech': [...]} 或直接是 mech_info 字典
        if isinstance(loaded_data, dict) and 'mech' in loaded_data:
            mech_info_list = loaded_data['mech']
            print(f"Loaded {len(mech_info_list)} levels of mech_info")
        else:
            # 直接是 mech_info 字典（单层）
            mech_info_list = [loaded_data]
            print(f"Loaded single level mech_info")
        
        # 检查是否有 node_ids 信息
        has_node_ids = all('node_ids' in info and info['node_ids'] is not None 
                          for info in mech_info_list)
        
        if has_node_ids:
            # 找到在所有层中都出现的点
            common_node_ids = find_common_nodes_across_all_levels(mech_info_list)
            
            # 选择最多 10 个空间上分散的点
            if len(common_node_ids) > 0:
                # 使用第 0 层的顶点位置来选择分散的节点
                vertices = mech_info_list[0].get('vertices')
                if vertices is not None:
                    # vertices 可能是 list，取第一个元素
                    if isinstance(vertices, list):
                        vertices = vertices[0]
                    selected_nodes = select_scattered_nodes(common_node_ids, vertices, num_nodes=10)
                else:
                    selected_nodes = common_node_ids[:10]
                print(f"Selected {len(selected_nodes)} scattered common nodes for visualization: {selected_nodes}")
                
                # 为每一层重新构建仿真器并获取轨迹
                print("\nRunning simulation for each level...")
                pred_vertices_list = []
                
                # 创建数据集实例用于创建 simulator
                class Args:
                    def __init__(self):
                        self.object_case = "double_lift_zebra"
                        self.multi_mesh_layer = len(mech_info_list)
                        self.recal_mesh = False
                        self.consist_mesh = False
                
                args = Args()
                dataset = End2EndReductionDataset(
                    root="./data/different_types/",
                    layer_num=args.multi_mesh_layer,
                    stride=1,
                    mode='train',
                    recal_mesh=args.recal_mesh,
                    consist_mesh=args.consist_mesh,
                    object_case=args.object_case,
                    args=args,
                    device='cuda:0'
                )
                
                # 为每一层运行仿真
                for level_idx, mech_info in enumerate(mech_info_list):
                    print(f"\n--- Running simulation for Level {level_idx} ---")
                    level_pred_vertices = run_simulation_for_level(mech_info, dataset, level_idx)
                    pred_vertices_list.append(level_pred_vertices)
                
                # 为每个选中的节点分配固定颜色（用于跨层识别）
                node_colors = plt.cm.tab10(np.linspace(0, 1, len(selected_nodes)))
                print(f"\nNode colors (for cross-level identification):")
                for i, (node_id, color) in enumerate(zip(selected_nodes, node_colors)):
                    print(f"  Node {node_id}: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
                
                # 为每一层可视化
                selected_nodes_output_dir = os.path.join(output_dir, "selected_nodes_visualization")
                
                # 创建同一 frame 下不同 level 的对比可视化（使用固定颜色）
                visualize_selected_nodes_comparison_video(
                    mech_info_list=mech_info_list,
                    selected_node_ids=selected_nodes,
                    pred_vertices_list=pred_vertices_list,
                    output_dir=selected_nodes_output_dir,
                    frame_indices=list(range(0, pred_vertices_list[0].shape[0], 10)),
                    node_colors=node_colors
                )
            else:
                print("No common nodes found across all levels, skipping selected nodes visualization")
        else:
            print("No node_ids information available in mech_info, skipping common node analysis")
            print("Available keys in mech_info:", list(mech_info_list[0].keys()) if mech_info_list else "N/A")
    else:
        print(f"Mech info file not found: {mech_info_path}")
    
    print("\n" + "="*60)
    print("Warp Simulator analysis completed!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Node Degree Analysis and Warp Simulator Visualization")
    parser.add_argument("--mode", type=str, default="warp", choices=["warp", "node_degree"],
                        help="Analysis mode: 'warp' for warp simulator visualization, 'node_degree' for node degree analysis")
    parser.add_argument("--data_dir", type=str, default="./data/different_types/",
                        help="Data directory path")
    parser.add_argument("--object_case", type=str, default="double_lift_zebra",
                        help="Object case name")
    parser.add_argument("--mech_info_path", type=str, default=None,
                        help="Path to mechanical parameters file (.pth)")
    parser.add_argument("--output_dir", type=str, default="./visualization/warp_simulator_results/",
                        help="Output directory for visualization results")
    parser.add_argument("--visualize_mass", action="store_true",
                        help="Enable visualization of node mass distribution")
    
    args = parser.parse_args()
    
    if args.mode == "warp":
        # 运行 warp simulator 分析
        run_warp_simulator_analysis()
    else:
        # 运行节点度分析
        print(f"Starting node degree analysis for End2End_Reduction...")
        print(f"Prediction directory: {prediction_path}")
        print(f"Base path: {base_path}")
        
        results = analyze_node_degree()
        
        # 写入 CSV 结果
        write_results_to_csv(results)
        write_detailed_results_to_csv(results)
        
        # 创建节点连接可视化
        visualize_all_cases(results, visualize_mass=args.visualize_mass)
        
        print("\nNode degree analysis and visualization completed!")
