"""
Visualization utilities for End2End Reduction training.
This module provides functions to visualize ground truth and rendered results
for each level, saving them as MP4 videos.
"""

import torch
import numpy as np
import os
import cv2
from tqdm import tqdm
from qqtt.utils import logger, cfg
import warp as wp
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


class LevelVisualizer:
    """
    Visualizer for multi-level simulation results.
    Creates side-by-side comparison videos of ground truth vs rendered results.
    """
    
    def __init__(self, output_dir, fps=30, figsize=(16, 8)):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save output videos
            fps: Frames per second for output video
            figsize: Figure size for matplotlib (width, height)
        """
        self.output_dir = output_dir
        self.fps = fps
        self.figsize = figsize
        os.makedirs(output_dir, exist_ok=True)
        
    def visualize_level_comparison(self, 
                                   level_idx,
                                   gt_vertices, 
                                   pred_vertices,
                                   output_filename=None,
                                   title_prefix=""):
        """
        Visualize ground truth vs prediction for a single level (point cloud only).
        
        Args:
            level_idx: Level index
            gt_vertices: Ground truth vertices [T, N, 3]
            pred_vertices: Predicted vertices [T, N, 3]
            edges: Edge connectivity (ignored, for compatibility)
            output_filename: Output video filename (optional)
            title_prefix: Prefix for title
            
        Returns:
            Path to saved video
        """
        if output_filename is None:
            output_filename = f"level_{level_idx}_comparison.mp4"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        T = gt_vertices.shape[0]
        
        # Create figure with two subplots (GT and Pred)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, subplot_kw={'projection': '3d'})
        fig.suptitle(f'{title_prefix} Level {level_idx}: Ground Truth vs Prediction', fontsize=16)
        
        # Set same axis limits for both GT and Pred based on combined bounds
        self._set_axis_limits_same(ax1, ax2, gt_vertices, pred_vertices)
        
        # Create temp directory for frames
        temp_images_dir = os.path.join(self.output_dir, f'temp_level_{level_idx}')
        os.makedirs(temp_images_dir, exist_ok=True)
        
        # Render each frame - recreate plots each time to avoid artist removal issues
        for t in tqdm(range(T), desc=f"Rendering Level {level_idx}"):
            # Clear axes
            ax1.clear()
            ax2.clear()
            
            # Create new point clouds
            self._create_point_cloud(ax1, gt_vertices[t], 'Ground Truth', 'green')
            self._create_point_cloud(ax2, pred_vertices[t], 'Prediction', 'blue')
            
            # Set titles and axis limits (use same limits for both axes)
            ax1.set_title(f'Ground Truth (Frame {t})')
            ax2.set_title(f'Prediction (Frame {t})')
            self._set_axis_limits_same(ax1, ax2, gt_vertices, pred_vertices)
            
            # Save frame
            frame_path = os.path.join(temp_images_dir, f'frame_{t:05d}.png')
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        
        plt.close(fig)
        
        # Convert images to video
        self._images_to_video(temp_images_dir, output_path)
        
        # Cleanup temp files
        self._cleanup_temp_dir(temp_images_dir)
        
        logger.info(f"Level {level_idx} comparison video saved to: {output_path}")
        return output_path
    
    def _create_point_cloud(self, ax, vertices, label, color):
        """
        Create point cloud visualization (no edges, only points).
        
        Args:
            ax: Matplotlib axis
            vertices: Vertex positions [N, 3]
            label: Label for the point cloud
            color: Color for the points
            
        Returns:
            List of matplotlib artists
        """
        vertices = vertices.detach().cpu().numpy() if isinstance(vertices, torch.Tensor) else vertices
        
        artists = []
        
        # Check if vertices is valid
        if vertices is None or len(vertices) == 0:
            logger.warning("Vertices is None or empty in _create_point_cloud")
            return artists
        
        try:
            if ax.name == '3d':
                scatter = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                                    c=color, s=10, alpha=0.8, label=label)
            else:
                scatter = ax.scatter(vertices[:, 0], vertices[:, 1], 
                                    c=color, s=10, alpha=0.8, label=label)
            artists.append(scatter)
        except Exception as e:
            logger.warning(f"Failed to plot point cloud: {e}")
        
        return artists
    
    def _update_point_cloud(self, artists, vertices, ax, color):
        """
        Update point cloud with new vertex positions.
        
        Args:
            artists: List of artists to update
            vertices: New vertex positions [N, 3]
            ax: Matplotlib axis
            color: Color for the points
            
        Returns:
            List of new matplotlib artists
        """
        vertices = vertices.detach().cpu().numpy() if isinstance(vertices, torch.Tensor) else vertices
        
        # Remove old artists safely (check if artist still exists)
        for artist in artists[:]:  # Use slice copy to avoid modification during iteration
            try:
                artist.remove()
            except (ValueError, AttributeError) as e:
                # Artist already removed or invalid, skip
                logger.debug(f"Skipping artist removal: {e}")
        
        # Create new artists and return them
        return self._create_point_cloud(ax, vertices, '', color)
    
    def _set_axis_limits_same(self, ax1, ax2, vertices1, vertices2):
        """Set same axis limits for two axes based on combined bounds of both vertex sets."""
        # Handle both torch tensors and numpy arrays
        if isinstance(vertices1, torch.Tensor):
            vertices1 = vertices1.detach().cpu().numpy()
        if isinstance(vertices2, torch.Tensor):
            vertices2 = vertices2.detach().cpu().numpy()
        margin = 0.15  # Increased margin to ensure all points are visible
        
        # Combine bounds from both vertex sets
        x_min = min(vertices1[:, 0].min(), vertices2[:, 0].min())
        x_max = max(vertices1[:, 0].max(), vertices2[:, 0].max())
        y_min = min(vertices1[:, 1].min(), vertices2[:, 1].min())
        y_max = max(vertices1[:, 1].max(), vertices2[:, 1].max())
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Add absolute minimum margin to avoid zero range issues
        x_margin = max(margin * x_range, 0.05)
        y_margin = max(margin * y_range, 0.05)
        
        ax1.set_xlim(x_min - x_margin, x_max + x_margin)
        ax2.set_xlim(x_min - x_margin, x_max + x_margin)
        ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)
        
        if ax1.name == '3d':
            z_min = min(vertices1[:, 2].min(), vertices2[:, 2].min())
            z_max = max(vertices1[:, 2].max(), vertices2[:, 2].max())
            z_range = z_max - z_min
            z_margin = max(margin * z_range, 0.05)
            ax1.set_zlim(z_min - z_margin, z_max + z_margin)
            ax2.set_zlim(z_min - z_margin, z_max + z_margin)
    
    def _create_error_scatter(self, ax, gt_verts, pred_verts, errors, max_error, view_angle):
        """Create scatter plot colored by error."""
        gt_verts = gt_verts.detach().cpu().numpy() if isinstance(gt_verts, torch.Tensor) else gt_verts
        pred_verts = pred_verts.detach().cpu().numpy() if isinstance(pred_verts, torch.Tensor) else pred_verts
        errors = errors.detach().cpu().numpy() if isinstance(errors, torch.Tensor) else errors
        
        norm = Normalize(vmin=0, vmax=max_error)
        cmap = cm.RdYlGn_r  # Red for high error, Green for low error
        
        if view_angle == '3d':
            scatter = ax.scatter(gt_verts[:, 0], gt_verts[:, 1], gt_verts[:, 2], 
                                c=errors, cmap=cmap, norm=norm, s=20)
            fig = ax.figure
            fig.colorbar(scatter, ax=ax, label='Error')
        else:
            scatter = ax.scatter(gt_verts[:, 0], gt_verts[:, 1], c=errors, cmap=cmap, norm=norm, s=20)
            fig = ax.figure
            fig.colorbar(scatter, ax=ax, label='Error')
        
        return scatter
    
    def _update_error_scatter(self, scatter, gt_verts, pred_verts, errors, max_error, view_angle, ax):
        """Update error scatter plot."""
        gt_verts = gt_verts.detach().cpu().numpy() if isinstance(gt_verts, torch.Tensor) else gt_verts
        errors = errors.detach().cpu().numpy() if isinstance(errors, torch.Tensor) else errors
        
        scatter.remove()
        self._create_error_scatter(ax, gt_verts, pred_verts, errors, max_error, view_angle)
    
    def _images_to_video(self, images_dir, output_path):
        """Convert a directory of PNG images to MP4 video."""
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        
        if len(image_files) == 0:
            logger.warning(f"No images found in {images_dir}")
            return
        
        # Read first image to get dimensions
        first_frame = cv2.imread(os.path.join(images_dir, image_files[0]))
        height, width = first_frame.shape[:2]
        
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                                 self.fps, (width, height))
        
        for image_file in tqdm(image_files, desc="Creating video"):
            image_path = os.path.join(images_dir, image_file)
            frame = cv2.imread(image_path)
            writer.write(frame)
        
        writer.release()
        logger.info(f"Video saved: {output_path}")
    
    def _cleanup_temp_dir(self, temp_dir):
        """Remove temporary directory and its contents."""
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def visualize_level_results(trainer, 
                            mlvl_mech_info, 
                            output_dir=None,
                            num_frames=None):
    """
    Main function to visualize all levels' ground truth and rendered results.
    
    Args:
        trainer: E2EReductionTrainer instance
        mlvl_mech_info: List of mechanical info for each level
        output_dir: Output directory (default: trainer.args.dump_dir + '/visualization')
        num_frames: Number of frames to render (default: cfg.train_frame + cfg.test_frame)
        
    Returns:
        List of output video paths
    """
    if output_dir is None:
        output_dir = os.path.join(trainer.args.dump_dir, 'visualization')
    
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = LevelVisualizer(output_dir)
    output_videos = []
    
    frame_len = cfg.train_frame + cfg.test_frame
    if num_frames is not None:
        frame_len = min(frame_len, num_frames)
    
    logger.info(f"Starting visualization for {len(trainer.mlvl_simulators)} levels...")

    for level_idx, simulator in enumerate(trainer.mlvl_simulators):
        logger.info(f"Visualizing Level {level_idx}...")
        
        # Get GT data
        gt_object_points = trainer.m_gt_object_points[level_idx]  # [T, N, 3]

        # Run simulation with predicted parameters
        if level_idx < len(mlvl_mech_info) and mlvl_mech_info[level_idx] is not None:
            mech_info = mlvl_mech_info[level_idx]

            if 'log_spring_Y' in mech_info:
                wp_predicted_spring_Y = wp.from_torch(
                    mech_info['log_spring_Y'].contiguous(), dtype=wp.float32, requires_grad=False
                )
                simulator.set_spring_Y(wp_predicted_spring_Y)
            
            if 'drag_damping' in mech_info:
                wp_predicted_drag_damping = wp.from_torch(
                    mech_info['drag_damping'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False
                )
                simulator.set_drag_damping(wp_predicted_drag_damping)
            
            if 'dashpot_damping' in mech_info:
                wp_predicted_dashpot_damping = wp.from_torch(
                    mech_info['dashpot_damping'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False
                )
                simulator.set_dashpot_damping(wp_predicted_dashpot_damping)
            
            if 'collision_elas' in mech_info:
                wp_predicted_collision_elas = wp.from_torch(
                    mech_info['collision_elas'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False
                )
                simulator.set_collision_elas(wp_predicted_collision_elas)
            
            if 'collision_fric' in mech_info:
                wp_predicted_collision_fric = wp.from_torch(
                    mech_info['collision_fric'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False
                )
                simulator.set_collision_fric(wp_predicted_collision_fric)
            
            if 'collision_object_elas' in mech_info:
                wp_predicted_collision_object_elas = wp.from_torch(
                    mech_info['collision_object_elas'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False
                )
                simulator.set_collision_object_elas(wp_predicted_collision_object_elas)
            
            if 'collision_object_fric' in mech_info:
                wp_predicted_collision_object_fric = wp.from_torch(
                    mech_info['collision_object_fric'].reshape(1).contiguous(), dtype=wp.float32, requires_grad=False
                )
                simulator.set_collision_object_fric(wp_predicted_collision_object_fric)

        # Initialize simulation
        simulator.set_init_state(simulator.wp_init_vertices, 
                                 simulator.wp_init_velocities)
        
        # Get node_type for this level to determine the full vertex dimension
        # node_type: 0=object, 1=surface, 2=interior, 3=controller
        if level_idx < len(trainer.m_node_type):
            node_type = trainer.m_node_type[level_idx]
            object_mask = (node_type == 0)  # Only keep object points (type 0)
            num_total_points = len(node_type)  # Total number of points including all types
            num_object_points = object_mask.sum().item() if isinstance(object_mask, torch.Tensor) else int(object_mask.sum())
            
            logger.info(f"Level {level_idx}: node_type has {num_total_points} points, {num_object_points} are object points")
        else:
            node_type = None
            num_total_points = None
            num_object_points = None
            logger.warning(f"Level {level_idx}: node_type not available")
        
        # Collect predicted trajectory
        pred_vertices_full = []  # Full trajectory with all point types
        
        for frame_idx in tqdm(range(frame_len), desc=f"Simulating Level {level_idx}"):
            # Get object points (wp_x) - this is what the simulator returns
            x = wp.to_torch(simulator.wp_states[0].wp_x, requires_grad=False).cpu().numpy()

            if cfg.data_type == "real":
                simulator.set_controller_target(frame_idx, pure_inference=False)
            
            if simulator.object_collision_flag:
                simulator.update_collision_graph()
            
            if cfg.use_graph:
                wp.capture_launch(simulator.forward_graph)
            else:
                simulator.step()
            
            if node_type is not None:
                # Create a full array with the correct dimension
                full_x = np.zeros((num_object_points, 3), dtype=np.float32)
                # Fill in object points from simulation
       
                full_x[:num_object_points] = x[:num_object_points]
                pred_vertices_full.append(full_x)
            else:
                pred_vertices_full.append(x)
            
            simulator.set_init_state(simulator.wp_states[-1].wp_x, 
                                     simulator.wp_states[-1].wp_v)
        
        pred_vertices = np.stack(pred_vertices_full, axis=0)  # [T, N_total, 3]
        gt_vertices = gt_object_points.cpu().numpy()[:frame_len]  # [T, N_gt, 3]
        
        
        # Filter pred_vertices to only include object points
        if node_type is not None:
            pred_vertices_filtered = pred_vertices[:, object_mask, :]
            logger.info(f"Level {level_idx}: Filtered pred vertices from {pred_vertices.shape[1]} to {pred_vertices_filtered.shape[1]} object points (node_type==0)")
        else:
            pred_vertices_filtered = pred_vertices
            logger.warning(f"Level {level_idx}: node_type not available, using all {pred_vertices.shape[1]} points")
        
        # Check if shapes match for trajectory error visualization
        shapes_match = (gt_vertices.shape == pred_vertices_filtered.shape)
        import pdb; pdb.set_trace()
        if not shapes_match:
            logger.warning(f"Level {level_idx}: GT shape {gt_vertices.shape} != Pred shape {pred_vertices_filtered.shape}, skipping trajectory error visualization")
        import pdb; pdb.set_trace()
        # Create comparison video (point cloud only, no edges)
        video_path = visualizer.visualize_level_comparison(
            level_idx=level_idx,
            gt_vertices=gt_vertices,
            pred_vertices=pred_vertices_filtered,
            output_filename=f"level_{level_idx}_comparison.mp4",
            title_prefix=f"Level {level_idx}"
        )
        output_videos.append(video_path)
        
    
    logger.info(f"All visualization videos saved to: {output_dir}")
    return output_videos
