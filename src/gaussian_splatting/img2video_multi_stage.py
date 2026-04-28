import os
import imageio.v2 as imageio
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Convert images to video for multi-stage rendering')
parser.add_argument('--image_folder', type=str, help='Path of image folder (base output directory)')
parser.add_argument('--video_folder', type=str, default=None, help='Output video folder (default: same as image_folder)')
parser.add_argument('--fps', type=int, default=15, help='Frame per second')
parser.add_argument('--view', type=int, default=0, help='View index to convert')
parser.add_argument('--pattern', type=str, default='stage_*', help='Pattern for stage directories (e.g., stage_*, stage_*)')
args = parser.parse_args()


def convert_images_to_video(image_folder, video_path, fps=15):
    """
    Convert a sequence of images to a video
    
    Args:
        image_folder: Path to folder containing images
        video_path: Output video path
        fps: Frames per second
    """
    images_path = sorted([img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")])
    
    if len(images_path) == 0:
        print(f"No images found in {image_folder}")
        return False
    
    frame_series = []
    for image_path in images_path:
        image = imageio.imread(os.path.join(image_folder, image_path)).astype(np.uint8)
        h = image.shape[0] if image.shape[0] % 2 == 0 else image.shape[0] - 1
        w = image.shape[1] if image.shape[1] % 2 == 0 else image.shape[1] - 1
        frame_series.append(image[:h, :w])
    
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    imageio.mimsave(video_path, frame_series, fps=fps, macro_block_size=1)
    print(f"Video saved: {video_path} ({len(frame_series)} frames)")
    return True


def convert_multi_stage_to_video(
    base_image_folder,
    output_video_folder=None,
    fps=15,
    view_idx=0,
    stage_pattern='stage_*'
):
    """
    Convert multi-stage rendered images to videos
    
    Args:
        base_image_folder: Base output directory (e.g., ./gaussian_output_dynamic/scene_name)
        output_video_folder: Output folder for videos (default: base_image_folder/videos)
        fps: Frames per second
        view_idx: View index to convert
        stage_pattern: Pattern for stage directories
    """
    import glob
    
    if output_video_folder is None:
        output_video_folder = os.path.join(base_image_folder, 'videos')
    
    os.makedirs(output_video_folder, exist_ok=True)
    
    # Find all stage directories
    stage_dirs = sorted(glob.glob(os.path.join(base_image_folder, stage_pattern)))
    
    if len(stage_dirs) == 0:
        print(f"No stage directories found matching pattern '{stage_pattern}' in {base_image_folder}")
        # Try to convert as single-stage (no stage subdirectories)
        view_folder = os.path.join(base_image_folder, str(view_idx))
        if os.path.exists(view_folder):
            video_path = os.path.join(output_video_folder, f'view_{view_idx}.mp4')
            convert_images_to_video(view_folder, video_path, fps)
        else:
            # Try direct folder
            video_path = os.path.join(output_video_folder, 'output.mp4')
            convert_images_to_video(base_image_folder, video_path, fps)
        return
    
    print(f"Found {len(stage_dirs)} stage directories")
    
    for stage_dir in stage_dirs:
        stage_name = os.path.basename(stage_dir)
        print(f"\nProcessing {stage_name}...")
        
        # Get view folder
        view_folder = os.path.join(stage_dir, str(view_idx))
        
        if not os.path.exists(view_folder):
            print(f"  View folder not found: {view_folder}")
            continue
        
        # Create video
        video_path = os.path.join(output_video_folder, f'{stage_name}_view{view_idx}.mp4')
        convert_images_to_video(view_folder, video_path, fps)


def create_comparison_video(
    base_image_folder,
    output_video_folder=None,
    fps=15,
    view_idx=0,
    stage_pattern='stage_*'
):
    """
    Create a side-by-side comparison video of all stages
    
    Args:
        base_image_folder: Base output directory
        output_video_folder: Output folder for videos
        fps: Frames per second
        view_idx: View index
        stage_pattern: Pattern for stage directories
    """
    import glob
    import cv2
    
    if output_video_folder is None:
        output_video_folder = os.path.join(base_image_folder, 'videos')
    
    os.makedirs(output_video_folder, exist_ok=True)
    
    # Find all stage directories
    stage_dirs = sorted(glob.glob(os.path.join(base_image_folder, stage_pattern)))
    
    if len(stage_dirs) < 2:
        print("Need at least 2 stages for comparison video")
        return
    
    print(f"Creating comparison video from {len(stage_dirs)} stages...")
    
    # Load all frame sequences
    stage_frames = {}
    max_frames = 0
    
    for stage_dir in stage_dirs:
        stage_name = os.path.basename(stage_dir)
        view_folder = os.path.join(stage_dir, str(view_idx))
        
        if not os.path.exists(view_folder):
            continue
        
        images_path = sorted([img for img in os.listdir(view_folder) if img.endswith(".png") or img.endswith(".jpg")])
        frames = []
        for image_path in images_path:
            image = imageio.imread(os.path.join(view_folder, image_path))
            h = image.shape[0] if image.shape[0] % 2 == 0 else image.shape[0] - 1
            w = image.shape[1] if image.shape[1] % 2 == 0 else image.shape[1] - 1
            frames.append(image[:h, :w])
        
        stage_frames[stage_name] = frames
        max_frames = max(max_frames, len(frames))
    
    if len(stage_frames) < 2:
        print("Need at least 2 stages with valid frames for comparison video")
        return
    
    # Determine output size (2x width for 2 stages, or Nx for more)
    n_stages = len(stage_frames)
    sample_frame = list(stage_frames.values())[0][0]
    h, w = sample_frame.shape[:2]
    
    # Arrange in 2 columns if more than 2 stages
    n_cols = min(2, n_stages)
    n_rows = (n_stages + n_cols - 1) // n_cols
    output_w = w * n_cols
    output_h = h * n_rows
    
    # Create video
    video_path = os.path.join(output_video_folder, f'stage_comparison_view{view_idx}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (output_w, output_h))
    
    for frame_idx in range(max_frames):
        # Create blank frame
        combined_frame = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        
        stage_list = list(stage_frames.keys())
        for idx, stage_name in enumerate(stage_list):
            row = idx // n_cols
            col = idx % n_cols
            
            frames = stage_frames[stage_name]
            frame = frames[frame_idx % len(frames)]  # Loop if fewer frames
            
            # Add stage label
            labeled_frame = frame.copy()
            cv2.putText(labeled_frame, stage_name, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            y_start = row * h
            y_end = (row + 1) * h
            x_start = col * w
            x_end = (col + 1) * w
            
            combined_frame[y_start:y_end, x_start:x_end] = labeled_frame[:h, :w]
        
        out.write(combined_frame)
    
    out.release()
    print(f"Comparison video saved: {video_path}")


if __name__ == "__main__":
    image_folder = args.image_folder
    video_folder = args.video_folder
    fps = args.fps
    
    # if args.stage is not None:
    #     # Convert specific stage
    #     stage_dir = os.path.join(image_folder, f'stage_{args.stage}')
    #     view_folder = os.path.join(stage_dir, str(args.view))
        
    #     if video_folder is None:
    #         video_folder = os.path.join(image_folder, 'videos')
        
    #     video_path = os.path.join(video_folder, f'stage_{args.stage}_view{args.view}.mp4')
    #     convert_images_to_video(view_folder, video_path, fps)
    # else:

    # Convert all stages
    convert_multi_stage_to_video(
        image_folder,
        video_folder,
        fps,
        args.view,
        args.pattern
    )
    
    # # Also create comparison video if multiple stages exist
    # create_comparison_video(
    #     image_folder,
    #     video_folder,
    #     fps,
    #     args.view,
    #     args.pattern
    # )