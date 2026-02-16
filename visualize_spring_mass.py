"""
Visualization script for spring mass simulation.
This script loads a trained model and best_mech_info, runs pure inference simulation,
and creates a PyVista 3D visualization video.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.distributed as dist
from argparse import ArgumentParser
import random
import numpy as np

# Import the trainer
from src.spring_mass_trainer import SpringMassTrainer

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = ArgumentParser(description='Visualize spring mass simulation')

    # Model and data paths
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the data directory')
    parser.add_argument('--dump_dir', type=str, required=True,
                        help='Path to the model checkpoint directory')
    parser.add_argument('--mech_info_path', type=str, required=True,
                        help='Path to the best_mech_info .pth file')
    parser.add_argument('--output_video', type=str, default='simulation_visualization.mp4',
                        help='Output video path')

    # Model parameters (should match training config)
    parser.add_argument('--case', type=str, default='spring_mass')
    parser.add_argument('--object_case', type=str, required=True,
                        help='Object case name (e.g., cylinder, cloth, etc.)')
    parser.add_argument('--space_dim', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--multi_mesh_layer', type=int, default=7)
    parser.add_argument('--pre_layer_num', type=int, default=2)
    parser.add_argument('--bottom_layer_num', type=int, default=2)
    parser.add_argument('--hidden_depth', type=int, default=2)
    parser.add_argument('--mp_time', type=int, default=7)
    parser.add_argument('--enhance', action='store_true')
    parser.add_argument('--agg_conv_pos', action='store_true')

    # Training parameters (not used for visualization but needed for initialization)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_valid', type=int, default=0)
    parser.add_argument('--n_test', type=int, default=0)

    # Dataset parameters
    parser.add_argument('--noise_level', type=float, default=0.0)
    parser.add_argument('--noise_gamma', type=float, default=0.0)
    parser.add_argument('--recal_mesh', action='store_true')
    parser.add_argument('--consist_mesh', action='store_true')
    parser.add_argument('--train_frame', type=int, default=10)
    parser.add_argument('--test_frame', type=int, default=48)

    # Model loading
    parser.add_argument('--path', type=str, default='',
                        help='Path to pretrained model')
    parser.add_argument('--restart_epoch', type=int, default=-1)
    parser.add_argument('--scratch', action='store_true')

    # Distributed training (required for initialization)
    parser.add_argument('--local_rank', type=int, default=0)

    return parser.parse_args()


def main():
    # Set random seed
    seed = 42
    set_all_seeds(seed)

    # Parse arguments
    args = parse_args()

    # Initialize distributed training (even for single GPU)
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    torch.cuda.set_device(args.local_rank)

    # Set device
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model directory: {args.dump_dir}")
    print(f"Mech info path: {args.mech_info_path}")
    print(f"Output video: {args.output_video}")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = SpringMassTrainer(args, device)

    # Run visualization
    print("\nStarting visualization...")
    try:
        obj_seq, ctrl_seq, gt_seq = trainer.visualize_simulation(
            mech_info_path=args.mech_info_path,
            output_video_path=args.output_video
        )

        print(f"\n✓ Visualization completed successfully!")
        print(f"  - Video saved to: {args.output_video}")
        print(f"  - Data saved to: {args.output_video.replace('.mp4', '.h5')}")
        print(f"  - Total frames: {len(obj_seq)}")
        print(f"  - Object points per frame: {obj_seq[0].shape[0]}")
        print(f"  - Controller points per frame: {ctrl_seq[0].shape[0]}")

    except Exception as e:
        print(f"\n✗ Visualization failed with error:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    # Cleanup
    dist.destroy_process_group()

    return 0


if __name__ == '__main__':
    exit(main())
