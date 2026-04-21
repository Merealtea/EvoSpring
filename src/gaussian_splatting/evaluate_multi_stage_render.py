import os
from PIL import Image
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
import json
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import numpy as np
import glob
import csv


def img2tensor(img):
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
    img = img.transpose(2, 0, 1)  # Change shape from (H, W, C) to (C, H, W)
    return torch.from_numpy(img).unsqueeze(0).cuda()


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 1.0


def evaluate_frame(gt, gt_mask, render, render_mask, human_mask):
    """
    Evaluate a single frame and return metrics
    """
    gt_mask = gt_mask.astype(np.float32) / 255.
    inv_human_mask = (1.0 - human_mask / 255.).astype(np.float32)

    gt = gt.astype(np.float32) * gt_mask[..., None]
    bg_mask = gt_mask == 0
    gt[bg_mask] = [0, 0, 0]
    render = render[:, :, :3].astype(np.float32)

    gt = gt * inv_human_mask[..., None]
    render = render * inv_human_mask[..., None]
    render_mask = render_mask * inv_human_mask

    gt_tensor = img2tensor(gt)
    render_tensor = img2tensor(render)

    psnr_val = psnr(render_tensor, gt_tensor).item()
    ssim_val = ssim(render_tensor, gt_tensor).item()
    lpips_val = lpips(render_tensor, gt_tensor).item()
    iou_val = compute_iou(gt_mask > 0, render_mask > 0)

    return psnr_val, ssim_val, lpips_val, iou_val


def load_stage_render_data(output_dir, scene, stage_idx):
    """
    Load render data for a specific stage
    Returns the output directory path for the stage
    """
    # Check if stage-specific directory exists
    stage_output_dir = os.path.join(output_dir, scene, f"stage_{stage_idx}")
    
    if not os.path.exists(stage_output_dir):
        # Fallback to original structure without stage subdirectory
        stage_output_dir = os.path.join(output_dir, scene)
        if not os.path.exists(stage_output_dir):
            return None
    
    return stage_output_dir


def evaluate_multi_stage_render():
    """
    Multi-stage render evaluation main function
    Evaluates rendering metrics (PSNR, SSIM, LPIPS, IoU) for each stage
    """
    # Configuration
    render_path = './data/render_eval_data'
    human_mask_path = "./data/different_types_human_mask"
    root_data_dir = './data/gaussian_data'
    output_dir = './gaussian_output_dynamic'
    
    log_dir = './results'
    os.makedirs(log_dir, exist_ok=True)
    
    # Multi-stage configuration
    
    log_file_path = os.path.join(log_dir, 'output_multi_stage_render.txt')
    csv_file_path = os.path.join(log_dir, 'multi_stage_render_results.csv')

    with open(log_file_path, 'w') as log_file:
        scene_name = sorted(os.listdir(render_path))

        all_stage_metrics = []  # Store metrics for all scenes and stages

        for scene in scene_name:
            print(f"\n{'='*60}")
            print(f"Processing scene: {scene}")
            print(f"{'='*60}\n")

            scene_dir = os.path.join(output_dir, scene)
            render_path_dir = os.path.join(render_path, scene)
            human_mask_dir = os.path.join(human_mask_path, scene)
            
            # Load frame split info
            with open(f"{root_data_dir}/split.json", 'r') as f:
                info = json.load(f)
            frame_len = info['frame_len']
            train_f_idx_range = list(range(info["train"][0] + 1, info["train"][1]))
            test_f_idx_range = list(range(info["test"][0], info["test"][1]))

            print(f"Train indices range from {train_f_idx_range[0]} to {train_f_idx_range[-1]}")
            print(f"Test indices range from {test_f_idx_range[0]} to {test_f_idx_range[-1]}")

            scene_stage_results = {
                'scene': scene,
                'stages': []
            }

            num_stage = 0
            stages_pic_dir = os.listdir(scene_dir)
            for dir_name in stages_pic_dir:
                if 'stage' in dir_name:
                    num_stage += 1

            # Evaluate each stage
            for stage_idx in range(num_stage):
                print(f"\n--- Evaluating Stage {stage_idx} ---")

                # Load stage-specific render output
                stage_output_dir = load_stage_render_data(output_dir, scene, stage_idx)
                
                if stage_output_dir is None:
                    print(f"Warning: Stage {stage_idx} output not found for scene {scene}")
                    scene_stage_results['stages'].append({
                        'stage': stage_idx,
                        'psnr_train': None,
                        'ssim_train': None,
                        'lpips_train': None,
                        'iou_train': None,
                        'psnr_test': None,
                        'ssim_test': None,
                        'lpips_test': None,
                        'iou_test': None,
                        'train_frame_num': 0,
                        'test_frame_num': 0
                    })
                    continue

                psnrs_train, ssims_train, lpipss_train, ious_train = [], [], [], []
                psnrs_test, ssims_test, lpipss_test, ious_test = [], [], [], []

                # Evaluate only the first view (can be extended to multiple views)
                for view_idx in range(1):
                    # Evaluate training frames
                    for frame_idx in tqdm(train_f_idx_range, desc=f"Stage {stage_idx} Train"):
                        gt = np.array(Image.open(os.path.join(render_path_dir, 'color', str(view_idx), f'{frame_idx}.png')))
                        gt_mask = np.array(Image.open(os.path.join(render_path_dir, 'mask', str(view_idx), f'{frame_idx}.png')))
                        render = np.array(Image.open(os.path.join(stage_output_dir, str(view_idx), f'{frame_idx:05d}.png')))
                        render_mask = render[:, :, 3] if render.shape[-1] == 4 else np.ones_like(render[:, :, 0])
                        human_mask = np.array(Image.open(os.path.join(human_mask_dir, 'mask', str(view_idx), '0', f'{frame_idx}.png')))

                        psnr_val, ssim_val, lpips_val, iou_val = evaluate_frame(
                            gt, gt_mask, render, render_mask, human_mask
                        )

                        psnrs_train.append(psnr_val)
                        ssims_train.append(ssim_val)
                        lpipss_train.append(lpips_val)
                        ious_train.append(iou_val)

                    # Evaluate test frames
                    for frame_idx in tqdm(test_f_idx_range, desc=f"Stage {stage_idx} Test"):
                        gt = np.array(Image.open(os.path.join(render_path_dir, 'color', str(view_idx), f'{frame_idx}.png')))
                        gt_mask = np.array(Image.open(os.path.join(render_path_dir, 'mask', str(view_idx), f'{frame_idx}.png')))
                        render = np.array(Image.open(os.path.join(stage_output_dir, str(view_idx), f'{frame_idx:05d}.png')))
                        render_mask = render[:, :, 3] if render.shape[-1] == 4 else np.ones_like(render[:, :, 0])
                        human_mask = np.array(Image.open(os.path.join(human_mask_dir, 'mask', str(view_idx), '0', f'{frame_idx}.png')))

                        psnr_val, ssim_val, lpips_val, iou_val = evaluate_frame(
                            gt, gt_mask, render, render_mask, human_mask
                        )

                        psnrs_test.append(psnr_val)
                        ssims_test.append(ssim_val)
                        lpipss_test.append(lpips_val)
                        ious_test.append(iou_val)

                stage_metrics = {
                    'stage': stage_idx,
                    'psnr_train': np.mean(psnrs_train) if psnrs_train else None,
                    'ssim_train': np.mean(ssims_train) if ssims_train else None,
                    'lpips_train': np.mean(lpipss_train) if lpipss_train else None,
                    'iou_train': np.mean(ious_train) if ious_train else None,
                    'psnr_test': np.mean(psnrs_test) if psnrs_test else None,
                    'ssim_test': np.mean(ssims_test) if ssims_test else None,
                    'lpips_test': np.mean(lpipss_test) if lpipss_test else None,
                    'iou_test': np.mean(ious_test) if ious_test else None,
                    'train_frame_num': len(psnrs_train),
                    'test_frame_num': len(psnrs_test)
                }

                scene_stage_results['stages'].append(stage_metrics)

                print(f"\nStage {stage_idx} Results:")
                print(f"\t PSNR (train): {stage_metrics['psnr_train']:.4f}" if stage_metrics['psnr_train'] else "\t PSNR (train): N/A")
                print(f"\t SSIM (train): {stage_metrics['ssim_train']:.4f}" if stage_metrics['ssim_train'] else "\t SSIM (train): N/A")
                print(f"\t LPIPS (train): {stage_metrics['lpips_train']:.4f}" if stage_metrics['lpips_train'] else "\t LPIPS (train): N/A")
                print(f"\t IoU (train): {stage_metrics['iou_train']:.4f}" if stage_metrics['iou_train'] else "\t IoU (train): N/A")
                print(f"\t PSNR (test): {stage_metrics['psnr_test']:.4f}" if stage_metrics['psnr_test'] else "\t PSNR (test): N/A")
                print(f"\t SSIM (test): {stage_metrics['ssim_test']:.4f}" if stage_metrics['ssim_test'] else "\t SSIM (test): N/A")
                print(f"\t LPIPS (test): {stage_metrics['lpips_test']:.4f}" if stage_metrics['lpips_test'] else "\t LPIPS (test): N/A")
                print(f"\t IoU (test): {stage_metrics['iou_test']:.4f}" if stage_metrics['iou_test'] else "\t IoU (test): N/A")

            all_stage_metrics.append(scene_stage_results)

        # Write overall results
        print('\n' + '='*80)
        print('Overall Results Across All Scenes and Stages')
        print('='*80)

        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("OVERALL RESULTS ACROSS ALL SCENES AND STAGES\n")
        log_file.write("=" * 80 + "\n\n")

        for stage_idx in range(num_stage):
            all_psnrs_train = []
            all_ssims_train = []
            all_lpipss_train = []
            all_ious_train = []
            all_psnrs_test = []
            all_ssims_test = []
            all_lpipss_test = []
            all_ious_test = []

            for scene_result in all_stage_metrics:
                stage_result = scene_result['stages'][stage_idx] if stage_idx < len(scene_result['stages']) else None
                if stage_result and stage_result['psnr_train'] is not None:
                    all_psnrs_train.append(stage_result['psnr_train'])
                    all_ssims_train.append(stage_result['ssim_train'])
                    all_lpipss_train.append(stage_result['lpips_train'])
                    all_ious_train.append(stage_result['iou_train'])
                    all_psnrs_test.append(stage_result['psnr_test'])
                    all_ssims_test.append(stage_result['ssim_test'])
                    all_lpipss_test.append(stage_result['lpips_test'])
                    all_ious_test.append(stage_result['iou_test'])

            if all_psnrs_train:
                print(f'\nStage {stage_idx}:')
                print(f'\t Overall PSNR (train): {np.mean(all_psnrs_train):.4f}')
                print(f'\t Overall SSIM (train): {np.mean(all_ssims_train):.4f}')
                print(f'\t Overall LPIPS (train): {np.mean(all_lpipss_train):.4f}')
                print(f'\t Overall IoU (train): {np.mean(all_ious_train):.4f}')
                print(f'\t Overall PSNR (test): {np.mean(all_psnrs_test):.4f}')
                print(f'\t Overall SSIM (test): {np.mean(all_ssims_test):.4f}')
                print(f'\t Overall LPIPS (test): {np.mean(all_lpipss_test):.4f}')
                print(f'\t Overall IoU (test): {np.mean(all_ious_test):.4f}')

                log_file.write(f"\nStage {stage_idx}:\n")
                log_file.write(f"\t Overall PSNR (train): {np.mean(all_psnrs_train):.6f}\n")
                log_file.write(f"\t Overall SSIM (train): {np.mean(all_ssims_train):.6f}\n")
                log_file.write(f"\t Overall LPIPS (train): {np.mean(all_lpipss_train):.6f}\n")
                log_file.write(f"\t Overall IoU (train): {np.mean(all_ious_train):.6f}\n")
                log_file.write(f"\t Overall PSNR (test): {np.mean(all_psnrs_test):.6f}\n")
                log_file.write(f"\t Overall SSIM (test): {np.mean(all_ssims_test):.6f}\n")
                log_file.write(f"\t Overall LPIPS (test): {np.mean(all_lpipss_test):.6f}\n")
                log_file.write(f"\t Overall IoU (test): {np.mean(all_ious_test):.6f}\n")

        # Write compact table to log file
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("COMPACT METRICS TABLE BY SCENE AND STAGE\n")
        log_file.write("=" * 80 + "\n\n")

        header = f"{'Scene':<30} | {'Stage':<6} | {'PSNR-train':<12} | {'SSIM-train':<12} | {'LPIPS-train':<14} | {'IoU-train':<12} | "
        header += f"{'PSNR-test':<12} | {'SSIM-test':<12} | {'LPIPS-test':<14} | {'IoU-test':<12}\n"
        log_file.write(header)
        log_file.write("-" * 180 + "\n")

        for scene_result in all_stage_metrics:
            scene = scene_result['scene']
            for stage_result in scene_result['stages']:
                stage = stage_result['stage']
                psnr_train = f"{stage_result['psnr_train']:.6f}" if stage_result['psnr_train'] is not None else "N/A"
                ssim_train = f"{stage_result['ssim_train']:.6f}" if stage_result['ssim_train'] is not None else "N/A"
                lpips_train = f"{stage_result['lpips_train']:.6f}" if stage_result['lpips_train'] is not None else "N/A"
                iou_train = f"{stage_result['iou_train']:.6f}" if stage_result['iou_train'] is not None else "N/A"
                psnr_test = f"{stage_result['psnr_test']:.6f}" if stage_result['psnr_test'] is not None else "N/A"
                ssim_test = f"{stage_result['ssim_test']:.6f}" if stage_result['ssim_test'] is not None else "N/A"
                lpips_test = f"{stage_result['lpips_test']:.6f}" if stage_result['lpips_test'] is not None else "N/A"
                iou_test = f"{stage_result['iou_test']:.6f}" if stage_result['iou_test'] is not None else "N/A"

                row = f"{scene[:30]:<30} | {stage:<6} | {psnr_train:<12} | {ssim_train:<12} | {lpips_train:<14} | {iou_train:<12} | "
                row += f"{psnr_test:<12} | {ssim_test:<12} | {lpips_test:<14} | {iou_test:<12}\n"
                log_file.write(row)

        print(f"\nMetrics have been saved to: {log_file_path}")

    # Write CSV file
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        header = [
            "Scene Name",
            "Stage",
            "Train Frame Num",
            "Train PSNR",
            "Train SSIM",
            "Train LPIPS",
            "Train IoU",
            "Test Frame Num",
            "Test PSNR",
            "Test SSIM",
            "Test LPIPS",
            "Test IoU"
        ]
        writer.writerow(header)

        # Data rows
        for scene_result in all_stage_metrics:
            scene = scene_result['scene']
            for stage_result in scene_result['stages']:
                row = [
                    scene,
                    stage_result['stage'],
                    stage_result.get('train_frame_num', 'N/A'),
                    stage_result.get('psnr_train', 'N/A'),
                    stage_result.get('ssim_train', 'N/A'),
                    stage_result.get('lpips_train', 'N/A'),
                    stage_result.get('iou_train', 'N/A'),
                    stage_result.get('test_frame_num', 'N/A'),
                    stage_result.get('psnr_test', 'N/A'),
                    stage_result.get('ssim_test', 'N/A'),
                    stage_result.get('lpips_test', 'N/A'),
                    stage_result.get('iou_test', 'N/A')
                ]
                writer.writerow(row)

    print(f"CSV results have been saved to: {csv_file_path}")


if __name__ == "__main__":
    print("Starting multi-stage render evaluation...")
    evaluate_multi_stage_render()
    print("\nMulti-stage render evaluation completed!")