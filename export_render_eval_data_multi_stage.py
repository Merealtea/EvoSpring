import os
import csv
import json
import argparse
import shutil

parser = argparse.ArgumentParser(description='Export multi-stage render evaluation data')
parser.add_argument('--base_path', type=str, default="./data/different_types",
                    help="Base path containing original data")
parser.add_argument('--output_path', type=str, default="./data/render_eval_data",
                    help="Output path for multi-stage data")
parser.add_argument('--controller_name', type=str, default="hand",
                    help="Name of the controller (e.g., 'hand')")
parser.add_argument('--picture_output_dir', type=str, default="gaussian_output_dynamic")

args = parser.parse_args()

base_path = args.base_path
output_path = args.output_path
CONTROLLER_NAME = args.controller_name

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def copy_file_or_dir(src, dst):
    """Copy file or directory, handling both cases"""
    if os.path.exists(src):
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        else:
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def process_stage(base_path, output_path, case_name, stage_idx=None):
    """
    Process data for a specific stage
    
    Args:
        base_path: Base path containing original data
        output_path: Output path for this stage
        case_name: Case/scene name
        stage_idx: Stage index (None for original data)
    """
    print(f"Processing {case_name}" + (f" (Stage {stage_idx})" if stage_idx is not None else ""))
    
    # Create the directory for the case
    existDir(output_path)
    existDir(f"{output_path}/mask")
    
    for i in range(3):
        # Copy the original RGB image
        color_src = f"{base_path}/color"
        color_dst = f"{output_path}/color"
        if os.path.exists(color_src):
            copy_file_or_dir(color_src, color_dst)
        
        # Copy only the object mask image
        # Get the mask path for the image
        mask_info_path = f"{base_path}/mask/mask_info_{i}.json"
        if not os.path.exists(mask_info_path):
            print(f"  Warning: Mask info not found: {mask_info_path}")
            continue
            
        with open(mask_info_path, "r") as f:
            data = json.load(f)
        
        obj_idx = None
        for key, value in data.items():
            if value != CONTROLLER_NAME:
                if obj_idx is not None:
                    print(f"  Warning: More than one object detected in {case_name}")
                obj_idx = int(key)
        
        if obj_idx is None:
            print(f"  Warning: No object found in {case_name}, using all masks")
            # Copy all masks if no object is detected
            mask_src = f"{base_path}/mask/{i}"
            mask_dst = f"{output_path}/mask/{i}"
            if os.path.exists(mask_src):
                copy_file_or_dir(mask_src, mask_dst)
        else:
            existDir(f"{output_path}/mask/{i}")
            mask_src = f"{base_path}/mask/{i}/{obj_idx}"
            mask_dst = f"{output_path}/mask/{i}/"
            if os.path.exists(mask_src):
                # Copy contents of the source directory
                for item in os.listdir(mask_src):
                    s = os.path.join(mask_src, item)
                    d = os.path.join(mask_dst, item)
                    if os.path.isfile(s):
                        shutil.copy2(s, d)
        
        # Copy the split.json
        split_src = f"{base_path}/split.json"
        split_dst = f"{output_path}/split.json"
        if os.path.exists(split_src):
            shutil.copy2(split_src, split_dst)


def export_multi_stage_data():
    """
    Export multi-stage render evaluation data
    Each stage will have its own directory with the processed data
    """
    existDir(output_path)
    
    # Read data_config.csv to get all cases
    config_path = "data_config.csv"

    with open(config_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        case_names = [row[0] for row in reader if row]
    
    for case_name in case_names:
        case_base_path = f"{base_path}/{case_name}"
        
        if not os.path.exists(case_base_path):
            print(f"Skipping {case_name}: Directory not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {case_name}")
        print(f"{'='*60}\n")
        
        # Export data for each stage
        stages_pic_dir = os.path.join(args.picture_output_dir, case_name)
        num_stage = 0
        for dir_name in stages_pic_dir:
            if 'stage' in dir_name:
                num_stage += 1

        for stage_idx in range(num_stage):
            stage_output_path = f"{output_path}/{case_name}_stage_{stage_idx}"
            process_stage(case_base_path, stage_output_path, case_name, stage_idx)
            print(f"  Stage {stage_idx} data exported to: {stage_output_path}")
        
        # Also export original/combined data (stage without suffix)
        original_output_path = f"{output_path}/{case_name}"
        process_stage(case_base_path, original_output_path, case_name, None)
        print(f"  Original data exported to: {original_output_path}")


def export_single_stage_data(stage_idx, source_stage_path=None):
    """
    Export data for a single specific stage
    
    Args:
        stage_idx: Stage index to export
        source_stage_path: Optional path to stage-specific source data
    """
    existDir(output_path)
    
    # Read data_config.csv to get all cases
    config_path = "data_config.csv"
    if not os.path.exists(config_path):
        for alt_path in ["../data_config.csv", "../../data_config.csv",
                         "/mnt/pool1/cxy/phystwin-v2/PhysTwin/data_config.csv"]:
            if os.path.exists(alt_path):
                config_path = alt_path
                break
    
    if not os.path.exists(config_path):
        case_names = [d for d in os.listdir(base_path)
                     if os.path.isdir(os.path.join(base_path, d))]
    else:
        with open(config_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            case_names = [row[0] for row in reader if row]
    
    for case_name in case_names:
        case_base_path = f"{base_path}/{case_name}"
        
        if not os.path.exists(case_base_path):
            continue
        
        print(f"\nProcessing {case_name} (Stage {stage_idx})...")
        
        # If source_stage_path is provided, use it as the base
        if source_stage_path:
            src_path = f"{source_stage_path}/{case_name}"
            if os.path.exists(src_path):
                case_base_path = src_path
        
        stage_output_path = f"{output_path}/{case_name}"
        process_stage(case_base_path, stage_output_path, case_name, stage_idx)


if __name__ == "__main__":
    print(f"Exporting multi-stage render evaluation data...")
    print(f"Base path: {args.base_path}")
    print(f"Output path: {args.output_path}")
    print(f"Controller name: {args.controller_name}")

    export_multi_stage_data()
    
    print(f"\n{'='*60}")
    print("Export completed!")
    print(f"{'='*60}")
    print(f"\nOutput structure:")
    print(f"  {output_path}/")