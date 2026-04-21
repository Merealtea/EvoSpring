#!/bin/bash

# Multi-stage simulation and rendering script for EvoSpring
# This script runs multi-stage rendering and converts results to videos

output_dir="./gaussian_output_dynamic"

# Camera views to render
views=("0")

# Scene names
scenes=(
    "double_lift_cloth_1"
    "double_lift_cloth_3"
    "double_lift_sloth"
    "double_lift_zebra"
    "double_stretch_sloth"
    "double_stretch_zebra"
    "rope_double_hand"
    "single_lift_dinosor"
    "single_lift_rope"
    "single_lift_sloth"
    "single_lift_zebra"
    "single_push_rope"
    "single_push_rope_1"
    "single_push_rope_4"
    "single_push_sloth"
    "weird_package"
)

# Experiment name (Gaussian model checkpoint)
exp_name='init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0'


# Process each scene
for scene_name in "${scenes[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing scene: ${scene_name}"
    echo "========================================"
    
    
    # Run multi-stage rendering
    python gs_render_multi_stage.py \
        -s ./data/gaussian_data/${scene_name} \
        -m ./gaussian_output/${scene_name}/${exp_name} \
        --name ${scene_name} \
        --output_dir ${output_dir} 
    
    # Convert each stage's rendered images to video
    for view_name in "${views[@]}"; do
        echo ""
        echo "Converting stage images to videos for view ${view_name}..."
        
        python src/gaussian_splatting/img2video_multi_stage.py \
            --image_folder ${output_dir}/${scene_name} \
            --video_folder ${output_dir}/${scene_name}/videos \
            --view ${view_name} \
            --fps 15
    done
    
    echo ""
    echo "Scene ${scene_name} completed!"
    echo ""
done

echo ""
echo "========================================"
echo "All scenes completed!"
echo "========================================"