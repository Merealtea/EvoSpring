import glob
import json
import numpy as np
import cv2

base_path = "./data/different_types"
prediction_dir = "./gaussian_output_dynamic_learnable_white"
human_mask_path = (
    "./data/different_types_human_mask"
)
object_mask_path = (
    "./data/render_eval_data"
)

height, width = 480, 848
FPS = 30
alpha = 0.7

dir_names = glob.glob(f"{base_path}/*")

for dir_name in dir_names:
    
    if 'single_lift_cloth_3' not in dir_name:
        continue
    case_name = dir_name.split("/")[-1]
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")

    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)
    frame_len = split["frame_len"]

    # Find all stage directories
    stage_dirs = sorted(glob.glob(f"{prediction_dir}/{case_name}/stage_*"))
    if len(stage_dirs) == 0:
        print(f"No stage directories found for {case_name}, exiting.")
        continue

    print(f"Found {len(stage_dirs)} stage(s): {[s.split('/')[-1] for s in stage_dirs]}")

    for stage_dir in stage_dirs:
        stage_name = stage_dir.split("/")[-1]  # e.g., "stage_0"

        for i in range(3):
            # Process each camera view
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            video_path = f"{prediction_dir}/{case_name}/{i}_{stage_name}_integrate.mp4"
            video_writer = cv2.VideoWriter(
                video_path,
                fourcc,
                FPS,
                (width, height),
            )

            for frame_idx in range(frame_len):

                render_path = f"{prediction_dir}/{case_name}/{stage_name}/{i}/{frame_idx:05d}.png"
                origin_image_path = f"{base_path}/{case_name}/color/{i}/{frame_idx}.png"
                human_mask_image_path = (
                    f"{human_mask_path}/{case_name}/mask/{i}/0/{frame_idx}.png"
                )
                object_image_path = (
                    f"{object_mask_path}/{case_name}/mask/{i}/{frame_idx}.png"
                )

                render_img = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)
                origin_img = cv2.imread(origin_image_path)
                human_mask = cv2.imread(human_mask_image_path)
                human_mask = cv2.cvtColor(human_mask, cv2.COLOR_BGR2GRAY)
                human_mask = human_mask > 0
                object_mask = cv2.imread(object_image_path)
                object_mask = cv2.cvtColor(object_mask, cv2.COLOR_BGR2GRAY)
                object_mask = object_mask > 0

                final_image = origin_img.copy()
                render_mask = np.logical_and(
                    (render_img != 0).any(axis=2), render_img[:, :, 3] > 100
                )
                render_img[~render_mask, 3] = 0

                final_image[:, :, :] = alpha * final_image + (1 - alpha) * np.array(
                    [255, 255, 255], dtype=np.uint8
                )

                test_alpha = render_img[:, :, 3] / 255
                final_image[:, :, :] = render_img[:, :, :3] * test_alpha[
                    :, :, None
                ] + final_image * (1 - test_alpha[:, :, None])

                final_image[human_mask] = alpha * origin_img[human_mask] + (
                    1 - alpha
                ) * np.array([255, 255, 255], dtype=np.uint8)

                video_writer.write(final_image)

            video_writer.release()
            print(f"  Saved {video_path}")
