import glob
import os
import json

case = "End2End_Reduction" # 'neural_spring_field' # "evospring" # End2End # End2End_Reduction
CONFIG_FILE=f"./configs/{case}"
# 0: train 1: local test 2: global
MODE=0
RESTART_EPOCH=-1
local_rank=1

# Load config file (key=value format)
config = {}
with open(CONFIG_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            key, value = line.split('=', 1)
            # Remove quotes if present
            value = value.strip().strip("'\"")
            config[key] = value

base_path = "./evomesh_optimization_outputs"
res_path = f"./res/{case}"

if not os.path.exists(res_path):
    os.makedirs(res_path)
if not os.path.exists(base_path):
    print(f"Base path {base_path} does not exist.")
    exit(1)

dir_names = os.listdir(base_path)
finished_cases = os.listdir(res_path)

# dir_names = ['double_lift_cloth_1', 'double_lift_cloth_3', 'rope_double_hand', 'single_lift_zebra', 'single_push_rope_4', 'single_lift_cloth_4', 'single_push_rope_1'] # small batch test
# dir_names = ['weird_package', 'single_lift_cloth_4']
# exclusive_cases = ['single_clift_cloth_3'] #, 'double_lift_cloth_1', 'double_stretch_sloth', 'single_lift_rope', 'double_lift_cloth_3', 'single_lift_cloth_3', 'rope_double_hand', 'double_lift_zebra', 'double_lift_sloth']
for idx, case_name in enumerate(dir_names):
    # case_name = 'single_clift_cloth_3' # DEBUG
    # if case_name in exclusive_cases:
    #     continue
    
    print(f"Running case: {case_name}")
    # if case_name in finished_cases:
    #     print(f"Case {case_name} already finished, skipping.")
    #     continue

    os.system(
        f"python src/spring_mass_main.py \
            -case {case} -space_dim {config['space_dim']} -local_rank {local_rank} \
            -n_train {config['n_train']} -n_valid {config['n_valid']} -n_test {config['n_test']} -time_len {config['time_len']} \
            -noise_level {config['noise_level']} \
            -multi_mesh_layer {config['multi_mesh_layer']} -consist_mesh {config['consist_mesh']} \
            -num_epochs {config['num_epochs']} -batch {config['batch']} -lr {config['lr']} -gamma {config['gamma']} \
            -restart_epoch {RESTART_EPOCH} -mp_time {config['MP_time']} \
            -data_dir {config['data_dir']} -dump_dir {config['dump_dir']} -mode {MODE} -object_case {case_name} --reduction {config['reduction']}"
    )

    # if idx == 5: 
    # break

