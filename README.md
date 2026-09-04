# Physpring

> Physpring: Differentiable Spring-Mass Simulation for Estimating Mechanical Properties of Deformable Objects.

Physpring is a **fully differentiable framework** that learns the **mechanical properties** (spring stiffness and damping) of deformable objects such as cloth and ropes. It combines a graph neural network (GNN) built on the [EvoMesh](https://arxiv.org/abs/2410.03779) backbone with the [NVIDIA Warp](https://github.com/NVIDIA/warp) differentiable simulator (`SpringMassSystemWarp`) to identify per-spring physical parameters from observed point-cloud / mesh trajectories.


## Environment Requirements

- Ubuntu 22.04
- Python 3.10
- CUDA 12.8

Install the environment with the following commands:

```bash
conda create -n physpring python=3.10 -y
conda activate physpring
bash env_install/env_install.sh
```

`env_install/env_install.sh` installs the CUDA 12.8 toolkit, PyTorch 2.8 (cu128), NVIDIA Warp, and compiles the Gaussian Splatting CUDA extensions.

## Datasets Preparation

Follow the instruction from PhysTwin to download the datasets and struture them into the project's root folder.

- [data](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/data.zip): this includes the original data for different cases and the processed data for quick run. The different case_name can be found under `different_types` folder.
- [experiments_optimization](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/experiments_optimization.zip): results of our first-stage zero-order optimization.
- [experiments](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/experiments.zip): results of our second-order optimization.
- [gaussian_output](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/gaussian_output.zip): results of our static gaussian appearance.

The final folder struture is like this

```
PhySpring/
├── ...
├── data/
├── experiments/
├── experiments_optimization/
└── gaussian_output/
└── ...
```

Then transfer data for the PhySpring model input format in 'evomesh_optimization_outputs' folder
```bash
python ./tools/phystwin_2_evomesh.py

```

## Training

Train all scenes and the result will be stored in './res/End2End_Reduction'

```bash
python script_train.py --local_rank 0
```

You can assign the device by change the parameters local rank 

## Evaluating Spring Mass Chamfer Loss and Tracking Loss


```bash
bash evaluate_global.sh
```

The chamfer loss and tracking loss in the './results/final_global_results.csv' and './results/final_global_track.csv'

## Evaluating Rendering Loss

Render predicted dynamics and convert frames to video:

```bash
# Use LBS to render the dynamic videos (The final videos in ./gaussian_output_dynamic folder)
bash gs_run_simulate_multi_stage.sh
python export_render_eval_data_multi_stage.py
# Get the quantative results
bash evaluate_global.sh

# Get the qualitative results
bash gs_run_simulator_white_multi_stage.sh
python visualize_render_results_multi_stage.py
```
## Real2sim Deployment

You can export the phystwin assets into real2sim simulation for robot grasp and place.

Follow the readme in [Real2sim-extent repo](https://github.com/Merealtea/Real2sim-eval-extend)

## Citation

```
@inproceedings{physpring,
    title={Physpring: Differentiable Spring-Mass Simulation for Estimating Mechanical Properties of Deformable Objects},
    author={TODO},
    booktitle={TODO},
    year={TODO},
    url={TODO},
}
```

## Credits

The codes refer to the implementation of [EvoMesh](https://github.com/hbell99/evo-mesh) and [PhysTwin](https://github.com/jianghanxiao/phystwin). Thanks to the authors!
