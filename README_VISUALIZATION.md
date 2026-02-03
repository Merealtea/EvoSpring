# Spring Mass Simulation Visualization

这个文档说明如何使用PyVista可视化弹簧质点系统的仿真结果。

## 功能特性

- 🎥 生成3D可视化视频
- 🎨 不同颜色区分controller points和object points
- 📊 同时显示仿真结果和真值(ground truth)
- 💾 保存仿真数据到HDF5文件

## 可视化说明

- **绿色点**: 仿真的物体点 (Simulated Object Points)
- **蓝色点**: 控制点 (Controller Points)
- **红色半透明点**: 真值物体点 (Ground Truth Object Points)

## 使用方法

### 1. 基本用法

```bash
python visualize_spring_mass.py \
    --data_dir /path/to/your/data \
    --dump_dir /path/to/model/checkpoints \
    --mech_info_path /path/to/best_mech_info.pth \
    --object_case your_object_case \
    --output_video output_visualization.mp4
```

### 2. 完整示例

假设你的数据结构如下：
```
EvoMesh/
├── data/
│   └── cylinder/
│       ├── final_data.pkl
│       └── ...
├── experiments/
│   └── cylinder/
│       ├── spring_mech_info/
│       │   └── best_199.pth
│       └── ...
```

运行命令：
```bash
cd /mnt/pool1/cxy/phystwin-v2/EvoMesh

python visualize_spring_mass.py \
    --data_dir data/cylinder \
    --dump_dir experiments/cylinder \
    --mech_info_path experiments/cylinder/spring_mech_info/best_199.pth \
    --object_case cylinder \
    --output_video cylinder_visualization.mp4 \
    --hidden_dim 128 \
    --multi_mesh_layer 3 \
    --mp_time 3 \
    --train_frame 100
```

### 3. 参数说明

#### 必需参数
- `--data_dir`: 数据目录路径
- `--dump_dir`: 模型检查点目录
- `--mech_info_path`: best_mech_info.pth文件的完整路径
- `--object_case`: 物体案例名称 (如: cylinder, cloth, package等)

#### 模型参数 (需要与训练时一致)
- `--hidden_dim`: 隐藏层维度 (默认: 128)
- `--multi_mesh_layer`: 多层网格层数 (默认: 3)
- `--mp_time`: 消息传递次数 (默认: 3)
- `--hidden_depth`: MLP隐藏层深度 (默认: 2)
- `--pre_layer_num`: 预处理层数 (默认: 2)
- `--bottom_layer_num`: 底层层数 (默认: 2)

#### 可选参数
- `--output_video`: 输出视频路径 (默认: simulation_visualization.mp4)
- `--train_frame`: 仿真帧数 (默认: 100)
- `--space_dim`: 空间维度 (默认: 3)

## 输出文件

运行后会生成两个文件：

1. **视频文件** (`.mp4`):
   - 分辨率: 1920x1080
   - 帧率: 30 FPS
   - 包含所有仿真帧的3D可视化

2. **数据文件** (`.h5`):
   - `object_points`: 仿真的物体点序列 (shape: [num_frames, num_points, 3])
   - `controller_points`: 控制点序列 (shape: [num_frames, num_ctrl_points, 3])
   - `gt_object_points`: 真值物体点序列 (shape: [num_frames, num_points, 3])

## 在代码中直接调用

你也可以在Python代码中直接调用可视化函数：

```python
from spring_mass_trainer import SpringMassTrainer

# 初始化trainer (需要正确的args)
trainer = SpringMassTrainer(args, device)

# 运行可视化
obj_seq, ctrl_seq, gt_seq = trainer.visualize_simulation(
    mech_info_path='path/to/best_mech_info.pth',
    output_video_path='output.mp4'
)

print(f"Generated {len(obj_seq)} frames")
```

## 注意事项

1. **环境要求**: 确保已安装PyVista和h5py
   ```bash
   pip install pyvista h5py
   ```

2. **GPU内存**: 如果仿真帧数很多，可能需要较大的GPU内存

3. **渲染模式**: 使用离屏渲染(off_screen=True)，不需要显示器

4. **配置文件**: 会自动根据object_case加载对应的配置文件:
   - cloth/package → `configs/phystwin_configs/cloth.yaml`
   - 其他 → `configs/phystwin_configs/real.yaml`

## 故障排除

### 问题1: 找不到mech_info文件
```
解决: 确保mech_info_path指向正确的.pth文件，通常在 dump_dir/spring_mech_info/ 目录下
```

### 问题2: CUDA out of memory
```
解决: 减少train_frame参数，或使用更小的batch
```

### 问题3: PyVista渲染错误
```
解决: 确保设置了正确的环境变量:
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
```

## 示例输出

```
Using device: cuda:0
Data directory: data/cylinder
Model directory: experiments/cylinder
Mech info path: experiments/cylinder/spring_mech_info/best_199.pth
Output video: cylinder_visualization.mp4

Initializing trainer...
Starting visualization...
Loading mech_info from: experiments/cylinder/spring_mech_info/best_199.pth
Running pure inference simulation for 100 frames...
100%|████████████████████| 100/100 [00:15<00:00,  6.45it/s]
Simulation completed. Creating PyVista visualization...
Rendering frames: 100%|████████████████████| 100/100 [00:30<00:00,  3.33it/s]
Video saved to: cylinder_visualization.mp4
Saving simulation data to: cylinder_visualization.h5
Visualization complete!

✓ Visualization completed successfully!
  - Video saved to: cylinder_visualization.mp4
  - Data saved to: cylinder_visualization.h5
  - Total frames: 100
  - Object points per frame: 512
  - Controller points per frame: 128
```
