import os
# 强制开启软件渲染，绕过显卡驱动
# 【核心修改 1】必须在 import open3d 之前设置环境变量
# 2. 告诉 EGL/Mesa: "不要找显卡，直接用 CPU 算"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["GALLIUM_DRIVER"] = "llvmpipe"

# 3. 如果是 OffscreenRenderer，无头模式仍需保留
os.environ["EGL_PLATFORM"] = "surfaceless"
import h5py
import numpy as np

def h5_to_dict(h5_obj):
    """
    递归将 h5 对象 (File 或 Group) 转换为嵌套字典。
    """
    result = {}
    
    # 遍历当前组下的所有对象 (key: 名称, item: 对象)
    for key, item in h5_obj.items():
        if isinstance(item, h5py.Dataset):
            # 如果是数据集，读取数据
            # item[()] 获取所有数据为 numpy array
            data = item[()]
            
            # 【可选】如果你希望结果里完全没有 numpy 类型，可以转为 list
            # if isinstance(data, np.ndarray):
            #     data = data.tolist()
            
            # 处理 bytes 类型字符串 (h5py 读取字符串常为 bytes)
            if isinstance(data, bytes):
                data = data.decode('utf-8')
                
            result[key] = data
            
        elif isinstance(item, h5py.Group):
            # 如果是组，递归调用自身
            result[key] = h5_to_dict(item)
            
    return result

# --- 使用示例 ---
file_path = '/mnt/pool1/cxy/phystwin-v2/cylinder/outputs_test/1.h5'  # 请替换为你的文件名

try:
    with h5py.File(file_path, 'r') as f:
        # 调用函数
        data_dict = h5_to_dict(f)
        
        # 打印结果查看 (打印顶级 keys)
        print("Keys:", data_dict.keys())
        
        # 此时 data_dict 就是一个标准的 Python 字典了
        # 例如访问: data_dict['group1']['dataset1']
        
except Exception as e:
    print(f"发生错误: {e}")

import pickle
import pandas as pd  # 如果处理 DataFrame 需要用到

def pkl_to_dict(file_path):
    """
    读取 pkl 文件并根据内容类型尝试转换为 dict
    """
    try:
        # 'rb' 模式非常重要，因为 pickle 是二进制格式
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        print(f"原始数据类型: {type(data)}")

        # --- 情况 1: 数据本身就是字典 ---
        if isinstance(data, dict):
            return data

        # --- 情况 2: 数据是 Pandas DataFrame ---
        # 很多 pkl 文件其实是 pandas 保存的
        elif isinstance(data, pd.DataFrame):
            print("检测到 Pandas DataFrame，正在转换为 dict...")
            # 'records' 格式通常最像列表字典，也可以用 'index', 'columns' 等
            return data.to_dict(orient='records') 

        # --- 情况 3: 数据是自定义类的对象 (Class Instance) ---
        # 尝试获取对象的属性字典
        elif hasattr(data, '__dict__'):
            print("检测到类对象，正在提取属性为 dict...")
            return vars(data)

        # --- 情况 4: 数据是列表或元组 ---
        # 强行转为字典，key 为索引
        elif isinstance(data, (list, tuple)):
            print("检测到列表/元组，正在转换为带索引的 dict...")
            return {i: item for i, item in enumerate(data)}

        # --- 其他情况 ---
        else:
            print("警告: 数据类型难以直接转换，已包装在 'content' 键中")
            return {"content": data}

    except FileNotFoundError:
        return {"error": "找不到文件"}
    except Exception as e:
        return {"error": f"发生错误: {e}"}

# --- 使用示例 ---
pcd_path = '/mnt/pool1/cxy/phystwin-v2/PhysTwin/data/different_types/double_lift_zebra/final_data.pkl'  # 替换你的文件名
connect_path = '/mnt/pool1/cxy/phystwin-v2/PhysTwin/experiments_optimization/double_lift_zebra/optimal_params.pkl'
pth_path = '/mnt/pool1/cxy/phystwin-v2/PhysTwin/experiments/double_lift_zebra/train/best_199.pth'
pcd_dict = pkl_to_dict(pcd_path)
connect_dict = pkl_to_dict(connect_path)
import pyvista as pv
import numpy as np

# 假设 points 是 (N, 3) 的 numpy 数组
points = pcd_dict['object_points'][25]  # 取第一个时间步的点云数据

surface_points = pcd_dict['surface_points']

interior_points = pcd_dict['interior_points']

# 1. 创建点云对象
cloud = pv.PolyData(points)

# 2. 初始化绘图器 (关键: off_screen=True)
# window_size 决定分辨率
plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])

# 3. 添加点云
# color: 点的颜色, point_size: 点的大小
plotter.add_mesh(cloud, color='green', point_size=5.0, render_points_as_spheres=True)

# (B) 绘制 Surface Points (表面先验) - 蓝色，半透明
if surface_points is not None:
    plotter.add_mesh(
        pv.PolyData(surface_points), 
        color='red', 
        point_size=3.0, 
        opacity=0.4,          # 设置透明度，防止挡住里面的点
        render_points_as_spheres=True,
        label='Surface Points'
    )

# (C) 绘制 Interior Points (内部填充) - 绿色，小点，很淡
if interior_points is not None:
    plotter.add_mesh(
        pv.PolyData(interior_points), 
        color='purple',      # 也可以用 Hex 颜色代码
        point_size=2.0, 
        opacity=0.2, 
        render_points_as_spheres=True,
        label='Interior Points'
    )

# 4. 设置背景
plotter.set_background('black')

# 5. 设置初始相机视角 (自动居中)
plotter.camera_position = 'xy' 
plotter.camera.azimuth = 45
plotter.camera.elevation = 30

# 6. 打开视频流
plotter.open_movie('pyvista_render.mp4', framerate=30)

# 7. 开始渲染循环 (替代 orbit_on_path)
print("开始 PyVista 渲染...")
plotter.show(auto_close=False)  # 必须先 show，建立窗口上下文

# 使用最稳健的手动旋转方式
for i in range(360):
    # 每一帧写入视频
    plotter.write_frame()
    
    # 每次循环让相机方位角增加 1 度 (相当于绕 Z 轴旋转)
    plotter.camera.azimuth += 1

plotter.close()
print("视频保存成功！")
import torch
pth_content = torch.load(pth_path, map_location='cpu')

# 打印前几个 key 查看结果
if pcd_dict:
    print("\n转换后的字典 Keys (前5个):")
    print(list(pcd_dict.keys())[:5])
    
    # print(result_dict) # 如果数据量小，可以取消注释打印全部