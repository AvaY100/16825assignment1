import os
import sys
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d
import pytorch3d.io
from pytorch3d.vis.plotly_vis import plot_scene
from tqdm.auto import tqdm

import starter.utils
import imageio

# This should print True if you are using your GPU
print("Using GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

from starter.utils import load_cow_mesh
def color_cow_front_to_back():
    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh("data/cow.obj")
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    
    
    # 获取z坐标的最大值和最小值
    # import pdb; pdb.set_trace()
    z_min = vertices[0, :, 2].min()
    z_max = vertices[0, :, 2].max()
    
    # 定义前后两种颜色
    color1 = torch.tensor([1.0, 0.0, 0.0], device=device)  # 前面颜色（红色）
    color2 = torch.tensor([0.0, 0.0, 1.0], device=device)  # 后面颜色（蓝色）
    
    # 计算每个顶点的颜色插值权重
    weights = (vertices[0, :, 2] - z_min) / (z_max - z_min)  # Shape: (N_v,)

    # import pdb; pdb.set_trace()

    colors = (1 - weights[:, None]) * color1 + weights[:, None] * color2  # Shape: (N_v, 3)
    colors = colors.unsqueeze(0)  # Add batch dimension -> (1, N_v, 3)
    # print(colors.shape)
    # 创建新的网格，带有渐变颜色
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(verts_features=colors),
    )
    
    return mesh

# 创建渐变色奶牛
mesh = color_cow_front_to_back()

# 渲染360度视图
num_views = 36
R, T = pytorch3d.renderer.look_at_view_transform(
    dist=3,
    elev=0,
    azim=np.linspace(0, 360, num_views, endpoint=False),
)

cameras = pytorch3d.renderer.FoVPerspectiveCameras(
    R=R,
    T=T,
    device=device
)

# 渲染
image_size = 512

raster_settings = pytorch3d.renderer.RasterizationSettings(image_size=image_size)
rasterizer = pytorch3d.renderer.MeshRasterizer(
    raster_settings=raster_settings,
)
shader = pytorch3d.renderer.HardPhongShader(device=device)
renderer = pytorch3d.renderer.MeshRenderer(
    rasterizer=rasterizer,
    shader=shader,
)

lights = pytorch3d.renderer.PointLights(location=[[3, 3, -3]], device=device)

images = renderer(mesh.extend(num_views), cameras=cameras, lights=lights)

# 保存GIF
my_images = []
for image in images:
    img = (image.cpu().numpy() * 255).astype(np.uint8)[:, :, :3]
    my_images.append(img)



imageio.mimsave('results/gradient_cow.gif', my_images, duration=1000//15, loop=0)