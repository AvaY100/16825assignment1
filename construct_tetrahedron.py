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

import argparse

# This should print True if you are using your GPU
print("Using GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

    

def construct_tetrahedron(multiple_color=False):
    # 定义4个顶点
    vertices = torch.tensor([
        [0.0, 0.0, 0.0],    # 底部顶点 0
        [1.0, 0.0, 0.0],    # 底部顶点 1
        [0.5, 1.0, 0.0],    # 底部顶点 2
        [0.5, 0.5, 1.0],    # 顶部顶点 3
    ])

    # 定义4个面（每个面是一个三角形）
    faces = torch.tensor([
        [0, 1, 2],  # 底面
        [0, 1, 3],  # 侧面1
        [1, 2, 3],  # 侧面2
        [0, 2, 3],  # 侧面3
    ])

    # SingleColor
    if not multiple_color:
        # 创建一个单一颜色的纹理（这里用蓝色）
        verts_rgb = torch.ones_like(vertices)[None]  # (1, V, 3)
        verts_rgb = verts_rgb * torch.tensor([0.7, 0.7, 1.0])  # 蓝色

        # 创建网格
        tetra_mesh = pytorch3d.structures.Meshes(
            verts=[vertices],
            faces=[faces],
            textures=pytorch3d.renderer.TexturesVertex(verts_rgb)
        )
    else:
        # multiple_color
        # 定义每个面的颜色
        face_colors = torch.tensor([
            [1.0, 0.0, 0.0],  # 红色 - 底面
            [0.0, 1.0, 0.0],  # 绿色 - 侧面1
            [0.0, 0.0, 1.0],  # 蓝色 - 侧面2
            [1.0, 1.0, 0.0],  # 黄色 - 侧面3
        ])

        # 创建网格
        tetra_mesh = pytorch3d.structures.Meshes(
            verts=[vertices],
            faces=[faces],
            textures=pytorch3d.renderer.TexturesVertex(face_colors[None, ...])
        )

    return tetra_mesh

# 创建四面体
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiple_color", type=bool, default=False)
    args = parser.parse_args()
    multiple_color = args.multiple_color

    mesh = construct_tetrahedron(multiple_color=multiple_color)

    # 设置36个不同的视角（每10度一个）
    num_views = 72
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=3,
        elev=30,  # 从上方30度看
        azim=np.linspace(0, 360, num_views, endpoint=False),
    )

    # 创建相机
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

    lights = pytorch3d.renderer.PointLights(location=[[1.0, 1.0, 1.5]], device=device)

    images = renderer(mesh.extend(num_views), cameras=cameras, lights=lights)

    # 转换并保存为GIF
    my_images = []
    for image in images:
        img = (image.cpu().numpy() * 255).astype(np.uint8)[:, :, :3]
        my_images.append(img)

    imageio.mimsave('results/tetrahedron.gif', my_images, duration=1000//15, loop=0)
