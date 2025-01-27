import numpy as np
import torch
import pytorch3d
import matplotlib.pyplot as plt
from starter.utils import get_device, get_points_renderer
import imageio
import os

def render_heart(image_size=256, num_samples=100, scale=1.0, angle=0, device=None):
    """
    Renders a 3D heart using parametric sampling with improved shape.
    """
    if device is None:
        device = get_device()

    # Sample points from parametric equations
    u = torch.linspace(0, 2 * np.pi, num_samples)
    v = torch.linspace(0, np.pi, num_samples)
    U, V = torch.meshgrid(u, v)

    # Improved parametric equations for a heart shape
    x = scale * (16 * torch.pow(torch.sin(U), 3)) / 13
    z = scale * (13 * torch.cos(V) - 5 * torch.cos(2*V) - 2 * torch.cos(3*V) - torch.cos(4*V)) / 13
    y = scale * (13 * torch.cos(U) - 5 * torch.cos(2*U) - 2 * torch.cos(3*U) - torch.cos(4*U)) / 13

    # Combine points
    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    
    # Create a more vibrant red to pink gradient
    color = torch.stack([
        torch.ones_like(U.flatten()),  # Red channel always 1
        0.2 + 0.3 * torch.sin(V.flatten()),  # Green channel
        0.4 + 0.3 * torch.sin(U.flatten())   # Blue channel
    ], dim=1)
    
    # Create point cloud
    heart_cloud = pytorch3d.structures.Pointclouds(
        points=[points], 
        features=[color]
    ).to(device)

    # Set up camera for rotation
    rot_angle = angle * np.pi / 180
    R = torch.tensor([
        [np.cos(rot_angle), -np.sin(rot_angle), 0],
        [np.sin(rot_angle), np.cos(rot_angle), 0],
        [0, 0, 1]
    ]).unsqueeze(0)
    T = torch.tensor([[0, 0, 3]])
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    # Render
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(heart_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()

def create_360_gif(output_path="results/heart2.gif", num_frames=60):
    """Creates a 360-degree rotation animation of the heart"""
    frames = []
    for i in range(num_frames):
        angle = i * (360.0 / num_frames)
        image = render_heart(angle=angle, num_samples=150)  # Increased sample density
        frames.append((image * 255).astype(np.uint8))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=30, loop=0)

if __name__ == "__main__":
    create_360_gif() 