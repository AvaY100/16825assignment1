import numpy as np
import torch
import pytorch3d
import matplotlib.pyplot as plt
from starter.utils import get_device, get_points_renderer
import imageio
import os

def render_torus(image_size=256, num_samples=100, R=1.0, r=0.3, angle=0, device=None):
    """
    Renders a torus using parametric sampling.
    R: distance from the center of the tube to the center of the torus
    r: radius of the tube
    angle: rotation angle for animation
    """
    if device is None:
        device = get_device()

    # Sample points from parametric equations
    u = torch.linspace(0, 2 * np.pi, num_samples)
    v = torch.linspace(0, 2 * np.pi, num_samples)
    U, V = torch.meshgrid(u, v)

    # Parametric equations of a torus
    x = (R + r * torch.cos(V)) * torch.cos(U)
    y = (R + r * torch.cos(V)) * torch.sin(U)
    z = r * torch.sin(V)

    # Combine points
    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    
    # Create a colorful texture based on position
    color = torch.stack([
        torch.sin(U).flatten(),  # Red channel
        torch.cos(V).flatten(),  # Green channel
        torch.sin(U + V).flatten()  # Blue channel
    ], dim=1)
    color = (color + 1) / 2  # Normalize to [0, 1]

    # Create point cloud
    torus_cloud = pytorch3d.structures.Pointclouds(
        points=[points], 
        features=[color]
    ).to(device)

    # Set up camera
    rot_angle = angle * np.pi / 180
    R = torch.tensor([
        [np.cos(rot_angle), 0, -np.sin(rot_angle)],
        [0, 1, 0],
        [np.sin(rot_angle), 0, np.cos(rot_angle)]
    ]).unsqueeze(0)
    T = torch.tensor([[0, 0, 3]])
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    # Render
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(torus_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()

def create_360_gif(output_path="results/torus.gif", num_frames=60):
    """Creates a 360-degree rotation animation of the torus"""
    frames = []
    for i in range(num_frames):
        angle = i * (360.0 / num_frames)
        image = render_torus(angle=angle)
        frames.append((image * 255).astype(np.uint8))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=30, loop=0)

if __name__ == "__main__":
    create_360_gif() 