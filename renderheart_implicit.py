import numpy as np
import torch
import pytorch3d
import matplotlib.pyplot as plt
import mcubes
from starter.utils import get_device, get_mesh_renderer
import imageio
import os

def render_heart_mesh(image_size=256, voxel_size=48, scale=1.0, angle=0, device=None):
    """
    Renders a heart mesh using the implicit function:
    (x² + 9/4*y² + z² - 1)³ - x²*z³ - 9/200*y²*z³ = 0
    """
    if device is None:
        device = get_device()

    # Create a voxel grid with reduced size
    min_value = -2
    max_value = 2
    x = torch.linspace(min_value, max_value, voxel_size)
    y = torch.linspace(min_value, max_value, voxel_size)
    z = torch.linspace(min_value, max_value, voxel_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Scale the coordinates
    X = X * scale
    Y = Y * scale
    Z = Z * scale

    # Compute the implicit function more efficiently
    XY_term = X*X + (9/4)*Y*Y
    Z_term = Z*Z
    voxels = torch.pow(XY_term + Z_term - 1, 3) - X*X * torch.pow(Z, 3) - (9/200) * Y*Y * torch.pow(Z, 3)
    
    # Convert to numpy and free memory
    voxels_np = voxels.numpy()
    del X, Y, Z, XY_term, Z_term, voxels
    torch.cuda.empty_cache()

    # Extract vertices and faces using marching cubes
    vertices, faces = mcubes.marching_cubes(voxels_np, isovalue=0)
    del voxels_np
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    
    # Normalize vertex coordinates to the desired range
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    
    # Create a gradient color from red to pink
    colors = torch.zeros_like(vertices)
    colors[:, 0] = 1.0  # Red channel always 1
    # Create heart-like gradient
    height_gradient = (vertices[:, 1] - vertices[:, 1].min()) / (vertices[:, 1].max() - vertices[:, 1].min())
    colors[:, 1] = 0.2 + 0.3 * height_gradient  # Green channel
    colors[:, 2] = 0.4 + 0.3 * height_gradient  # Blue channel
    textures = pytorch3d.renderer.TexturesVertex(colors.unsqueeze(0))

    # Create mesh
    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)

    # Set up camera for rotation around Y-axis
    rot_angle = angle * np.pi / 180
    R = torch.tensor([
        [np.cos(rot_angle), 0, -np.sin(rot_angle)],
        [0, 1, 0],
        [np.sin(rot_angle), 0, np.cos(rot_angle)]
    ]).unsqueeze(0)
    
    # Position camera to see the heart properly
    T = torch.tensor([[0, 0.5, 4]])
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    
    # Add multiple lights for better visualization
    lights = pytorch3d.renderer.PointLights(
        location=[[2, 2, -2], [-2, 2, -2], [0, -2, -2]],
        device=device
    )
    
    # Render
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()

def create_360_gif(output_path="results/heart_implicit.gif", num_frames=60):
    """Creates a 360-degree rotation animation of the heart mesh"""
    frames = []
    for i in range(num_frames):
        angle = i * (360.0 / num_frames)
        image = render_heart_mesh(angle=angle, scale=1.2, voxel_size=48)  # Lower resolution, adjusted scale
        frames.append((image * 255).astype(np.uint8))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=30)

if __name__ == "__main__":
    create_360_gif() 