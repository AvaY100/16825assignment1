import numpy as np
import torch
import pytorch3d
import matplotlib.pyplot as plt
import mcubes
from starter.utils import get_device, get_mesh_renderer
import imageio
import os

def render_torus_mesh(image_size=256, voxel_size=48, R=2.0, r=1.0, angle=0, device=None):
    """
    Renders a torus mesh using implicit function and marching cubes.
    R: major radius (distance from center of tube to center of torus)
    r: minor radius (radius of the tube)
    """
    if device is None:
        device = get_device()

    # Create a voxel grid with reduced size
    min_value = -4
    max_value = 4
    x = torch.linspace(min_value, max_value, voxel_size)
    y = torch.linspace(min_value, max_value, voxel_size)
    z = torch.linspace(min_value, max_value, voxel_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Compute the implicit function
    voxels = (X*X + Y*Y + Z*Z + R*R - r*r) * (X*X + Y*Y + Z*Z + R*R - r*r) - 4*R*R*(X*X + Y*Y)
    
    # Convert to numpy and free some memory
    voxels_np = voxels.numpy()
    del X, Y, Z, voxels
    torch.cuda.empty_cache()  # Clear GPU memory if any

    # Extract vertices and faces using marching cubes
    vertices, faces = mcubes.marching_cubes(voxels_np, isovalue=0)
    del voxels_np
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    
    # Normalize vertex coordinates to the desired range
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    
    # Create colors based on normalized position
    colors = torch.zeros_like(vertices)
    colors[:, 0] = torch.sin(vertices[:, 0] * np.pi) * 0.5 + 0.5  # Red channel
    colors[:, 1] = torch.sin(vertices[:, 1] * np.pi) * 0.5 + 0.5  # Green channel
    colors[:, 2] = torch.sin(vertices[:, 2] * np.pi) * 0.5 + 0.5  # Blue channel
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
    
    # Move camera up slightly and back
    T = torch.tensor([[0, 0.5, 6]])
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

def create_360_gif(output_path="results/torus_implicit.gif", num_frames=60):
    """Creates a 360-degree rotation animation of the torus mesh"""
    frames = []
    for i in range(num_frames):
        angle = i * (360.0 / num_frames)
        image = render_torus_mesh(angle=angle, voxel_size=48, R=2.0, r=0.8)  # Lower resolution, adjusted parameters
        frames.append((image * 255).astype(np.uint8))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=30)

if __name__ == "__main__":
    create_360_gif() 