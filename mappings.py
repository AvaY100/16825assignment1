import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesVertex,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights
)
from pytorch3d.io import load_objs_as_meshes
import torchvision.transforms as T
from pytorch3d.utils import ico_sphere
import imageio
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create sphere
mesh = ico_sphere(3, device=device)
verts = mesh.verts_packed()
faces = mesh.faces_packed()

# Generate vertex colors instead of UV textures
def generate_vertex_colors(verts):
    """Generate colors for each vertex based on position"""
    colors = torch.zeros_like(verts)
    # Normalize vertex positions to [0, 1] range for coloring
    colors[:, 0] = (verts[:, 0] - verts[:, 0].min()) / (verts[:, 0].max() - verts[:, 0].min())
    colors[:, 1] = (verts[:, 1] - verts[:, 1].min()) / (verts[:, 1].max() - verts[:, 1].min())
    colors[:, 2] = 0.5 * torch.ones_like(verts[:, 2])
    return colors

# Create vertex colors and textures
vertex_colors = generate_vertex_colors(verts)
textures = TexturesVertex(vertex_colors.unsqueeze(0))

# Create mesh with vertex colors
mesh = Meshes(
    verts=[verts],
    faces=[faces],
    textures=textures
)

# Set up camera
cameras = PerspectiveCameras(
    device=device,
    R=torch.eye(3).unsqueeze(0),
    T=torch.tensor([[0, 0, 3]]),
)

# Set up rasterizer
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Set up lights
lights = PointLights(
    device=device,
    location=[[0, 0, -3]],
)

# Create renderer
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
)

# Render
image = renderer(mesh)

# Display result
plt.figure(figsize=(10, 10))
plt.imshow(image[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.title("Vertex Color Mapping")
plt.show()

# Function to apply displacement to vertices
def apply_displacement(verts, strength=0.1):
    """Apply sinusoidal displacement to vertices"""
    displacement = torch.sin(verts[:, 0] * 10) * torch.cos(verts[:, 1] * 10) * strength
    displaced_verts = verts.clone()
    displaced_verts[:, 2] += displacement
    return displaced_verts

# Create displaced mesh
verts_displaced = apply_displacement(verts)
mesh_displaced = Meshes(
    verts=[verts_displaced],
    faces=[faces],
    textures=textures
)

# Render displaced mesh
image_displaced = renderer(mesh_displaced)

# Display displaced result
plt.figure(figsize=(10, 10))
plt.imshow(image_displaced[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.title("Displaced Mesh")
plt.show()

# Create 360-degree animation
def render_360_animation(mesh, renderer, num_frames=36):
    images = []
    for i in range(num_frames):
        angle = 360.0 * i / num_frames
        R = torch.tensor([
            [np.cos(np.radians(angle)), 0, -np.sin(np.radians(angle))],
            [0, 1, 0],
            [np.sin(np.radians(angle)), 0, np.cos(np.radians(angle))]
        ], device=device).unsqueeze(0)
        
        cameras = PerspectiveCameras(
            device=device,
            R=R,
            T=torch.tensor([[0, 0, 3]], device=device),
        )
        
        renderer.rasterizer.cameras = cameras
        renderer.shader.cameras = cameras
        
        image = renderer(mesh)
        image_np = (image[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        images.append(image_np)
    return images

# Render and save 360-degree animations
# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Save static images
plt.imsave('results/vertex_color_mapping.png', image[0, ..., :3].cpu().numpy())
plt.imsave('results/displaced_mesh.png', image_displaced[0, ..., :3].cpu().numpy())

# Generate and save 360-degree animations
print("Generating 360-degree animation for basic mesh...")
images_360 = render_360_animation(mesh, renderer)
imageio.mimsave('results/vertex_color_360.gif', images_360, duration=1000/30)  # 30 FPS

print("Generating 360-degree animation for displaced mesh...")
images_360_displaced = render_360_animation(mesh_displaced, renderer)
imageio.mimsave('results/displaced_360.gif', images_360_displaced, duration=1000/30)  # 30 FPS

print("All results have been saved to the results directory.")
