import torch
import numpy as np
import matplotlib.pyplot as plt
from starter.render_generic import load_rgbd_data
from starter.utils import get_device, get_points_renderer, unproject_depth_image
import pytorch3d
from pytorch3d.structures import Pointclouds
from PIL import Image
import os

def create_point_cloud_gif(point_cloud, output_path, image_size=256, num_frames=20, distance=6):
    device = get_device()
    renderer = get_points_renderer(image_size=image_size, background_color=(1, 1, 1))
    
    images = []
    angles = torch.linspace(0, 360, num_frames)
    
    for angle in angles:
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=distance, 
            elev=-5,
            azim=angle,
            at=((0, 0, 0),),
            up=((0, -1, 0),),
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rend = (rend * 255).astype(np.uint8)
        images.append(Image.fromarray(rend))
    
    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0
    )

def main():
    device = get_device()
    
    # Load RGBD data
    data = load_rgbd_data()
    
    # Convert numpy arrays to torch tensors
    rgb1 = torch.from_numpy(data["rgb1"]).to(device)
    mask1 = torch.from_numpy(data["mask1"]).to(device)
    depth1 = torch.from_numpy(data["depth1"]).to(device)
    
    rgb2 = torch.from_numpy(data["rgb2"]).to(device)
    mask2 = torch.from_numpy(data["mask2"]).to(device)
    depth2 = torch.from_numpy(data["depth2"]).to(device)
    
    # Create point clouds for both images
    points1, colors1 = unproject_depth_image(
        rgb1, mask1, depth1, data["cameras1"]
    )
    points2, colors2 = unproject_depth_image(
        rgb2, mask2, depth2, data["cameras2"]
    )
    
    # Create individual point clouds
    point_cloud1 = Pointclouds(
        points=points1.unsqueeze(0).to(device),
        features=colors1.unsqueeze(0).to(device)
    )
    point_cloud2 = Pointclouds(
        points=points2.unsqueeze(0).to(device),
        features=colors2.unsqueeze(0).to(device)
    )
    
    # Create combined point cloud
    combined_points = torch.cat([points1, points2], dim=0)
    combined_colors = torch.cat([colors1, colors2], dim=0)
    combined_cloud = Pointclouds(
        points=combined_points.unsqueeze(0).to(device),
        features=combined_colors.unsqueeze(0).to(device)
    )
    
    # Generate and save GIFs
    create_point_cloud_gif(point_cloud1, "results/point_cloud1.gif")
    create_point_cloud_gif(point_cloud2, "results/point_cloud2.gif")
    create_point_cloud_gif(combined_cloud, "results/point_cloud_combined.gif")

if __name__ == "__main__":
    main()
