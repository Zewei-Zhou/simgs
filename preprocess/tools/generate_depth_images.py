import os
import sys
import torch
import numpy as np
import argparse
import math
import imageio
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import pipeline
from plyfile import PlyData, PlyElement
import torch.nn.functional as F
# Removed unused import 'cv2'

import multiprocessing
from functools import partial

# Set the multiprocessing start method
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Ensure the colmap_utils path is correct
sys.path.append(str(Path(__file__).resolve().parents[1]))
from colmap_utils.loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat, read_points3D_binary, storePly
from colmap_utils.utils import visualize_depth, visualize_normal

def fetchPly(path):
    """Reads point cloud positions from a PLY file."""
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    return positions

def gen_ply(bin_path, ply_path):
    """Generates a PLY file from COLMAP's .bin file if it doesn't exist."""
    if os.path.exists(ply_path):
        return
    print(f"Generating {ply_path} from {bin_path}...")
    xyz, rgb, _ = read_points3D_binary(bin_path)
    storePly(ply_path, xyz, rgb)
    print("PLY file generated successfully.")

def focal2fov(focal, pixels):
    """Calculates the field of view from focal length and pixel count."""
    return 2 * math.atan(pixels / (2 * focal))

def find_image_path(path, image_name):
    """Find image in gtimages or genimages subdirectories."""
    # First try gtimages
    gt_path = os.path.join(path, 'images', 'gtimages', image_name)
    if os.path.exists(gt_path):
        return gt_path
    
    # Then try genimages subdirectories
    genimages_dir = os.path.join(path, 'images', 'genimages')
    if os.path.exists(genimages_dir):
        for subfolder in os.listdir(genimages_dir):
            subfolder_path = os.path.join(genimages_dir, subfolder)
            if os.path.isdir(subfolder_path):
                gen_path = os.path.join(subfolder_path, image_name)
                if os.path.exists(gen_path):
                    return gen_path
    
    # Fallback: try old structure (images/ flat)
    flat_path = os.path.join(path, 'images', image_name)
    if os.path.exists(flat_path):
        return flat_path
    
    print(f"Warning: Could not find image {image_name} in gtimages or genimages")
    return None

def readColmapSceneInfo(path):
    """Reads camera intrinsics and extrinsics for the COLMAP scene."""
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    cam_infos = []
    not_found_count = 0
    
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        # The R and T matrices from COLMAP transform world points to camera points (world-to-camera)
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model in ["PINHOLE", "OPENCV"]:
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, f"Unsupported COLMAP camera model: {intr.model}"

        # Find image in gtimages or genimages subdirectories
        image_name = os.path.basename(extr.name)
        image_path = find_image_path(path, image_name)
        
        if image_path is None:
            not_found_count += 1
            continue  # Skip this camera if image not found
            
        K = np.array([[focal_length_x, 0, width/2], [0, focal_length_y, height/2], [0, 0, 1]])
        
        # Create world-to-camera transformation matrix
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T

        cam = {
            "uid": uid, 
            "image_path": image_path,
            "K": K,
            # Despite the name 'c2w', this is the world-to-camera matrix (extrinsics)
            "c2w": w2c, 
            "FovX": FovX, 
            "FovY": FovY, 
            "H": height, 
            "W": width,
            "image_name": image_name
        }
        cam_infos.append(cam)
    
    if not_found_count > 0:
        print(f"\nWarning: {not_found_count} images not found in gtimages or genimages directories")
    print(f"Successfully read {len(cam_infos)} camera poses.")
    return cam_infos

@torch.no_grad()
def process_image(cam, pts, pipe, device):
    """Processes a single image to generate a depth map aligned with SfM points.
    Uses the depth calculation logic from generate_depth.py.
    """
    K = torch.tensor(cam["K"], dtype=torch.float32, device=device)
    c2w = torch.tensor(cam["c2w"], dtype=torch.float32, device=device)
    H, W = cam["H"], cam["W"]

    # Transform world points to camera coordinates
    pts_tensor = torch.tensor(pts, dtype=torch.float32, device=device)
    pts_h = torch.cat([pts_tensor, torch.ones((pts_tensor.shape[0], 1), dtype=torch.float32, device=device)], dim=1)
    pts_cam = torch.mm(c2w, pts_h.T).T
    pts_cam = pts_cam[:, :3]
    
    # Project to image plane
    pts_cam = torch.mm(K, pts_cam.T).T
    depth = pts_cam[:, 2]
    pts_cam = pts_cam / (depth.unsqueeze(1) + 1e-9)
    
    # Filter valid points
    depth_mask = depth > 0
    vis_mask = (pts_cam[:, 0] >= 0) & (pts_cam[:, 0] < W) & (pts_cam[:, 1] >= 0) & (pts_cam[:, 1] < H)
    mask = torch.logical_and(depth_mask, vis_mask)
    
    pts_cam = pts_cam[mask]
    depth = depth[mask]
    
    # If there are no valid SfM points for this image, we cannot proceed with alignment.
    if pts_cam.shape[0] < 10:
        print(f"Warning: Less than 10 valid SfM points for image {cam['image_name']}. Skipping alignment.")
        # Return a zero depth map as a fallback
        return torch.zeros((H, W), dtype=torch.float32), {'depth': np.array([]), 'pts_cam': np.array([])}
    
    sfm_depth = {
        'depth': depth.cpu().numpy(),
        'pts_cam': pts_cam[:, :2].cpu().numpy()
    }

    # Run the depth estimation model
    image = Image.open(cam["image_path"])
    result = pipe(image)
    pred_dis = result['predicted_depth'].to(device).squeeze(0)
    
    # Resize if needed
    pred_H, pred_W = pred_dis.shape
    if pred_H != H or pred_W != W:
        pred_dis = F.interpolate(pred_dis[None, None, ...], (H, W), mode='bilinear', align_corners=False)[0, 0]
    
    # Normalize the predicted disparity
    pred_dis = pred_dis / pred_dis.max()
    
    # Sample predicted disparity at sparse points
    pred_dis_sparse = pred_dis[pts_cam[:, 1].long(), pts_cam[:, 0].long()]
    
    # Filter out zero values
    zero_mask = pred_dis_sparse != 0
    sfm_dis = 1.0 / depth
    
    pred_dis_sparse = pred_dis_sparse[zero_mask]
    sfm_dis = sfm_dis[zero_mask]

    # Calculate scale and offset using robust statistics
    t_colmap = torch.median(sfm_dis)
    s_colmap = torch.mean(torch.abs(sfm_dis - t_colmap))

    t_mono = torch.median(pred_dis_sparse)
    s_mono = torch.mean(torch.abs(pred_dis_sparse - t_mono))
    
    scale = s_colmap / s_mono
    offset = t_colmap - t_mono * scale

    # Apply scale and offset to the entire disparity map
    full_zero_mask = pred_dis != 0
    pred_dis[full_zero_mask] = pred_dis[full_zero_mask] * scale + offset

    # Return disparity as the depth output (to match generate_depth.py behavior)
    # The depth will be saved as-is
    return pred_dis, sfm_depth

def save_depth_as_png(output_path, image_name, pred_depth):
    """Saves the depth map as a 16-bit PNG file, scaled for Gaussian Splatting."""
    # Scale depth values and convert to uint16
    depth_scaled = pred_depth * 65536.0
    
    # Clip to the valid range for uint16
    depth_scaled = np.clip(depth_scaled, 0, 65535)
    depth_uint16 = depth_scaled.astype(np.uint16)
    
    # Save as PNG
    depth_filename = os.path.join(output_path, os.path.splitext(image_name)[0] + ".png")
    imageio.imwrite(depth_filename, depth_uint16)
    
    return depth_filename

def save_sfm_depth(output_path, image_name, sfm_depth):
    """Saves the SfM depth information as an NPZ file."""
    sfm_filename = os.path.join(output_path, os.path.splitext(image_name)[0] + ".npz")
    np.savez(sfm_filename, **sfm_depth)
    return sfm_filename

def save_visualizations(vis_depth_path, vis_normal_path, image_name, pred_depth, cam):
    """Saves depth and normal map visualizations."""
    # Convert numpy depth back to a torch tensor for the utility functions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_depth_tensor = torch.from_numpy(pred_depth).to(device)
    
    # Generate depth visualization
    vis_depth = visualize_depth(pred_depth_tensor)
    if vis_depth.dim() == 3:  # Ensure format is H, W, C
        vis_depth = vis_depth.permute(1, 2, 0)
    vis_depth_np = vis_depth.cpu().numpy()
    vis_depth_np = (vis_depth_np[:, :, :3] * 255).astype(np.uint8)
    
    # Generate normal visualization
    # Note: visualize_normal may require camera intrinsics in a specific format
    vis_normal_tensor = visualize_normal(pred_depth_tensor.cpu().numpy(), cam)
    vis_normal_np = vis_normal_tensor.cpu().numpy()
    if vis_normal_np.shape[2] > 3: # Handle potential alpha channel
        vis_normal_np = vis_normal_np[:, :, :3]
    vis_normal_np = (vis_normal_np * 255).astype(np.uint8)
    
    # Save visualization images
    depth_vis_file = os.path.join(vis_depth_path, image_name)
    normal_vis_file = os.path.join(vis_normal_path, image_name)
    
    imageio.imwrite(depth_vis_file, vis_depth_np)
    imageio.imwrite(normal_vis_file, vis_normal_np)
    
    return depth_vis_file, normal_vis_file

def worker_process(cam, args, pts, model_path, device_id):
    """The function executed by each worker process."""
    device = f"cuda:{device_id}" if torch.cuda.is_available() and device_id >= 0 else "cpu"
    
    # Initialize the model pipeline within each individual process
    pipe = pipeline(task="depth-estimation", model=model_path, device=device)
    
    # Process the image to get the aligned depth map
    pred_depth_tensor, sfm_depth = process_image(cam, pts, pipe, device)
    pred_depth = pred_depth_tensor.cpu().numpy()
    
    image_name = cam["image_name"]
    
    # Save the final depth map as a 16-bit PNG
    depth_path = os.path.join(args.path, "depths")
    save_depth_as_png(depth_path, image_name, pred_depth)
    
    # Save sparse SfM depth points if requested
    if args.save_sfm:
        sfm_path = os.path.join(args.path, "sfm_gt_depths")
        save_sfm_depth(sfm_path, image_name, sfm_depth)
    
    # Save visual representations of depth and normals if requested
    if args.save_vis:
        vis_depth_path = os.path.join(args.path, "vis_depths")
        vis_normal_path = os.path.join(args.path, "vis_normal")
        save_visualizations(vis_depth_path, vis_normal_path, image_name, pred_depth, cam)
    
    return image_name

@torch.no_grad()
def main_process(args):
    path = args.path
    
    # Read camera poses and intrinsics from COLMAP binary files
    cam_infos = readColmapSceneInfo(path)
    
    # Sort cameras by image name for consistent processing order
    # Handle both numeric and string filenames
    def sort_key(x):
        name = x['image_name'].split('.')[0]
        # Try to extract numeric part for proper sorting
        # For gtimages: "00001" -> 1
        # For genimages: "00010_forward_000000" -> extract frame number
        import re
        # Try to extract leading digits
        match = re.match(r'(\d+)', name)
        if match:
            return int(match.group(1))
        return name
    
    cam_infos = sorted(cam_infos, key=sort_key)
    
    # Generate and load the 3D point cloud from the sparse reconstruction
    ply_path = os.path.join(path, "sparse/0", "points3D.ply")
    bin_path = os.path.join(path, "sparse/0", "points3D.bin")
    gen_ply(bin_path, ply_path)
    pts = fetchPly(ply_path)
    
    print(f"Loaded {len(pts)} 3D points from the sparse reconstruction.")
    
    # Set depth estimation model
    model_path = args.model
    
    # Create all necessary output directories
    depth_out_path = os.path.join(path, "depths")
    os.makedirs(depth_out_path, exist_ok=True)
    
    if args.save_sfm:
        sfm_depth_out_path = os.path.join(path, "sfm_gt_depths")
        os.makedirs(sfm_depth_out_path, exist_ok=True)
    
    if args.save_vis:
        vis_depth_out_path = os.path.join(path, "vis_depths")
        vis_normal_out_path = os.path.join(path, "vis_normal")
        os.makedirs(vis_depth_out_path, exist_ok=True)
        os.makedirs(vis_normal_out_path, exist_ok=True)
    
    # Clear existing files if the user requests it
    if args.clear_existing:
        print("Clearing existing depth files...")
        os.system(f'rm -rf {depth_out_path}/*')
        if args.save_sfm:
            os.system(f'rm -rf {sfm_depth_out_path}/*')
        if args.save_vis:
            os.system(f'rm -rf {vis_depth_out_path}/*')
            os.system(f'rm -rf {vis_normal_out_path}/*')
    
    # Configure multiprocessing based on available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("⚠️ Warning: No GPU detected. Using CPU, which will be very slow.")
        num_processes = min(multiprocessing.cpu_count(), 4)
        device_ids = [-1] * num_processes
    else:
        num_processes = min(num_gpus, args.max_processes) if args.max_processes > 0 else num_gpus
        device_ids = list(range(num_gpus))
        print(f"Found {num_gpus} GPUs. Using {num_processes} processes.")
    
    # Create a list of tasks for the multiprocessing pool
    tasks = [(cam, args, pts, model_path, device_ids[i % len(device_ids)]) for i, cam in enumerate(cam_infos)]
    
    # Process all images in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        list(tqdm(
            pool.starmap(worker_process, tasks), 
            total=len(tasks), 
            desc="Generating Depth Maps"
        ))
    
    print("\n✅ All depth maps have been processed successfully!")
    print(f"   - Depth maps (16-bit PNG) saved to: {depth_out_path}")
    if args.save_sfm:
        print(f"   - SfM depth data saved to: {os.path.join(path, 'sfm_gt_depths')}")
    if args.save_vis:
        print(f"   - Depth visualizations saved to: {os.path.join(path, 'vis_depths')}")
        print(f"   - Normal visualizations saved to: {os.path.join(path, 'vis_normal')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate depth maps from COLMAP data for Gaussian Splatting.")
    parser.add_argument("path", type=str, help="Path to the COLMAP scene root (containing sparse/0 and images folders).")
    parser.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Large-hf", 
                       help="Depth estimation model from HuggingFace (default: depth-anything/Depth-Anything-V2-Large-hf)")
    parser.add_argument("--save_sfm", action="store_true", help="Save sparse SfM depth data as .npz files")
    parser.add_argument("--save_vis", action="store_true", help="Save colorized depth and normal map visualizations")
    parser.add_argument("--clear_existing", action="store_true", help="Clear all existing output files before processing")
    parser.add_argument("--max_processes", type=int, default=0, 
                       help="Maximum number of GPU processes to use (default: 0, which means use all available GPUs)")
    
    args = parser.parse_args()
    
    # For backward compatibility, enable sfm and vis saving by default if not specified.
    if not args.save_sfm and not args.save_vis:
        print("Note: Enabling visualizations and SfM depth saving by default. Use --no-save_sfm and --no-save_vis flags to disable.")
        args.save_sfm = True
        args.save_vis = True
    
    main_process(args)
