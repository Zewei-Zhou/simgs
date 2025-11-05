import os
import glob
import sys
import re
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from colorama import Fore, init, Style
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
try:
    import laspy
except:
    print("No laspy")
from utils.graphics_utils import BasicPointCloud
import concurrent.futures

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    CX: np.array
    CY: np.array
    image: np.array
    mask: np.array
    sky_mask: np.array
    depth: np.array
    depth_params: dict
    image_path: str
    image_name: str
    width: int
    height: int
    validity_mask: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def read_las_file(path):
    las = laspy.read(path)
    positions = np.vstack((las.x, las.y, las.z)).transpose()
    try:
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.random.rand(positions.shape[0], positions.shape[1])

    return positions, colors, normals

def read_multiple_las_files(paths, ply_path):
    all_positions = []
    all_colors = []
    all_normals = []

    for path in paths:
        positions, colors, normals = read_las_file(path)
        all_positions.append(positions)
        all_colors.append(colors)
        all_normals.append(normals)

    all_positions = np.vstack(all_positions)
    all_colors = np.vstack(all_colors)
    all_normals = np.vstack(all_normals)

    print("Saving point cloud to .ply file...")
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    elements = np.empty(all_positions.shape[0], dtype=dtype)
    attributes = np.concatenate((all_positions, all_normals, all_colors), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)

    return BasicPointCloud(points=all_positions, colors=all_colors, normals=all_normals)

def fetchPly(path):
    print("read ply file from {}".format(path))
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        label_ids = np.array(vertices['label']).T
    except:
        label_ids = np.zeros_like(vertices['x'])

    return BasicPointCloud(points=positions, colors=colors, normals=normals, label_ids=label_ids)

def storePly(path, xyz, rgb, label_ids):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('label', 'u1')]
    
    normals = np.zeros_like(xyz)
    label_ids = label_ids.reshape(-1, 1)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, label_ids), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def scan_all_images_in_folders(images_folder):
    """
    Scan all images from gtimages and genimages folders.
    Returns a dict mapping image_name -> full_image_path
    
    Structure:
    - images/gtimages/*.jpg
    - images/genimages/1/*.jpg, 2/*.jpg, 3/*.jpg, etc.
    """
    image_map = {}
    
    # Check flat structure first (legacy support)
    if os.path.exists(images_folder):
        for item in os.listdir(images_folder):
            item_path = os.path.join(images_folder, item)
            if os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Legacy flat structure - use this and return immediately
                image_map[item] = item_path
        
        if image_map:
            return image_map
    
    # Scan gtimages folder
    gt_folder = os.path.join(images_folder, "gtimages")
    if os.path.exists(gt_folder) and os.path.isdir(gt_folder):
        for item in os.listdir(gt_folder):
            if item.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_map[item] = os.path.join(gt_folder, item)
    
    # Scan genimages subfolders
    gen_seq_folder = os.path.join(images_folder, "genimages")
    if os.path.exists(gen_seq_folder) and os.path.isdir(gen_seq_folder):
        try:
            subfolders = [f for f in os.listdir(gen_seq_folder) 
                         if os.path.isdir(os.path.join(gen_seq_folder, f))]
            # Sort subfolders numerically
            try:
                subfolders = sorted(subfolders, key=lambda x: int(x) if x.isdigit() else x)
            except:
                subfolders = sorted(subfolders)
            
            for subfolder in subfolders:
                subfolder_path = os.path.join(gen_seq_folder, subfolder)
                for item in os.listdir(subfolder_path):
                    if item.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Store with full path
                        image_map[item] = os.path.join(subfolder_path, item)
        except Exception as e:
            print(f"Warning: Error scanning genimages folder: {e}")
    
    return image_map

def get_genimages_mask_path(image_path, images_folder):
    """
    Get the mask path for a genimages image.
    If image is in images/genimages/1/*.jpg, mask is in images/genimages/1_mask/*.jpg
    If image is in images/genimages/2/*.jpg, mask is in images/genimages/2_mask/*.jpg
    Returns None if not a genimages image or mask doesn't exist.
    """
    # Check if this is a genimages image
    if "genimages" not in image_path:
        return None
    
    # Extract the sequence number (1, 2, 3, etc.)
    try:
        parts = image_path.split(os.sep)
        gen_seq_idx = parts.index("genimages")
        if gen_seq_idx + 1 < len(parts):
            seq_num = parts[gen_seq_idx + 1]
            image_name = os.path.basename(image_path)
            
            # Construct mask path: images_folder/genimages/seq_num_mask/image_name
            gen_seq_folder = os.path.join(images_folder, "genimages")
            mask_folder = os.path.join(gen_seq_folder, f"{seq_num}_mask")
            mask_path = os.path.join(mask_folder, image_name)
            
            if os.path.exists(mask_path):
                return mask_path
    except (ValueError, IndexError):
        pass
    
    return None
    
def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, masks_folder, depths_folder, sky_masks_folder=None):
    cam_infos = []
    
    # First, scan all available images in gtimages and genimages folders
    print(f"Scanning images in {images_folder}...")
    image_map = scan_all_images_in_folders(images_folder)
    print(f"Found {len(image_map)} images in total")
    
    # Debug: print sample of image_map keys and COLMAP names
    if len(image_map) > 0:
        sample_keys = list(image_map.keys())[:3]
        print(f"Sample image_map keys: {sample_keys}")
    
    sample_colmap_names = [cam_extrinsics[key].name for key in list(cam_extrinsics.keys())[:3]]
    print(f"Sample COLMAP names: {sample_colmap_names}")
    
    def process_frame(idx, key):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            CX, CY = intr.params[1], intr.params[2]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE" or intr.model=="OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            CX, CY = intr.params[2], intr.params[3]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        # Look up image path from the image_map using COLMAP image name
        # Try direct match first
        image_path = None
        if extr.name in image_map:
            image_path = image_map[extr.name]
        else:
            # Try matching with basename only (in case COLMAP stores paths like "gtimages/00010.jpg")
            basename = os.path.basename(extr.name)
            if basename in image_map:
                image_path = image_map[basename]
            else:
                # Try without extension matching (e.g., .jpg vs .png)
                name_without_ext = os.path.splitext(basename)[0]
                for key in image_map.keys():
                    if os.path.splitext(key)[0] == name_without_ext:
                        image_path = image_map[key]
                        break
        
        if image_path is None:
            # Debug: print for first few mismatches
            if idx < 3:
                print(f"Warning: COLMAP image '{extr.name}' not found in image_map")
                print(f"  Tried: direct match, basename '{os.path.basename(extr.name)}', and extension-agnostic matching")
            return None
        
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.open(image_path)
        
        depth_params = None
        if depths_params is not None:
            try:
                # Use the image_name (without extension) as the key
                depth_params = depths_params[image_name]
            except:
                # Try with original COLMAP name if the first attempt fails
                try:
                    colmap_basename = os.path.splitext(os.path.basename(extr.name))[0]
                    depth_params = depths_params[colmap_basename]
                except:
                    if idx < 3:  # Only print for first few mismatches
                        print(f"\nWarning: depth_params not found for '{image_name}' or '{colmap_basename}'")
        
        # Load masks with distinction between alpha_mask and validity_mask
        # For genimages: validity_mask is from masks/genimages/, alpha_mask is from object_mask/
        # For gtimages: alpha_mask is from object_mask/, validity_mask is None
        mask = None
        validity_mask = None
        
        gen_mask_path = get_genimages_mask_path(image_path, images_folder)
        if gen_mask_path is not None:
            # This is a genimages image - load validity mask
            validity_mask = Image.open(gen_mask_path)
        
        # Load alpha_mask from standard masks_folder for all images
        # Use the basename of image_path to ensure correct filename
        image_basename_with_ext = os.path.basename(image_path)
        
        if masks_folder is not None:
            mask_path = os.path.join(masks_folder, image_basename_with_ext)
            # Try with .png extension if not found
            if not os.path.exists(mask_path):
                mask_path = os.path.join(masks_folder, os.path.splitext(image_basename_with_ext)[0] + ".png")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
            
        if sky_masks_folder is not None:
            sky_mask_path = os.path.join(sky_masks_folder, image_basename_with_ext)
            # Try with .png extension if not found
            if not os.path.exists(sky_mask_path):
                sky_mask_path = os.path.join(sky_masks_folder, os.path.splitext(image_basename_with_ext)[0] + ".png")
            if os.path.exists(sky_mask_path):
                sky_mask = Image.open(sky_mask_path).convert("L")
            else:
                sky_mask = None
        else:   
            sky_mask = None
            
        if depths_folder is not None:
            # Use image_name (without extension) and add .png
            depth_path = os.path.join(depths_folder, image_name + ".png")
            if os.path.exists(depth_path):
                depth = cv2.imread(depth_path, -1).astype(np.float32) / float(2**16)
            else:
                depth = None
        else:
            depth = None

        return CameraInfo(
            uid=uid, 
            R=R, 
            T=T, 
            FovY=FovY,
            FovX=FovX, 
            CX=CX,
            CY=CY,
            image=image,
            mask=mask,
            sky_mask=sky_mask,
            depth=depth,
            depth_params=depth_params,
            image_path=image_path, 
            image_name=image_name, 
            width=width, 
            height=height,
            validity_mask=validity_mask
        )

    ct = 0
    progress_bar = tqdm(cam_extrinsics, desc="Loading dataset")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, idx, key) for idx, key in enumerate(cam_extrinsics)]

        for future in concurrent.futures.as_completed(futures):
            cam_info = future.result()
            if cam_info is not None:
                cam_infos.append(cam_info)
            
            ct+=1
            if ct % 10 == 0:
                progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(cam_extrinsics)}"+Style.RESET_ALL})
                progress_bar.update(10)
        
        progress_bar.close()

    cam_infos = sorted(cam_infos, key = lambda x : x.image_path)
    print(f"Loaded {len(cam_infos)} images (from COLMAP extrinsics: {len(cam_extrinsics)})")
    return cam_infos

def readCamerasFromTransforms(path, transformsfile, add_mask, add_depth, center, scale, depth_type='ue', depths_params=None):

    cam_infos = []
    test_cam_infos = []
    
    # Scan all available images
    images_folder = os.path.join(path, "images")
    image_map = scan_all_images_in_folders(images_folder)
    
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # fovx = contents["camera_angle_x"]
        fl_y = contents["fl_y"]
        fl_x = contents["fl_x"]
        cx = contents["cx"]
        cy = contents["cy"]
        
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # Get image filename from frame
            image_filename = os.path.basename(frame["file_path"])
            
            # Look up in image_map
            if image_filename not in image_map:
                print(f"Warning: Image {image_filename} not found in scanned images")
                continue
            
            image_path = image_map[image_filename]
            cam_name = os.path.join("images", frame["file_path"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:, [0, 1]] = c2w[:, [1, 0]]  
            c2w[:3, 2:3] *= -1  

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            FovY = focal2fov(fl_y, image.size[1])
            FovX = focal2fov(fl_x, image.size[0])
            CX = cx
            CY = cy
            
            # Load mask - check if it's a genimages image first
            mask = None
            validity_mask = None
            gen_mask_path = get_genimages_mask_path(image_path, images_folder)
            if gen_mask_path is not None:
                validity_mask = Image.open(gen_mask_path)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, CX=CX, CY=CY, image=image, mask=mask, sky_mask=None, depth=None, depth_params=None,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], validity_mask=validity_mask))

        test_frames = contents["test_frames"]
        for idx, frame in enumerate(test_frames):
            # Get image filename from frame
            image_filename = os.path.basename(frame["file_path"])
            
            # Look up in image_map
            if image_filename not in image_map:
                print(f"Warning: Test image {image_filename} not found in scanned images")
                continue
            
            image_path = image_map[image_filename]
            cam_name = os.path.join("images", frame["file_path"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1
            c2w[:, [0, 1]] = c2w[:, [1, 0]]  
            c2w[:3, 2:3] *= -1 

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            FovY = focal2fov(fl_y, image.size[1])
            FovX = focal2fov(fl_x, image.size[0])
            CX = cx
            CY = cy
            
            # Load mask - check if it's a genimages image first
            mask = None
            validity_mask = None
            gen_mask_path = get_genimages_mask_path(image_path, images_folder)
            if gen_mask_path is not None:
                validity_mask = Image.open(gen_mask_path)

            test_cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, CX=CX, CY=CY, image=image, mask=mask, sky_mask=None, depth=None, depth_params=None,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], validity_mask=validity_mask))

    cam_infos = sorted(cam_infos, key = lambda x : x.image_path)
    test_cam_infos = sorted(test_cam_infos, key= lambda x : x.image_path)
    print(f"Loaded {len(cam_infos)} training images and {len(test_cam_infos)} test images")
    return cam_infos, test_cam_infos

def readColmapSceneInfo(path, eval, images, depths, masks, add_mask, add_depth, llffhold=32, sky_masks=None, add_sky_mask=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        try:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(path, "colmap", "images.txt")
            cameras_intrinsic_file = os.path.join(path, "colmap", "cameras_undistorted.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    depths_params = None
    if add_depth:
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    reading_dir = os.path.join(path, images)
    mask_dir = os.path.join(path, masks) if add_mask else None
    depth_dir = os.path.join(path, depths) if add_depth else None
    sky_mask_dir = os.path.join(path, sky_masks) if add_sky_mask and sky_masks else None
    cam_infos = readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, reading_dir, mask_dir, depth_dir, sky_mask_dir)
    gtimages_cams = []
    genimages_cams = []
    for cam in cam_infos:
        # We can check the path, which is stored in the CameraInfo object
        if "genimages" in cam.image_path:
            genimages_cams.append(cam)
        else:
            gtimages_cams.append(cam)

    print(f"Loaded {len(gtimages_cams)} gtimages images and {len(genimages_cams)} genimages images.")

    if eval:
        gt_train = []
        gt_test = []
        for idx, cam in enumerate(gtimages_cams):
            if idx % llffhold == 0:
                gt_test.append(cam)
            else:
                gt_train.append(cam)
        train_cam_infos = gt_train + genimages_cams
        test_cam_infos = gt_test
        print(f"Final splitting result: {len(train_cam_infos)} training images ({len(gt_train)} GT + {len(genimages_cams)} Gen), {len(test_cam_infos)} testing images.")


        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]

        # if "lerf" in path:
        #     if "waldo_kitchen" in path:
        #         test_frame = ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"]
        #         test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
        #     elif "ramen" in path:
        #         test_frame = ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"]
        #         test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
        #     elif "figurines" in path:
        #         test_frame = ["frame_00041", "frame_00105", "frame_00152", "frame_00195"]
        #         test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
        #     elif "teatime" in path:
        #         test_frame = ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
        #         test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
        # if "3dovs" in path:
        #     if "bed" in path:
        #         test_frame = ["00", "04", "10", "23", "30"]
        #         test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
        #     elif "bench" in path:
        #         test_frame = ["02", "05", "25", "27", "32"]
        #         test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
        #     elif "blue_sofa" in path:
        #         test_frame = ["03", "05", "13", "24", "27"]
        #         test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
        #     elif "covered_desk" in path:
        #         test_frame = ["00", "01", "11", "26", "29"]
        #         test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
        #     elif "lawn" in path:
        #         test_frame = ["01", "03", "09", "13", "29"]
        #         test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
        #     elif "office_desk" in path:
        #         test_frame = ["03", "07", "12", "14", "20"]
        #         test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
        #     elif "room" in path:
        #         test_frame = ["00", "04", "19", "25", "30"]
        #         test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
        #     elif "snacks" in path:
        #         test_frame = ["04", "08", "18", "26", "40"]
        #         test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
        #     elif "sofa" in path:
        #         test_frame = ["02", "04", "10", "15", "22"]
        #         test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]
        #     elif "table" in path:
        #         test_frame = ["00", "02", "14", "26", "30"]
        #         test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_frame]            
        # else:
        #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if "test" not in c.image_name]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if "test" in c.image_name]


    nerf_normalization = getNerfppNorm(train_cam_infos)

    if "3dovs" in path or "lerf_ovs" in path:
        ply_path = os.path.join(path, "sparse/0/points3D_deva.ply")
    elif "scannet" in path:
        ply_path = os.path.join(path, "points3D.ply")
    elif "mipnerf360" in path:
        ply_path = os.path.join(path, "points3D.ply")
    else:
        ply_path = os.path.join(path, "sparse/0/points3D_corr.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # txt_path = os.path.join(path, "sparse/0/points3D.txt")
    
    if not os.path.exists(ply_path):
        # print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        # try:
        #     xyz, rgb, _ = read_points3D_binary(bin_path)
        # except:
        #     xyz, rgb, _ = read_points3D_text(txt_path)
        # storePly(ply_path, xyz, rgb)
        raise FileNotFoundError
    # try:
    print(f'start fetching data from ply file')
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readNerfSyntheticInfo(path, eval, add_mask, add_depth, center, scale):
    # print("Reading Training Transforms")
    # train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", add_mask, add_depth, center, scale)
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", add_mask, add_depth, center, scale)
    print("Reading Training and Test Transforms")
    train_cam_infos, test_cam_infos = readCamerasFromTransforms(path, "nerfstudio/transforms_undistorted.json", add_mask, add_depth, center, scale)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_paths = glob.glob(os.path.join(path, "points3D.ply"))
    if len(ply_paths)==0:
        ply_path = os.path.join(path, "points3d.ply")
        # Since this data set has no colmap data, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        colors = np.random.random((num_pts, 3))
        normals=np.zeros((num_pts, 3))
        pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)

        storePly(ply_path, xyz, colors*255)
    else:
        ply_path = ply_paths[0]
        pcd = fetchPly(ply_path)
    
    pcd.points[:, :] -= center
    pcd.points[:, :] /= scale  # mainly adapt to params

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCityInfo(path, eval, add_mask, add_depth, center, scale, ds=1, llffhold=32, depth_type='ue'):
    
    # json_path = glob.glob(os.path.join(path, f"transforms.json"))[0].split('/')[-1]
    json_path = "transforms.json" if ds==1 else f"transforms_{ds}.json"
    print("Reading Training Transforms from {}".format(json_path))
    
    # load ply
    ply_path = glob.glob(os.path.join(path, "*.ply"))[0]
    if os.path.exists(ply_path):
        try:
            pcd = fetchPly(ply_path)
        except:
            raise ValueError("must have tiepoints!")
    else:
        las_paths = glob.glob(os.path.join(path, "LAS/*.las"))
        las_path = las_paths[0]
        print(f'las_path: {las_path}')
        try:
            pcd = read_multiple_las_files(las_paths, ply_path)
        except:
            raise ValueError("Load LAS failed!")
    
    # recenter poses and points clouds
    pcd.points[:,:] -= center
    pcd.points[:,:] /=scale # mainly adapt to params
    
    depth_params_file = os.path.join(path, "depth_params.json")
    depths_params = None
    if add_depth:
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)
            
    # load camera
    cam_infos = readCamerasFromTransforms(path, json_path, add_mask, add_depth, center, scale, depth_type=depth_type, depths_params=depths_params)
    
    print("Load Cameras: ", len(cam_infos))
    train_cam_infos = []
    test_cam_infos = []
    
    if not eval:
        train_cam_infos.extend(cam_infos)
        test_cam_infos = []
    else:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

    
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "City": readCityInfo
}