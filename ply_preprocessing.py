"""
PLY Preprocessing Script for ObjectGS

This script processes COLMAP 3D point clouds and assigns object labels based on 2D masks.
It supports both ground truth images and generated images, with special handling for 
generated images to filter out "unreal" pixels.

Key Features:
1. Scans all images from gtimages/ and genimages/ folders
2. For generated images, loads validity masks from masks/genimages/{xxxxx_forward}/
3. Filters out unreal pixels (mask == 0) from voting process
4. Assigns object labels to 3D points through multi-view voting
5. Outputs labeled point cloud as PLY file

Data Structure:
- images/gtimages/*.jpg                              - Ground truth images
- images/genimages/xxxxx_forward/*.png               - Generated images (starting from frame xxxxx)
- masks/genimages/xxxxx_forward/*.png                - Validity masks for generated images
- object_mask/*.png                                  - Object segmentation masks
- sparse/0/cameras.bin                               - Camera intrinsics
- sparse/0/images.bin                                - Camera extrinsics
- sparse/0/points3D.bin                              - 3D point cloud

Usage:
    python ply_preprocessing.py --dataset_path /path/to/dataset
    python ply_preprocessing.py --dataset_path /path/to/dataset --target specific_scene
    python ply_preprocessing.py --dataset_path /path/to/dataset --max_cores 16
"""

import struct
import numpy as np
import cv2
from collections import Counter, defaultdict
from plyfile import PlyData, PlyElement
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary
from scene.dataset_readers import scan_all_images_in_folders
import multiprocessing as mp
from multiprocessing import Pool
import os

def read_points3D_binary(path_to_model_file):
    """
    Parses COLMAP's points3D.bin file and returns a dictionary:
    point3D_id -> (x, y, z, r, g, b, error, track)
    where track is a list of (image_id, point2D_idx) tuples.
    """
    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    import struct
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        points3D = {}

        for _ in range(num_points):
            data = read_next_bytes(fid, 43, "QdddBBBd")  # point3D_id + xyz + rgb + error
            point3D_id = data[0]
            xyz = np.array(data[1:4])
            rgb = np.array(data[4:7])
            error = data[7]

            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, 8 * track_length, "ii" * track_length)

            # track is a list of (image_id, point2D_idx)
            track = [(track_elems[i], track_elems[i + 1]) for i in range(0, len(track_elems), 2)]

            points3D[point3D_id] = (xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2], error, track)

    return points3D


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1

    point3D = {}

    # xyzs = np.empty((num_points, 3))
    # rgbs = np.empty((num_points, 3))
    # errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                # xyzs[count] = xyz
                # rgbs[count] = rgb
                # errors[count] = error
                point3D[count] = np.concatenate((xyz, rgb), axis=0)
                count += 1

    return point3D


class ID2RGBConverter:
    def __init__(self):
        self.all_id = []  # Store all generated IDs
        self.obj_to_id = {}  # Mapping from object ID to randomly generated color ID

    # Convert integer ID to RGB color
    def _id_to_rgb(self, id: int):
        rgb = np.zeros((3, ), dtype=np.uint8)  # Initialize RGB channels
        for i in range(3):
            rgb[i] = id % 256  # Take the lower 8 bits of the ID as the RGB channel value
            id = id // 256  # Right shift 8 bits to process the remaining part
        return rgb

    # Convert single-channel ID to random RGB value
    def convert(self, obj: int):
        if obj in self.obj_to_id:
            id = self.obj_to_id[obj]  # If the object already exists, directly get the corresponding ID
        else:
            # Randomly generate a unique ID and ensure no duplicates
            while True:
                id = np.random.randint(255, 256**3)
                if id not in self.all_id:
                    break
            self.obj_to_id[obj] = id  # Store the new ID in the dictionary
            self.all_id.append(id)  # Record this ID

        return id, self._id_to_rgb(id)  # Return the ID and corresponding RGB value


class ImageCache:
    """Simple image cache to avoid repeated file reads"""
    def __init__(self):
        self.cache = {}

    def get(self, image_path):
        if image_path not in self.cache:
            self.cache[image_path] = cv2.imread(image_path, -1)
        return self.cache[image_path]


def get_genimages_mask_path(image_path, dataset_root):
    """
    Get the mask path for a genimages image.
    If image is in images/genimages/xxxxx_forward/*.png, 
    mask is in masks/genimages/xxxxx_forward/*.png
    Returns None if not a genimages image or mask doesn't exist.
    """
    # Check if this is a genimages image
    if "genimages" not in image_path:
        return None
    
    # Replace images/genimages/ with masks/genimages/
    try:
        # Find the dataset root by going up from images folder
        if "/images/genimages/" in image_path:
            mask_path = image_path.replace("/images/genimages/", "/masks/genimages/")
        elif "\\images\\genimages\\" in image_path:
            mask_path = image_path.replace("\\images\\genimages\\", "\\masks\\genimages\\")
        else:
            return None
        
        if os.path.exists(mask_path):
            return mask_path
    except Exception as e:
        print(f"Warning: Error constructing mask path for {image_path}: {e}")
    
    return None


def process_points_batch(args):
    """Process a batch of 3D points in parallel"""
    points_batch, images, label_image_dir, converter, image_cache, image_map, dataset_root = args
    local_colors = []
    local_labels = []

    for point3D_id, point_data in points_batch:
        x, y, z, r, g, b, error, track = point_data

        for image_id, point2D_idx in track:
            if image_id not in images:
                continue

            # Get the stored image name
            _, _, _, _, stored_image_name, xys, _ = images[image_id]
            image_name = os.path.basename(stored_image_name)

            if point2D_idx >= len(xys):
                continue

            u, v = xys[point2D_idx]
            u, v = int(round(u)), int(round(v))

            # Get the actual image path from image_map
            if image_name not in image_map:
                continue
            
            image_path = image_map[image_name]
            
            # Check if this is a genimages image and load its validity mask
            gen_mask_path = get_genimages_mask_path(image_path, dataset_root)
            if gen_mask_path is not None:
                # This is a generated image, check validity mask
                validity_mask = image_cache.get(gen_mask_path)
                if validity_mask is None:
                    continue
                
                # Check if the pixel is valid (mask > 0 means valid/real)
                if v < 0 or v >= validity_mask.shape[0] or u < 0 or u >= validity_mask.shape[1]:
                    continue
                
                # Skip unreal pixels (mask == 0)
                if validity_mask[v, u] == 0:
                    continue  # This pixel is "unreal", skip it from voting
            
            # Load object mask (label image) - all images use the same object_mask folder
            label_image_file = os.path.join(label_image_dir, image_name)
            
            # The replace logic remains the same
            label_image_file = label_image_file.replace('.jpg', '.png') if label_image_file.endswith('.jpg') else label_image_file.replace('.JPG', '.png')

            label_image = image_cache.get(label_image_file)
            if label_image is None or v < 0 or v >= label_image.shape[0] or u < 0 or u >= label_image.shape[1]:
                continue

            obj_id = label_image[v, u]
            _, rgb_color = converter.convert(obj_id)

            local_colors.append((point3D_id, rgb_color))
            local_labels.append((point3D_id, obj_id))

    return local_colors, local_labels


def assign_final_colors(points3D, all_colors, all_labels):
    """Assign final colors and labels through voting"""
    point_final_labels = {}
    point_final_colors = {}

    colors_dict = defaultdict(list)
    labels_dict = defaultdict(list)

    for pid, color in all_colors:
        colors_dict[pid].append(color)
    for pid, label in all_labels:
        labels_dict[pid].append(label)

    for point_id in colors_dict:
        colors = colors_dict[point_id]
        labels = labels_dict[point_id]

        # Filter out invalid labels
        filtered = [(c, l) for c, l in zip(colors, labels) if l != 0]
        if not filtered:
            continue

        filtered_colors, filtered_labels = zip(*filtered)
        counter = Counter(filtered_labels)
        max_value = max(counter, key=counter.get)

        # Find color for most common label
        label_indices = [i for i, label in enumerate(filtered_labels) if label == max_value]
        max_color = filtered_colors[label_indices[0]]

        point_final_labels[point_id] = max_value
        point_final_colors[point_id] = max_color

    # Create new points3D with final colors
    new_points3D = {}
    for point_id, point_data in points3D.items():
        x, y, z, r, g, b, error, track = point_data
        r_new, g_new, b_new = point_final_colors.get(point_id, (r, g, b))
        label = point_final_labels.get(point_id, 0)
        new_points3D[point_id] = (x, y, z, r_new, g_new, b_new, label)

    return new_points3D


def storePly(path, xyz, rgb, label):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
             ('label', 'u1')]
    
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, label), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# Main process
def main():
    import os
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Process COLMAP scenarios into PLYs.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data3/zewei/ObjectGS/waymo_example/10061305430875486848_1080_000_1100_000-50colmap-gengt",
        help="Path to either the parent folder containing scenario folders OR a single scenario folder."
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional: name of the specific scenario folder to process (basename). If provided, only this one will be processed."
    )
    parser.add_argument(
        "--max_cores",
        type=int,
        default=8,
        help="Maximum number of CPU cores to use for multiprocessing (default 8)."
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    target = args.target
    max_cores = args.max_cores

    print(f"Starting processing with dataset_path: {dataset_path}, target: {target}")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist!")
        return

    # If dataset_path looks like a single scenario (has a 'sparse' folder), treat it as single scenario
    if os.path.isdir(os.path.join(dataset_path, "sparse")):
        dataset_root = os.path.dirname(dataset_path)
        dataset_folders = [os.path.basename(dataset_path)]
    else:
        dataset_root = dataset_path
        dataset_folders = sorted(os.listdir(dataset_root))

    # If user provided --target, restrict to that one (name only)
    if target:
        if target in dataset_folders:
            dataset_folders = [target]
        else:
            print(f"Warning: target '{target}' not found under {dataset_root}. Available: {dataset_folders}")
            return

    print(f"Found {len(dataset_folders)} dataset folders to process: {dataset_folders}")

    for dataset_folder in dataset_folders:
        print(f"\n{'='*50}")
        print(f"Processing {dataset_folder}...")
        start_time = time.time()

        folder_path = os.path.join(dataset_root, dataset_folder)

        label_image_dir = os.path.join(folder_path, 'object_mask')
        output_ply_file = os.path.join(folder_path, 'sparse/0/points3D_corr.ply')
        camera_file = os.path.join(folder_path, 'sparse/0/cameras.bin')
        image_file = os.path.join(folder_path, 'sparse/0/images.bin')
        points3D_file = os.path.join(folder_path, 'sparse/0/points3D.bin')

        # Check if required files exist
        required_files = [camera_file, image_file, points3D_file, label_image_dir]
        missing = False
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"Warning: Required file/directory not found: {file_path}")
                missing = True
        if missing:
            print(f"Skipping {dataset_folder} due to missing files.")
            continue

        print(f"Reading camera data from: {camera_file}")
        cameras = read_intrinsics_binary(camera_file)
        print(f"Read {len(cameras)} cameras")

        print(f"Reading image data from: {image_file}")
        images = read_extrinsics_binary(image_file)
        print(f"Read {len(images)} images")

        print(f"Reading points3D data from: {points3D_file}")
        points3D = read_points3D_binary(points3D_file)
        print(f"Read {len(points3D)} 3D points")

        # Scan all images (gtimages and genimages)
        images_folder = os.path.join(folder_path, 'images')
        print(f"Scanning images in {images_folder}...")
        image_map = scan_all_images_in_folders(images_folder)
        print(f"Found {len(image_map)} images in total (gtimages + genimages)")
        
        # Count gtimages vs genimages
        gt_count = sum(1 for path in image_map.values() if 'gtimages' in path)
        gen_count = sum(1 for path in image_map.values() if 'genimages' in path)
        print(f"  - Ground truth images: {gt_count}")
        print(f"  - Generated images: {gen_count}")

        # Use parallel processing with image caching
        num_cores = min(mp.cpu_count(), max_cores)
        print(f"Using {num_cores} CPU cores for parallel processing")

        converter = ID2RGBConverter()
        image_cache = ImageCache()

        print(f"Processing {len(points3D)} points...")

        # Split points into batches for parallel processing
        points_items = list(points3D.items())
        batch_size = max(1000, len(points_items) // (num_cores * 2) if num_cores > 0 else len(points_items))
        batches = [points_items[i:i + batch_size] for i in range(0, len(points_items), batch_size)]

        print(f"Split into {len(batches)} batches")

        # Parallel processing
        all_colors = []
        all_labels = []

        with Pool(processes=num_cores) as pool:
            args_list = [(batch, images, label_image_dir, converter, image_cache, image_map, folder_path) for batch in batches]
            results = pool.map(process_points_batch, args_list)

            for batch_colors, batch_labels in results:
                all_colors.extend(batch_colors)
                all_labels.extend(batch_labels)

        print(f"Processed {len(all_colors)} color mappings, {len(all_labels)} label mappings")
        print(f"Cached {len(image_cache.cache)} images (including validity masks)")
        
        # Calculate how many observations were used
        total_observations_possible = sum(len(point_data[7]) for point_data in points3D.values())
        observations_used = len(all_colors)
        observations_filtered = total_observations_possible - observations_used
        print(f"Total possible observations: {total_observations_possible}")
        print(f"Observations used for voting: {observations_used}")
        print(f"Observations filtered out (unreal pixels from genimages): {observations_filtered}")
        if total_observations_possible > 0:
            print(f"Filtering rate: {observations_filtered/total_observations_possible*100:.2f}%")

        # Assign colors and labels through voting
        print("Assigning final colors and labels...")
        points3D = assign_final_colors(points3D, all_colors, all_labels)

        # Extract and save PLY file
        print("Extracting data for PLY file...")
        xyz = np.array([[p[0], p[1], p[2]] for p in points3D.values()])
        rgb = np.array([[p[3], p[4], p[5]] for p in points3D.values()])
        label = np.array([[p[6]] for p in points3D.values()])

        print(f"Saving PLY file to: {output_ply_file}")
        os.makedirs(os.path.dirname(output_ply_file), exist_ok=True)
        storePly(output_ply_file, xyz, rgb, label)

        elapsed_time = time.time() - start_time
        print(f"Point cloud saved to {output_ply_file}")
        print(f"Processing completed in {elapsed_time:.2f} seconds")
        print(f"Output PLY contains {len(xyz)} points")

    print(f"\n{'='*50}")
    print("All requested datasets processed successfully!")

if __name__ == "__main__":
    main()


# python ply_preprocessing.py --dataset_path /data3/zewei/ObjectGS/waymo_example
# python ply_preprocessing.py --dataset_path /data3/zewei/ObjectGS/waymo_example --target 10275144660749673822_5755_561_5775_561-50colmap