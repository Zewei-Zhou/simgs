import os
import argparse

def create_image_list(base_path, output_file):
    """
    Scans for camera folders (e.g., camera_front, camera_front_left)
    and generates an image_list.txt for COLMAP.
    """
    # Define the expected camera subdirectories
    camera_folders = ['camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT']
    
    # Filter to only include directories that actually exist
    existing_camera_folders = [f for f in camera_folders if os.path.isdir(os.path.join(base_path, f))]
    
    if not existing_camera_folders:
        print(f"Error: No camera folders found in {base_path}. Looked for: {camera_folders}")
        return

    print(f"Found camera folders: {existing_camera_folders}")
    
    with open(output_file, 'w') as f:
        # Assign a unique camera_id to each folder
        for camera_id, folder_name in enumerate(existing_camera_folders):
            camera_path = os.path.join(base_path, folder_name)
            
            # List all image files, sort them to ensure correct order
            try:
                images = sorted([img for img in os.listdir(camera_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
            except FileNotFoundError:
                print(f"Warning: Directory not found: {camera_path}")
                continue

            for image_name in images:
                # The path in the list must be relative to the --image_path argument in COLMAP
                relative_path = os.path.join(folder_name, image_name)
                # Write "image_path.jpg CAMERA_ID"
                f.write(f"{relative_path} {camera_id}\n")
                
    print(f"Successfully created {output_file} with {len(existing_camera_folders)} cameras.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image_list.txt for COLMAP multi-camera rig.")
    parser.add_argument("--source_path", "-s", required=True, help="The root path containing camera folders (e.g., camera_front).")
    args = parser.parse_args()
    
    output_txt_file = os.path.join(args.source_path, "image_list.txt")
    create_image_list(args.source_path, output_txt_file)