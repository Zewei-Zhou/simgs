import os
import logging
from argparse import ArgumentParser
import shutil

parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--force_reextract", action='store_true', help='Force re-extraction of features even if database exists')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--mask_path", "-m", type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--glomap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()

colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
glomap_command = '"{}"'.format(args.glomap_executable) if len(args.glomap_executable) > 0 else "glomap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0


if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    # If force re-extraction is requested, remove existing database
    if args.force_reextract:
        db_path = args.source_path + "/distorted/database.db"
        if os.path.exists(db_path):
            print(f"Removing existing database: {db_path}")
            os.remove(db_path)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/images \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu) 


    if args.mask_path is None:
        args.mask_path = args.source_path + "/masks"

    # Check if masks can be used (skip if naming doesn't match to avoid COLMAP errors)
    use_masks = False

    if os.path.exists(args.mask_path):
        try:
            import glob
            image_files = glob.glob(os.path.join(args.source_path, "images", "*.jpg"))

            if image_files:
                # Check first image and corresponding mask
                first_image = os.path.basename(image_files[0])
                expected_mask = os.path.join(args.mask_path, first_image)

                if os.path.exists(expected_mask):
                    use_masks = True
                    print(f"Using mask path: {args.mask_path}")
                    feat_extracton_cmd += " --ImageReader.mask_path " + args.mask_path

                else:
                    print(f"Warning: Mask files found but naming doesn't match images. Skipping masks.")
                    print(f"Images are named like: {first_image}")
                    print(f"Expected mask: {expected_mask}")
            else:
                print("Warning: No image files found. Skipping masks.")

        except Exception as e:
            print(f"Warning: Error checking masks: {e}. Skipping masks.")

    else:
        print("No mask directory found. Proceeding without masks.")

    print(f"Executing: {feat_extracton_cmd}")

    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " sequential_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu) + \
        " --SiftMatching.max_num_matches 16384"

    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    mapper_cmd = (glomap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/images \
        --output_path "  + args.source_path + "/distorted/sparse"
    )

    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/images \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")

print(f"Running image undistortion with command: {img_undist_cmd}")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Image undistortion failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)

# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue

    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if(args.resize):
    print("Copying and resizing...")
    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory

    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)
        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)

        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)

        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)

        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print(f"Finished processing {args.source_path}")