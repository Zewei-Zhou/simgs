import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from grounding_sam2_utils.video_utils import create_video_from_images
from grounding_sam2_utils.common_utils import CommonUtils
from grounding_sam2_utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy
import re
import shutil

def setup_environment():
    """Sets up the computing environment, especially GPU settings."""
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    return device

def load_models(device, model_name, grounding_model_id):
    """Loads the SAM2 and Grounding DINO models."""
    image_predictor = SAM2ImagePredictor.from_pretrained(model_name)
    video_predictor = SAM2VideoPredictor.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)
    return video_predictor, image_predictor, processor, grounding_model

def prepare_directories_and_frames(video_dir, output_dir, generate_sky_masks=False, is_genimages=False, dataset_root_dir=None):
    """Creates output directories and returns sorted frame names.
    
    Args:
        video_dir: Directory containing video frames
        output_dir: Output directory for mask_data and json_data
        generate_sky_masks: Whether to generate sky masks
        is_genimages: Whether processing genimages (affects sorting)
        dataset_root_dir: Root directory for shared masks (object_mask, vis_object_mask, sky_masks)
    """
    # Verify video_dir exists and has images
    if not os.path.exists(video_dir):
        print(f"Error: video_dir does not exist: {video_dir}")
        return None, None, [], None, None, None
    
    CommonUtils.creat_dirs(output_dir)
    mask_data_dir = os.path.join(output_dir, "mask_data")
    json_data_dir = os.path.join(output_dir, "json_data")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)

    # Use dataset_root_dir for shared mask directories if provided, otherwise use parent of output_dir
    if dataset_root_dir is None:
        parent_dir = os.path.dirname(output_dir)
    else:
        parent_dir = dataset_root_dir
    
    object_mask_dir = os.path.join(parent_dir, "object_mask")
    vis_object_mask_dir = os.path.join(parent_dir, "vis_object_mask")
    CommonUtils.creat_dirs(object_mask_dir)
    CommonUtils.creat_dirs(vis_object_mask_dir)
    
    sky_masks_dir = None
    if generate_sky_masks:
        sky_masks_dir = os.path.join(parent_dir, "sky_masks")
        CommonUtils.creat_dirs(sky_masks_dir)

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    
    if not frame_names:
        print(f"Warning: No image files found in {video_dir}")
        all_files = os.listdir(video_dir)
        print(f"Found {len(all_files)} files in directory: {all_files[:10]}")  # Show first 10
    
    # Sort frames based on whether it's gtimages or genimages
    if is_genimages:
        # For genimages like "00010_forward_000000.png", extract the sequence number
        def extract_seq_num(filename):
            match = re.search(r'_(\d+)\.(png|jpg|jpeg)$', filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return 0
        frame_names.sort(key=extract_seq_num)
    else:
        # For gtimages like "00000.jpg", extract frame number
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    return mask_data_dir, json_data_dir, frame_names, sky_masks_dir, object_mask_dir, vis_object_mask_dir

def load_reference_mask(reference_mask_path):
    """Load a reference mask and return a MaskDictionaryModel with object IDs."""
    if not os.path.exists(reference_mask_path):
        print(f"Warning: Reference mask not found at {reference_mask_path}")
        return None
    
    # Load mask as uint16
    reference_mask = cv2.imread(reference_mask_path, cv2.IMREAD_UNCHANGED)
    if reference_mask is None:
        print(f"Warning: Failed to load reference mask from {reference_mask_path}")
        return None
    
    # Get unique object IDs
    unique_ids = np.unique(reference_mask)
    unique_ids = unique_ids[unique_ids > 0]  # Remove background (0)
    
    print(f"Loaded reference mask with {len(unique_ids)} objects: {unique_ids.tolist()}")
    
    # Create MaskDictionaryModel
    mask_dict = MaskDictionaryModel(promote_type='mask')
    mask_dict.mask_height = reference_mask.shape[0]
    mask_dict.mask_width = reference_mask.shape[1]
    
    # Create ObjectInfo for each object ID
    for obj_id in unique_ids:
        obj_mask = (reference_mask == obj_id)
        # Convert to torch tensor
        obj_mask_torch = torch.from_numpy(obj_mask).bool()
        
        # Create ObjectInfo (assume generic class name, can be improved)
        obj_info = ObjectInfo(
            instance_id=int(obj_id),
            mask=obj_mask_torch,
            class_name=f"object_{obj_id}"  # Generic class name
        )
        obj_info.update_box()
        mask_dict.labels[int(obj_id)] = obj_info
    
    return mask_dict

def generate_sky_mask_sam2(image, image_predictor, processor, grounding_model, device):
    """
    Generate sky mask using SAM2 with 'sky' text prompt.
    Returns a binary mask where 1 indicates sky regions.
    """
    sky_text = "sky. clouds."
    inputs = processor(images=image, text=sky_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.15, text_threshold=0.15, target_sizes=[image.size[::-1]]
    )
    input_boxes = results[0]["boxes"]
    
    if input_boxes.shape[0] > 0:
        image_predictor.set_image(np.array(image.convert("RGB")))
        masks, scores, logits = image_predictor.predict(
            point_coords=None, point_labels=None, box=input_boxes, multimask_output=False,
        )
        if masks.ndim == 2:
            masks = masks[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)
        combined_sky_mask = np.zeros(masks.shape[1:], dtype=np.uint8)
        for mask in masks:
            combined_sky_mask = np.logical_or(combined_sky_mask, mask).astype(np.uint8)
        return combined_sky_mask * 255
    else:
        return generate_simple_sky_mask(image)

def process_sky_only_batch(frame_names, video_dir, device, processor, grounding_model, image_predictor, sky_masks_dir):
    """
    Process all frames to generate sky masks only, ensuring complete coverage.
    """
    print(f"Processing {len(frame_names)} frames for sky mask generation...")
    for frame_idx, frame_name in enumerate(frame_names):
        print(f"Generating sky mask for frame {frame_idx + 1}/{len(frame_names)}: {frame_name}")
        img_path = os.path.join(video_dir, frame_name)
        image = Image.open(img_path)
        sky_mask = generate_sky_mask_sam2(image, image_predictor, processor, grounding_model, device)
        sky_mask_name = frame_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        sky_mask_path = os.path.join(sky_masks_dir, sky_mask_name)
        cv2.imwrite(sky_mask_path, sky_mask)
    print(f"Successfully generated sky masks for all {len(frame_names)} frames")

def generate_simple_sky_mask(image):
    """Fallback sky detection using simple heuristics."""
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    if len(img_array.shape) == 3:
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    else:
        hsv = img_array
    sky_mask = np.zeros((height, width), dtype=np.uint8)
    upper_region = hsv[:height//2, :]
    lower_blue, upper_blue = np.array([100, 30, 50]), np.array([130, 255, 255])
    blue_mask = cv2.inRange(upper_region, lower_blue, upper_blue)
    lower_gray, upper_gray = np.array([0, 0, 120]), np.array([180, 50, 255])
    gray_mask = cv2.inRange(upper_region, lower_gray, upper_gray)
    combined_mask = cv2.bitwise_or(blue_mask, gray_mask)
    sky_mask[:height//2, :] = combined_mask
    kernel = np.ones((5, 5), np.uint8)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
    return sky_mask

def save_frame_results(frame_idx, frame_masks_info, mask_data_dir, json_data_dir, object_mask_dir, sky_mask=None, sky_masks_dir=None, frame_name=None):
    """Saves the mask and JSON data for a single frame, and optionally sky mask."""
    mask = frame_masks_info.labels
    mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
    for obj_id, obj_info in mask.items():
        mask_img[obj_info.mask == True] = obj_id

    mask_img_np = mask_img.cpu().numpy().astype(np.uint16)
    
    np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img_np)
    
    original_name = frame_masks_info.mask_name.replace("mask_", "").replace(".npy", "")
    png_filename = f"{original_name}.png"
    png_path = os.path.join(object_mask_dir, png_filename)
    cv2.imwrite(png_path, mask_img_np, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    json_data = frame_masks_info.to_dict()
    json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
    with open(json_data_path, "w") as f:
        json.dump(json_data, f)
    
    if sky_mask is not None and sky_masks_dir is not None and frame_name is not None:
        sky_mask_name = frame_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        sky_mask_path = os.path.join(sky_masks_dir, sky_mask_name)
        cv2.imwrite(sky_mask_path, sky_mask)


def process_frame_batch(start_frame_idx, frame_names, step, video_dir, text, device,
                      processor, grounding_model, image_predictor, video_predictor,
                      inference_state, global_masks, objects_count,
                      mask_data_dir, json_data_dir, object_mask_dir, prompt_type, 
                      sky_masks_dir=None, generate_sky_masks=False,
                      reference_mask_dict=None, use_reference_for_frame_0=False,
                      original_frame_names=None):
    """Processes a batch of frames: detection, SAM prediction, video propagation, saving results.
    
    Args:
        original_frame_names: Optional list of original frame names (for when frame_names are converted/renamed)
    """
    print(f"Processing frames starting from index: {start_frame_idx}")
    img_path = os.path.join(video_dir, frame_names[start_frame_idx])
    image = Image.open(img_path)
    # Use original frame name for mask naming if provided, otherwise use current frame name
    name_for_mask = original_frame_names[start_frame_idx] if original_frame_names else frame_names[start_frame_idx]
    image_base_name = os.path.splitext(name_for_mask)[0]
    current_frame_masks = MaskDictionaryModel(promote_type=prompt_type, mask_name=f"mask_{image_base_name}.npy")
    
    # Use reference mask for frame 0 if specified
    if use_reference_for_frame_0 and start_frame_idx == 0 and reference_mask_dict is not None:
        print(f"Using reference mask for frame 0 to maintain object ID consistency")
        current_frame_masks = copy.deepcopy(reference_mask_dict)
        current_frame_masks.mask_name = f"mask_{image_base_name}.npy"
        
        # Update objects_count based on reference mask
        if current_frame_masks.labels:
            max_id = max(current_frame_masks.labels.keys())
            objects_count = max(objects_count, max_id)
        
        # Save the reference mask for frame 0
        save_frame_results(start_frame_idx, current_frame_masks, mask_data_dir, json_data_dir, object_mask_dir,
                          frame_name=frame_names[start_frame_idx])
        
        if generate_sky_masks and sky_masks_dir is not None:
            print(f"Generating sky mask for frame 0: {frame_names[start_frame_idx]}")
            sky_mask_0 = generate_sky_mask_sam2(image, image_predictor, processor, grounding_model, device)
            sky_mask_name = frame_names[start_frame_idx].replace('.jpg', '.png').replace('.jpeg', '.png')
            sky_mask_path = os.path.join(sky_masks_dir, sky_mask_name)
            cv2.imwrite(sky_mask_path, sky_mask_0)
        
        # Update global_masks with reference
        global_masks = copy.deepcopy(current_frame_masks)
        
        # If step is 1, return immediately (only process frame 0)
        if step == 1 or start_frame_idx + 1 >= len(frame_names):
            return global_masks, objects_count
        
        # Otherwise, continue with video propagation from frame 0
        start_frame_idx_for_propagation = start_frame_idx
    else:
        if generate_sky_masks and sky_masks_dir is not None:
            print(f"Generating sky mask for initial frame {start_frame_idx}: {frame_names[start_frame_idx]}")
            initial_sky_mask = generate_sky_mask_sam2(image, image_predictor, processor, grounding_model, device)
            initial_sky_mask_name = frame_names[start_frame_idx].replace('.jpg', '.png').replace('.jpeg', '.png')
            initial_sky_mask_path = os.path.join(sky_masks_dir, initial_sky_mask_name)
            cv2.imwrite(initial_sky_mask_path, initial_sky_mask)

        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=0.25, text_threshold=0.25, target_sizes=[image.size[::-1]]
        )
        input_boxes = results[0]["boxes"]
        labels = results[0]["labels"]

        if input_boxes.shape[0] > 0:
            image_predictor.set_image(np.array(image.convert("RGB")))
            masks, _, _ = image_predictor.predict(
                point_coords=None, point_labels=None, box=input_boxes, multimask_output=False,
            )
            if masks.ndim == 2:
                masks = masks[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)
            
            if current_frame_masks.promote_type == "mask":
                current_frame_masks.add_new_frame_annotation(
                    mask_list=torch.tensor(masks).to(device),
                    box_list=torch.tensor(input_boxes).to(device),
                    label_list=labels
                )
            else:
                raise NotImplementedError("SAM 2 video predictor currently only supports mask prompts")
            
            objects_count = current_frame_masks.update_masks(
                tracking_annotation_dict=global_masks, iou_threshold=0.8, objects_count=objects_count
            )
            print(f"Updated objects count: {objects_count}")
        else:
            print(f"No objects detected in frame {frame_names[start_frame_idx]}, merging with previous masks.")
            current_frame_masks = copy.deepcopy(global_masks)
        
        start_frame_idx_for_propagation = start_frame_idx

    if len(current_frame_masks.labels) == 0:
        print(f"No objects to track from frame {start_frame_idx}, saving empty data for next {step} frames.")
        end_frame_idx = min(start_frame_idx + step, len(frame_names))
        current_frame_masks.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list=frame_names[start_frame_idx:end_frame_idx])
        
        if generate_sky_masks and sky_masks_dir is not None:
            print(f"Generating sky masks for frames {start_frame_idx} to {end_frame_idx-1} (no objects detected)")
            for frame_idx in range(start_frame_idx + 1, end_frame_idx):
                frame_img_path = os.path.join(video_dir, frame_names[frame_idx])
                frame_image = Image.open(frame_img_path)
                frame_sky_mask = generate_sky_mask_sam2(frame_image, image_predictor, processor, grounding_model, device)
                frame_sky_mask_name = frame_names[frame_idx].replace('.jpg', '.png').replace('.jpeg', '.png')
                frame_sky_mask_path = os.path.join(sky_masks_dir, frame_sky_mask_name)
                cv2.imwrite(frame_sky_mask_path, frame_sky_mask)
        return global_masks, objects_count
    else:
        video_predictor.reset_state(inference_state)
        for object_id, object_info in current_frame_masks.labels.items():
            _, _, _ = video_predictor.add_new_mask(
                inference_state, start_frame_idx_for_propagation, object_id, object_info.mask,
            )
        
        video_segments = {}
        last_propagated_masks = None
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx_for_propagation):
            frame_masks = MaskDictionaryModel()
            if not out_obj_ids:
                continue
            
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0)
                class_name = current_frame_masks.get_target_class_name(out_obj_id)
                if class_name is None:
                    print(f"Warning: Could not find class name for object ID {out_obj_id} in frame {out_frame_idx}. Skipping.")
                    continue
                object_info = ObjectInfo(instance_id=out_obj_id, mask=out_mask[0], class_name=class_name)
                object_info.update_box()
                frame_masks.labels[out_obj_id] = object_info
            
            if frame_masks.labels:
                # Use original frame name for mask naming if provided
                name_for_mask = original_frame_names[out_frame_idx] if original_frame_names else frame_names[out_frame_idx]
                image_base_name = os.path.splitext(name_for_mask)[0]
                frame_masks.mask_name = f"mask_{image_base_name}.npy"
                first_mask = next(iter(frame_masks.labels.values())).mask
                frame_masks.mask_height = first_mask.shape[-2]
                frame_masks.mask_width = first_mask.shape[-1]
                video_segments[out_frame_idx] = frame_masks
                last_propagated_masks = copy.deepcopy(frame_masks)
        
        print(f"Propagated {len(video_segments)} frames.")
        
        for frame_idx, frame_masks_info in video_segments.items():
            frame_sky_mask = None
            if generate_sky_masks and sky_masks_dir is not None:
                # Use original frame name for display
                display_name = original_frame_names[frame_idx] if original_frame_names else frame_names[frame_idx]
                print(f"Generating sky mask for propagated frame {frame_idx}: {display_name}")
                frame_img_path = os.path.join(video_dir, frame_names[frame_idx])
                frame_image = Image.open(frame_img_path)
                frame_sky_mask = generate_sky_mask_sam2(frame_image, image_predictor, processor, grounding_model, device)
            
            # Use original frame name if provided
            fname_for_save = original_frame_names[frame_idx] if original_frame_names else frame_names[frame_idx]
            save_frame_results(frame_idx, frame_masks_info, mask_data_dir, json_data_dir, object_mask_dir,
                               sky_mask=frame_sky_mask, sky_masks_dir=sky_masks_dir, 
                               frame_name=fname_for_save)
        
        if last_propagated_masks:
            global_masks = last_propagated_masks
        else:
            global_masks = current_frame_masks
        
        return global_masks, objects_count

def generate_visualization(video_dir, mask_data_dir, json_data_dir, output_dir, vis_output_dir, output_video_path, frame_rate=15):
    """Draws masks and boxes on frames and saves the final annotated video."""
    print("Generating visualization...")
    CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, output_dir, vis_output_dir)
    create_video_from_images(vis_output_dir, output_video_path, frame_rate=frame_rate)
    print(f"Output video saved to: {output_video_path}")


def process_genimages_sequence(genimages_subfolder_path, reference_mask_path, output_base_dir,
                               device, video_predictor, image_predictor, processor, grounding_model,
                               text_prompt, step, prompt_type, generate_sky_masks, dataset_root_dir):
    """
    Process a genimages sequence (e.g., 00010_forward) with reference mask from GT frame.
    
    Args:
        dataset_root_dir: Root directory for shared masks (same as gtimages)
    """
    subfolder_name = os.path.basename(genimages_subfolder_path)
    print(f"\n{'='*80}")
    print(f"Processing genimages sequence: {subfolder_name}")
    print(f"Reference mask: {reference_mask_path}")
    print(f"{'='*80}\n")
    
    # Load reference mask
    reference_mask_dict = load_reference_mask(reference_mask_path)
    if reference_mask_dict is None:
        print(f"Error: Could not load reference mask. Skipping sequence {subfolder_name}")
        return
    
    # Setup output directories
    output_dir = os.path.join(output_base_dir, subfolder_name)
    mask_data_dir, json_data_dir, frame_names, sky_masks_dir, object_mask_dir, vis_object_mask_dir = \
        prepare_directories_and_frames(genimages_subfolder_path, output_dir, generate_sky_masks, 
                                      is_genimages=True, dataset_root_dir=dataset_root_dir)
    
    # Check if frames were found
    if not frame_names:
        print(f"Warning: No image frames found in {genimages_subfolder_path}, skipping")
        return
    
    print(f"Found {len(frame_names)} frames in {genimages_subfolder_path}")
    print(f"First frame: {frame_names[0] if frame_names else 'None'}")
    
    # Check file extension - SAM2's load_video_frames_from_jpg_images only supports JPG
    first_ext = os.path.splitext(frame_names[0])[-1].lower()
    temp_jpg_dir = None
    video_path_for_sam = genimages_subfolder_path
    original_frame_names = frame_names.copy()  # Keep original names for reference
    
    if first_ext == '.png':
        # SAM2 only supports JPG with numeric filenames, so convert PNG to JPG temporarily
        print(f"Converting PNG frames to JPG for SAM2 compatibility...")
        temp_jpg_dir = os.path.join(output_dir, "temp_jpg_frames")
        
        # Clean up temp directory if it already exists to avoid old files
        if os.path.exists(temp_jpg_dir):
            print(f"Cleaning up existing temp directory: {temp_jpg_dir}")
            shutil.rmtree(temp_jpg_dir)
        
        os.makedirs(temp_jpg_dir, exist_ok=True)
        
        jpg_frame_names = []
        frame_name_mapping = {}  # Map numeric names back to original names
        
        for idx, frame_name in enumerate(frame_names):
            png_path = os.path.join(genimages_subfolder_path, frame_name)
            # Use simple numeric filename that SAM2 expects (e.g., 00000.jpg, 00001.jpg)
            jpg_name = f"{idx:05d}.jpg"
            jpg_path = os.path.join(temp_jpg_dir, jpg_name)
            
            # Convert PNG to JPG
            img = Image.open(png_path)
            if img.mode == 'RGBA':
                # Convert RGBA to RGB
                img = img.convert('RGB')
            img.save(jpg_path, 'JPEG', quality=95)
            jpg_frame_names.append(jpg_name)
            frame_name_mapping[idx] = frame_name  # Store original name
        
        print(f"Converted {len(frame_names)} frames to JPG in {temp_jpg_dir}")
        
        # Verify the files in temp directory
        temp_files = sorted(os.listdir(temp_jpg_dir))
        print(f"Temp directory contains {len(temp_files)} files")
        print(f"First 5 files: {temp_files[:5]}")
        print(f"Last 5 files: {temp_files[-5:]}")
        
        video_path_for_sam = temp_jpg_dir
        frame_names = jpg_frame_names  # Use numeric JPG names for SAM2 processing
    
    # Initialize inference state
    inference_state = video_predictor.init_state(
        video_path=video_path_for_sam,
        offload_video_to_cpu=True,
        async_loading_frames=True
    )
    
    global_masks = MaskDictionaryModel(promote_type=prompt_type)
    objects_count = 0
    
    print(f"Total frames to process: {len(frame_names)}")
    
    # Process frame 0 with reference mask, then propagate
    # For PNG images, use temp_jpg_dir; otherwise use original path
    video_dir_to_process = video_path_for_sam if temp_jpg_dir else genimages_subfolder_path
    
    try:
        for start_frame_idx in range(0, len(frame_names), step):
            use_reference = (start_frame_idx == 0)  # Only use reference for first frame
            
            global_masks, objects_count = process_frame_batch(
                start_frame_idx, frame_names, step, video_dir_to_process, text_prompt, device,
                processor, grounding_model, image_predictor, video_predictor,
                inference_state, global_masks, objects_count,
                mask_data_dir, json_data_dir, object_mask_dir, prompt_type,
                sky_masks_dir, generate_sky_masks,
                reference_mask_dict=reference_mask_dict,
                use_reference_for_frame_0=use_reference,
                original_frame_names=original_frame_names if temp_jpg_dir else None
            )
        
        # Generate visualization
        # IMPORTANT: Use original genimages path for visualization, not temp JPG dir
        # Because mask files are named after original PNG files, not temp numeric JPG files
        output_video_path = os.path.join(output_dir, "output.mp4")
        generate_visualization(genimages_subfolder_path, mask_data_dir, json_data_dir, 
                              output_dir, vis_object_mask_dir, output_video_path)
        
        print(f"Completed processing {subfolder_name}")
    
    finally:
        # Clean up temporary JPG directory if created
        if temp_jpg_dir and os.path.exists(temp_jpg_dir):
            print(f"Cleaning up temporary JPG directory: {temp_jpg_dir}")
            shutil.rmtree(temp_jpg_dir)


def main(model_name='facebook/sam2-hiera-large', 
         grounding_model_id='IDEA-Research/grounding-dino-base', 
         text_prompt='car.', 
         images_root_dir='',  # Root images directory containing gtimages and genimages
         output_base_dir='./outputs', 
         step=20, 
         prompt_type='mask',
         generate_sky_masks=False,
         sky_only=False,
         process_mode='both'):  # 'gtimages', 'genimages', or 'both'

    device = setup_environment()
    video_predictor, image_predictor, processor, grounding_model = load_models(device, model_name, grounding_model_id)
    
    if not images_root_dir:
        raise ValueError("images_root_dir must be specified")
    
    gtimages_dir = os.path.join(images_root_dir, "gtimages")
    genimages_dir = os.path.join(images_root_dir, "genimages")
    
    # Get dataset root directory (parent of images_root_dir)
    # images_root_dir example: waymo_example/.../images
    # dataset_root_dir should be: waymo_example/.../
    dataset_root_dir = os.path.dirname(images_root_dir)
    
    # Process gtimages first (if requested)
    if process_mode in ['gtimages', 'both'] and os.path.exists(gtimages_dir):
        print(f"\n{'='*80}")
        print(f"PROCESSING GTIMAGES")
        print(f"{'='*80}\n")
        
        output_dir_gt = os.path.join(output_base_dir, "gtimages")
        
        if sky_only:
            generate_sky_masks = True
        
        mask_data_dir, json_data_dir, frame_names, sky_masks_dir, object_mask_dir, vis_object_mask_dir = \
            prepare_directories_and_frames(gtimages_dir, output_dir_gt, generate_sky_masks, 
                                          is_genimages=False, dataset_root_dir=dataset_root_dir)

        if sky_only:
            print("Sky-only mode enabled: Generating sky masks for all GT frames and exiting.")
            process_sky_only_batch(
                frame_names, gtimages_dir, device, processor, grounding_model, 
                image_predictor, sky_masks_dir
            )
            print("Sky mask generation complete for GT images.")
            return

        print(f"Processing GT video for text prompt: '{text_prompt}'")
        if generate_sky_masks:
            print("Sky mask generation is enabled alongside object tracking.")

        inference_state = video_predictor.init_state(
            video_path=gtimages_dir, 
            offload_video_to_cpu=True, 
            async_loading_frames=True
        )
        global_masks = MaskDictionaryModel(promote_type=prompt_type)
        objects_count = 0

        print(f"Total GT frames to process: {len(frame_names)}")
        for start_frame_idx in range(0, len(frame_names), step):
            global_masks, objects_count = process_frame_batch(
                start_frame_idx, frame_names, step, gtimages_dir, text_prompt, device,
                processor, grounding_model, image_predictor, video_predictor,
                inference_state, global_masks, objects_count,
                mask_data_dir, json_data_dir, object_mask_dir, prompt_type, 
                sky_masks_dir, generate_sky_masks
            )

        output_video_path = os.path.join(output_dir_gt, "output.mp4")
        generate_visualization(gtimages_dir, mask_data_dir, json_data_dir, 
                             output_dir_gt, vis_object_mask_dir, output_video_path)
        
        print(f"Completed processing GT images")
    
    # Process genimages sequences (if requested)
    if process_mode in ['genimages', 'both'] and os.path.exists(genimages_dir):
        print(f"\n{'='*80}")
        print(f"PROCESSING GENIMAGES")
        print(f"{'='*80}\n")
        
        # Get all genimages subfolders
        genimages_subfolders = [
            f for f in os.listdir(genimages_dir)
            if os.path.isdir(os.path.join(genimages_dir, f))
        ]
        genimages_subfolders.sort()
        
        print(f"Found {len(genimages_subfolders)} genimages sequences: {genimages_subfolders}")
        
        # Get gtimages object_mask directory from dataset root
        gt_object_mask_dir = os.path.join(dataset_root_dir, "object_mask")
        
        if not os.path.exists(gt_object_mask_dir):
            print(f"Warning: GT object_mask directory not found at {gt_object_mask_dir}")
            print("Please process gtimages first to generate reference masks.")
            return
        
        output_genimages_base = os.path.join(output_base_dir, "genimages")
        
        for subfolder in genimages_subfolders:
            # Extract reference frame number (e.g., "00010_forward" -> "00010")
            match = re.match(r'(\d+)_', subfolder)
            if not match:
                print(f"Warning: Could not extract frame number from {subfolder}, skipping")
                continue
            
            ref_frame_num = match.group(1)
            reference_mask_path = os.path.join(gt_object_mask_dir, f"{ref_frame_num}.png")
            
            if not os.path.exists(reference_mask_path):
                print(f"Warning: Reference mask not found at {reference_mask_path}, skipping {subfolder}")
                continue
            
            genimages_subfolder_path = os.path.join(genimages_dir, subfolder)
            
            process_genimages_sequence(
                genimages_subfolder_path, reference_mask_path, output_genimages_base,
                device, video_predictor, image_predictor, processor, grounding_model,
                text_prompt, step, prompt_type, generate_sky_masks, dataset_root_dir
            )
        
        print(f"\nCompleted processing all genimages sequences")
    
    # Print completion message
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"All masks saved to: {dataset_root_dir}")
    print(f"  - object_mask/: Contains masks from both gtimages and genimages")
    print(f"  - vis_object_mask/: Contains visualization masks")
    print(f"  - sky_masks/: Contains sky masks")
    print(f"\nProcessing outputs saved to: {output_base_dir}")
    print(f"  - gtimages/: GT processing results (mask_data, json_data, output.mp4)")
    print(f"  - genimages/: Generated images processing results")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate dynamic masks using GroundingSAM2 for gtimages and genimages')
    parser.add_argument('--model_name', type=str, default='facebook/sam2-hiera-large', help='Path to SAM2 model config')
    parser.add_argument('--grounding_model_id', type=str, default='IDEA-Research/grounding-dino-base', help='ID of the Grounding DINO model')
    parser.add_argument('--text_prompt', type=str, default='car.', help='Text prompt for Grounding DINO')
    parser.add_argument('--images_root_dir', type=str, required=True, help='Root images directory containing gtimages and genimages folders')
    parser.add_argument('--output_base_dir', type=str, default='./outputs', help='Base output directory')
    parser.add_argument('--generate_sky_masks', action='store_true', help='Generate sky masks using SAM2')
    parser.add_argument('--sky_only', action='store_true', help='Generate only sky masks for all frames (skip object detection)')
    parser.add_argument('--process_mode', type=str, default='both', choices=['gtimages', 'genimages', 'both'],
                       help='Which images to process: gtimages, genimages, or both')
    args = parser.parse_args()
    
    main(
        model_name=args.model_name,
        grounding_model_id=args.grounding_model_id,
        text_prompt=args.text_prompt,
        images_root_dir=args.images_root_dir,
        output_base_dir=args.output_base_dir,
        generate_sky_masks=args.generate_sky_masks,
        sky_only=args.sky_only,
        process_mode=args.process_mode
    )
