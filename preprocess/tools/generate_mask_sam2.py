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

def prepare_directories_and_frames(video_dir, output_dir, generate_sky_masks=False):
    """Creates output directories and returns sorted frame names."""
    CommonUtils.creat_dirs(output_dir)
    mask_data_dir = os.path.join(output_dir, "mask_data")
    json_data_dir = os.path.join(output_dir, "json_data")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)

    parent_dir = os.path.dirname(output_dir)
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
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split("_")[-1]))
    return mask_data_dir, json_data_dir, frame_names, sky_masks_dir, object_mask_dir, vis_object_mask_dir

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
                      sky_masks_dir=None, generate_sky_masks=False):
    """Processes a batch of frames: detection, SAM prediction, video propagation, saving results."""
    print(f"Processing frames starting from index: {start_frame_idx}")
    img_path = os.path.join(video_dir, frame_names[start_frame_idx])
    image = Image.open(img_path)
    image_base_name = os.path.splitext(frame_names[start_frame_idx])[0]
    current_frame_masks = MaskDictionaryModel(promote_type=prompt_type, mask_name=f"mask_{image_base_name}.npy")
    
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
                inference_state, start_frame_idx, object_id, object_info.mask,
            )
        
        video_segments = {}
        last_propagated_masks = None
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
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
                image_base_name = os.path.splitext(frame_names[out_frame_idx])[0]
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
                print(f"Generating sky mask for propagated frame {frame_idx}: {frame_names[frame_idx]}")
                frame_img_path = os.path.join(video_dir, frame_names[frame_idx])
                frame_image = Image.open(frame_img_path)
                frame_sky_mask = generate_sky_mask_sam2(frame_image, image_predictor, processor, grounding_model, device)
            
            save_frame_results(frame_idx, frame_masks_info, mask_data_dir, json_data_dir, object_mask_dir,
                               sky_mask=frame_sky_mask, sky_masks_dir=sky_masks_dir, 
                               frame_name=frame_names[frame_idx])
        
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


def main(model_name='facebook/sam2-hiera-large', 
         grounding_model_id='IDEA-Research/grounding-dino-base', 
         text_prompt='car.', 
         video_dir='notebooks/videos/car', 
         output_dir='./outputs', 
         step=20, 
         prompt_type='mask',
         generate_sky_masks=False,
         sky_only=False):

    device = setup_environment()
    video_predictor, image_predictor, processor, grounding_model = load_models(device, model_name, grounding_model_id)
    
    if sky_only:
        generate_sky_masks = True
    
    mask_data_dir, json_data_dir, frame_names, sky_masks_dir, object_mask_dir, vis_object_mask_dir = prepare_directories_and_frames(video_dir, output_dir, generate_sky_masks)

    if sky_only:
        print("Sky-only mode enabled: Generating sky masks for all frames and exiting.")
        process_sky_only_batch(
            frame_names, video_dir, device, processor, grounding_model, 
            image_predictor, sky_masks_dir
        )
        print("Sky mask generation complete.")
        return

    print(f"Processing video for text prompt: '{text_prompt}'")
    if generate_sky_masks:
        print("Sky mask generation is enabled alongside object tracking.")

    inference_state = video_predictor.init_state(
        video_path=video_dir, 
        offload_video_to_cpu=True, 
        async_loading_frames=True
    )
    global_masks = MaskDictionaryModel(promote_type=prompt_type)
    objects_count = 0

    print(f"Total frames to process: {len(frame_names)}")
    for start_frame_idx in range(0, len(frame_names), step):
        global_masks, objects_count = process_frame_batch(
            start_frame_idx, frame_names, step, video_dir, text_prompt, device,
            processor, grounding_model, image_predictor, video_predictor,
            inference_state, global_masks, objects_count,
            mask_data_dir, json_data_dir, object_mask_dir, prompt_type, 
            sky_masks_dir, generate_sky_masks
        )

    output_video_path = os.path.join(output_dir, "output.mp4")
    generate_visualization(video_dir, mask_data_dir, json_data_dir, output_dir, vis_object_mask_dir, output_video_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate dynamic masks using GroundingSAM2')
    parser.add_argument('--model_name', type=str, default='facebook/sam2-hiera-large', help='Path to SAM2 model config')
    parser.add_argument('--grounding_model_id', type=str, default='IDEA-Research/grounding-dino-base', help='ID of the Grounding DINO model')
    parser.add_argument('--text_prompt', type=str, default='car.', help='Text prompt for Grounding DINO')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to video directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Path to temporary output directory (e.g., ./tmp)')
    parser.add_argument('--generate_sky_masks', action='store_true', help='Generate sky masks using SAM2')
    parser.add_argument('--sky_only', action='store_true', help='Generate only sky masks for all frames (skip object detection)')
    args = parser.parse_args()
    
    main(
        model_name=args.model_name,
        grounding_model_id=args.grounding_model_id,
        text_prompt=args.text_prompt,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        generate_sky_masks=args.generate_sky_masks,
        sky_only=args.sky_only
    )