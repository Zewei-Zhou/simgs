#!/usr/bin/env python3
"""
Generate sky masks for ObjectGS from images.
This is a simple sky detection script for demonstration purposes.
For production use, consider using more sophisticated sky segmentation models.
"""

import os
import cv2
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm


def simple_sky_detection(image_path, output_path):
    """
    Simple sky detection based on color and position heuristics.
    This is a basic implementation - for better results, use deep learning models.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return
        
    height, width = img.shape[:2]
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create mask for sky regions
    sky_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Sky is typically in the upper part of the image
    upper_region = hsv[:height//3, :]  # Top third of the image
    
    # Sky color ranges (adjust these based on your data)
    # Blue sky
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(upper_region, lower_blue, upper_blue)
    
    # Gray/white sky (cloudy)
    lower_gray = np.array([0, 0, 150])
    upper_gray = np.array([180, 30, 255])
    gray_mask = cv2.inRange(upper_region, lower_gray, upper_gray)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(blue_mask, gray_mask)
    
    # Apply to sky mask (only upper region)
    sky_mask[:height//3, :] = combined_mask
    
    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
    
    # Fill connected components from the top edge
    # This helps ensure sky regions are properly connected
    h, w = sky_mask.shape
    flood_mask = np.zeros((h+2, w+2), np.uint8)
    
    # Start flood fill from top edge pixels that are already marked as sky
    for x in range(0, w, 10):  # Sample every 10 pixels
        if sky_mask[0, x] > 0:
            cv2.floodFill(sky_mask, flood_mask, (x, 0), 255)
    
    # Save the mask
    Image.fromarray(sky_mask).save(output_path)


def generate_sky_masks_for_dataset(dataset_path):
    """Generate sky masks for the entire dataset."""
    images_dir = os.path.join(dataset_path, "images")
    sky_masks_dir = os.path.join(dataset_path, "sky_masks")
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
        
    # Create sky masks directory
    os.makedirs(sky_masks_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return
        
    print(f"Processing {len(image_files)} images...")
    
    for image_file in tqdm(image_files, desc="Generating sky masks"):
        image_path = os.path.join(images_dir, image_file)
        
        # Generate output filename (change extension to .png)
        mask_file = os.path.splitext(image_file)[0] + '.png'
        mask_path = os.path.join(sky_masks_dir, mask_file)
        
        try:
            simple_sky_detection(image_path, mask_path)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            
    print(f"Sky masks saved to: {sky_masks_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate sky masks for ObjectGS dataset")
    parser.add_argument("dataset_path", help="Path to the dataset directory containing 'images' folder")
    parser.add_argument("--single", help="Process single image file", default=None)
    parser.add_argument("--output", help="Output path for single image processing", default=None)
    
    args = parser.parse_args()
    
    if args.single:
        if not args.output:
            args.output = args.single.replace('.jpg', '_sky_mask.png').replace('.jpeg', '_sky_mask.png')
        simple_sky_detection(args.single, args.output)
        print(f"Sky mask saved to: {args.output}")
    else:
        generate_sky_masks_for_dataset(args.dataset_path)


if __name__ == "__main__":
    main()
