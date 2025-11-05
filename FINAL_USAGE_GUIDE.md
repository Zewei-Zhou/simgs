# ObjectGS with Generated Images - Complete Usage Guide

## Dataset Structure

```
your_dataset/
├── images/
│   ├── gtimages/                    # Ground truth images
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   ├── 00010.jpg
│   │   └── ...
│   └── genimages/                   # Generated sequences
│       ├── 00010_forward/           # Generated from GT frame 10
│       │   ├── 00010_forward_000000.png
│       │   ├── 00010_forward_000001.png
│       │   └── ...
│       └── ...
│
├── object_mask/                     # Flat! All object masks here
│   ├── 00000.png                    # GT masks
│   ├── 00010.png
│   ├── 00010_forward_000000.png     # Generated sequence masks
│   └── ...
│
├── depths/                          # (Optional) Flat!
├── sky_masks/                       # (Optional) Flat!
│
└── sparse/0/
    ├── cameras.bin                  # Contains ALL images (GT + generated)
    ├── images.bin                   # Contains ALL images (GT + generated)
    └── points3D_corr.ply            # (or points3D.bin)
```

## Complete Workflow

### Step 1: Generate Object Masks

#### 1a. Generate GT Image Masks First

```bash
cd preprocess/tools

python generate_mask_sam2.py \
    --images_root_dir /path/to/dataset/images \
    --output_base_dir /path/to/dataset/tmp \
    --text_prompt "car. vehicle." \
    --process_mode gtimages
```

**Output:**
- Creates `object_mask/00000.png`, `00001.png`, `00010.png`, etc.
- These serve as reference masks for generated sequences

#### 1b. Generate Masks for Generated Sequences

```bash
python generate_mask_sam2.py \
    --images_root_dir /path/to/dataset/images \
    --output_base_dir /path/to/dataset/tmp \
    --text_prompt "car. vehicle." \
    --process_mode genimages
```

**What happens:**
- For `genimages/00010_forward/`:
  - Loads GT mask from `object_mask/00010.png`
  - Applies to frame 0 of generated sequence
  - Propagates tracking to maintain object IDs
  - Saves as `object_mask/00010_forward_000000.png`, etc.

**Or process both at once:**
```bash
python generate_mask_sam2.py \
    --images_root_dir /path/to/dataset/images \
    --output_base_dir /path/to/dataset/tmp \
    --text_prompt "car. vehicle." \
    --process_mode both
```

### Step 2: Update COLMAP Files

Add entries for generated images to `sparse/0/images.bin` and `sparse/0/cameras.bin`.

**Option A: Manual** - Use COLMAP tools or Python scripts

**Option B: Copy from GT** - If generated images have same camera parameters as GT frame:

```python
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, write_extrinsics_binary
import os
import copy

# Read existing COLMAP data
cam_extrinsics = read_extrinsics_binary("sparse/0/images.bin")
cam_intrinsics = read_intrinsics_binary("sparse/0/cameras.bin")

# For each generated sequence
ref_frame_name = "00010.jpg"  # GT frame
ref_extr = cam_extrinsics[ref_frame_name]

# Add entries for generated frames
for i in range(100):  # Assuming 100 generated frames
    gen_frame_name = f"00010_forward_{i:06d}.png"
    gen_extr = copy.deepcopy(ref_extr)
    gen_extr.name = gen_frame_name
    # Assign unique ID
    gen_extr.id = max([e.id for e in cam_extrinsics.values()]) + 1
    cam_extrinsics[gen_frame_name] = gen_extr

# Write back
write_extrinsics_binary(cam_extrinsics, "sparse/0/images.bin")
```

### Step 3: (Optional) Preprocess Point Cloud

If you need labeled point cloud:

```bash
python ply_preprocessing.py \
    --data_path /path/to/dataset \
    --output_path /path/to/dataset/sparse/0/points3D_corr.ply
```

This will:
- Assign object labels to 3D points
- **Filter out invalid pixels from generated images** (using validity masks if available)
- Save labeled point cloud

### Step 4: Train

```bash
python train.py --config configs/waymo_config.yaml
# or
python train_vid2simloss.py --config configs/waymo_config.yaml
```

**Config example:**
```yaml
model_params:
  source_path: "/path/to/dataset"
  model_path: "/path/to/output"
  data_format: "colmap"
  eval: true
  llffhold: 32
  images: "images"
  masks: "object_mask"
  depths: "depths"
  add_mask: true
  add_depth: true
```

## How It Works

### Data Loading Pipeline

```
1. scan_all_images_in_folders()
   ├─> Scans images/gtimages/*.jpg
   ├─> Scans images/genimages/*/*.png
   └─> Returns image_map: {filename -> full_path}

2. readColmapCameras()
   ├─> For each COLMAP entry:
   │   ├─> Lookup image path from image_map
   │   ├─> Load image
   │   └─> Load mask from object_mask/
   └─> Split into gtimages_cams and genimages_cams

3. readColmapSceneInfo()
   ├─> Train = GT_train + ALL_genimages
   └─> Test = GT_test
```

### Mask Loading

```
Image Path                                  → Mask Path
───────────────────────────────────────────────────────
images/gtimages/00010.jpg                   → object_mask/00010.png
images/genimages/00010_forward/...000000.png → object_mask/00010_forward_000000.png
```

The `Camera` class automatically:
1. Extracts image basename
2. Finds dataset root
3. Constructs mask path in flat `object_mask/` directory
4. Loads corresponding mask

### Object ID Consistency

```
GT Frame 10:
  object_mask/00010.png
  └─> Objects: {1: car1, 2: car2, 3: pedestrian}

Generated Sequence (from frame 10):
  object_mask/00010_forward_000000.png
  └─> Objects: {1: car1, 2: car2, 3: pedestrian}  ← Same IDs!
  
  object_mask/00010_forward_000001.png
  └─> Objects: {1: car1, 2: car2, 3: pedestrian}  ← Tracked consistently
```

## Verification

### Check Mask Generation

```bash
# Verify GT masks generated
ls -lh object_mask/00*.png | head

# Verify generated sequence masks
ls -lh object_mask/*_forward_*.png | head

# Check object ID consistency
python -c "
import cv2
import numpy as np

gt_mask = cv2.imread('object_mask/00010.png', cv2.IMREAD_UNCHANGED)
gen_mask = cv2.imread('object_mask/00010_forward_000000.png', cv2.IMREAD_UNCHANGED)

print(f'GT objects: {np.unique(gt_mask)}')
print(f'Gen frame 0 objects: {np.unique(gen_mask)}')
print(f'Match: {np.array_equal(np.unique(gt_mask), np.unique(gen_mask))}')
"
```

### Check Data Loading

```bash
# Verify images found
python -c "
import sys
sys.path.insert(0, 'scene')
from dataset_readers import scan_all_images_in_folders

image_map = scan_all_images_in_folders('images')
print(f'Total images: {len(image_map)}')

gt_imgs = [k for k, v in image_map.items() if 'gtimages' in v]
gen_imgs = [k for k, v in image_map.items() if 'genimages' in v]

print(f'GT images: {len(gt_imgs)}')
print(f'Generated images: {len(gen_imgs)}')
"
```

## Training Logs to Check

When training starts, verify:

```
Scanning images in .../images...
Found X images in total
Loaded A gtimages images and B genimages images.
Final splitting result: C training images (D GT + B Gen), E testing images.
```

**Expected:**
- `B` = Number of generated images (from all sequences)
- `C` = `D` (GT training) + `B` (all generated)
- `E` = GT test images only

## Troubleshooting

### Issue: "Reference mask not found" during mask generation

**Cause:** GT masks not generated yet

**Solution:**
```bash
# Process GT first
python generate_mask_sam2.py --process_mode gtimages ...
# Then generated
python generate_mask_sam2.py --process_mode genimages ...
```

### Issue: "Image not found in scanned images"

**Cause:** COLMAP has entries but images missing

**Solution:**
- Check image filenames in `images.bin` match actual files
- Ensure COLMAP entries added for all generated images

### Issue: Object mask loading error

**Cause:** Mask path construction failed

**Solution:**
- Verify `object_mask/` at dataset root
- Check mask filenames match image basenames
- For `images/genimages/00010_forward/00010_forward_000000.png`:
  - Mask should be `object_mask/00010_forward_000000.png`

### Issue: Object IDs inconsistent

**Cause:** 
1. Different text prompts used for GT vs generated
2. Reference mask not applied correctly

**Solution:**
- Use identical `--text_prompt` for both GT and generated
- Verify frame 0 of generated sequence matches GT reference

## Best Practices

1. **Process masks in order:** GT → Generated
2. **Consistent prompts:** Same text prompt for all mask generation
3. **Verify references:** Check GT mask exists before processing generated sequence
4. **Check COLMAP:** Ensure all images have COLMAP entries
5. **Validate IDs:** Verify object IDs match between GT and generated frame 0

## File Modifications Summary

### Modified Files:
1. ✅ `scene/dataset_readers.py`
   - `scan_all_images_in_folders()`: Scans `gtimages/` and `genimages/`
   - `get_genimages_mask_path()`: (Currently unused, kept for compatibility)
   - `readColmapSceneInfo()`: Splits GT/generated for train/test

2. ✅ `scene/cameras.py`
   - Object mask loading handles nested image paths

3. ✅ `preprocess/tools/generate_mask_sam2.py`
   - Processes GT and generated sequences
   - Maintains object ID consistency via reference masks

### Unchanged Files:
- ❌ `train.py` - No changes needed!
- ❌ `train_vid2simloss.py` - No changes needed!

## Quick Start Example

```bash
# 1. Generate masks
cd preprocess/tools
python generate_mask_sam2.py \
    --images_root_dir ~/dataset/images \
    --output_base_dir ~/dataset/tmp \
    --text_prompt "car. vehicle. pedestrian." \
    --process_mode both

# 2. (Optional) Update COLMAP
# ... add generated image entries ...

# 3. (Optional) Preprocess point cloud
python ply_preprocessing.py \
    --data_path ~/dataset \
    --output_path ~/dataset/sparse/0/points3D_corr.ply

# 4. Train!
cd ../..
python train.py --config configs/my_config.yaml
```

Done! Your ObjectGS training now includes generated images with consistent object IDs.

