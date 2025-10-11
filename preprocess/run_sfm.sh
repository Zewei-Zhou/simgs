#!/bin/bash

# Usage: bash run_sfm.sh <seq_path>
set -e

if [ $# -lt 1 ]; then
  echo "Usage: bash run_sfm.sh <seq_path>"
  exit 1
fi

SEQ_PATH=$1

# Default values (edit if needed)
NO_GPU_FLAG=""                         # Use "--no_gpu" to disable GPU
SKIP_MATCHING_FLAG=""                  # Use "--skip_matching" to skip matching
SINGLE_CAMERA_FLAG=""                  # Use "--single_camera" to assume same camera for all images
MASK_PATH="$SEQ_PATH/masks"           # Default to <seq_path>/masks if not overridden
CAMERA="OPENCV"
COLMAP_EXECUTABLE=""                  # Default: use "colmap" in PATH
GLOMAP_EXECUTABLE=""                  # Default: use "glomap" in PATH
MAGICK_EXECUTABLE=""                 # Default: use "magick" in PATH
RESIZE_FLAG=""                        # Use "--resize" to enable resizing

# Allow overrides using environment variables
if [ ! -z "$NO_GPU" ]; then
  NO_GPU_FLAG="--no_gpu"
fi

if [ ! -z "$SKIP_MATCHING" ]; then
  SKIP_MATCHING_FLAG="--skip_matching"
fi

if [ ! -z "$SINGLE_CAMERA" ]; then
  SINGLE_CAMERA_FLAG="--single_camera"
fi

if [ ! -z "$CUSTOM_MASK_PATH" ]; then
  MASK_PATH="$CUSTOM_MASK_PATH"
fi

if [ ! -z "$CUSTOM_CAMERA" ]; then
  CAMERA="$CUSTOM_CAMERA"
fi

if [ ! -z "$COLMAP" ]; then
  COLMAP_EXECUTABLE="$COLMAP"
fi

if [ ! -z "$GLOMAP" ]; then
  GLOMAP_EXECUTABLE="$GLOMAP"
fi

if [ ! -z "$MAGICK" ]; then
  MAGICK_EXECUTABLE="$MAGICK"
fi

if [ ! -z "$RESIZE" ]; then
  RESIZE_FLAG="--resize"
fi

# Construct and run the command
python3 preprocess/tools/run_sfm.py \
  --source_path "$SEQ_PATH" \
  --mask_path "$MASK_PATH" \
  --camera "$CAMERA" \
  ${NO_GPU_FLAG} \
  ${SKIP_MATCHING_FLAG} \
  ${SINGLE_CAMERA_FLAG} \
  ${COLMAP_EXECUTABLE:+--colmap_executable "$COLMAP_EXECUTABLE"} \
  ${GLOMAP_EXECUTABLE:+--glomap_executable "$GLOMAP_EXECUTABLE"} \
  ${MAGICK_EXECUTABLE:+--magick_executable "$MAGICK_EXECUTABLE"} \
  ${RESIZE_FLAG}
