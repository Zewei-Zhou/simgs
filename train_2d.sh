#!/bin/bash

# Set the path to the scene directory
SCENE_DIR=$1
scene_dataset=$(basename "$(dirname "$SCENE_DIR")")
scene_name=$(basename "$SCENE_DIR")

if [ -d "$SCENE_DIR" ]; then
  echo "Training scene: $SCENE_DIR"
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --config config/objectgs/2d/$scene_dataset/config.yaml \
    --scene_name $scene_name
else
  echo "Error: Directory not found at $SCENE_DIR"
  exit 1
fi