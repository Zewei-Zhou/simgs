#!/bin/bash

# Set the path to the scene directory
SCENE_DIR=$1
scene_dataset=$(basename "$SCENE_DIR")

# Iterate through all scene folders in the specified directory
for scene in "$SCENE_DIR"/*; do
  if [ -d "$scene" ]; then
    echo "Training scene: $scene"
    scene_name=$(basename "$scene")
    CUDA_VISIBLE_DEVICES=0 python train.py \
      --config config/objectgs/2d/$scene_dataset/config.yaml \
      --scene_name $scene_name
  fi
done