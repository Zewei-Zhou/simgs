# Heuristic dynamic masks category
MASK_PROMPT="road.sidewalk.building.house.fence.tree.car.truck.bicycle.person.rider.traffic light.traffic sign.street light.mountain.hill.traffic cone."
# MASK_PROMPT="objects"

# Generate dynamic masks from SAM2 (recommended)
python preprocess/tools/generate_mask_sam2.py \
--text_prompt "$MASK_PROMPT" \
--video_dir $1/images \
--output_dir $1/tmp \
--generate_sky_masks
# --sky_only 

# Generate dynamic masks from DEVA (deprecated)
# cd submodules/vid2sim-deva-segmentation
# bash generate_mask.sh $1 $MASK_PROMPTc