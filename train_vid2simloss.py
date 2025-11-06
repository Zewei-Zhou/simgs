import os
import shutil
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

import torch
import torchvision
import json
import wandb
import time
from datetime import datetime
from os import makedirs
import shutil
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim, patch_based_ncc_loss, angular_loss, geo_consist_loss

try:
    from fused_ssim import fused_ssim
    USE_FUSED_SSIM = True
except ImportError:
    USE_FUSED_SSIM = False
    print("fused_ssim not available, using standard ssim")
import sys
from gaussian_renderer import network_gui
from scene import Scene
from utils.general_utils import get_expon_lr_func, safe_state, parse_cfg, visualize_depth, render_normal
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, save_rgba
from argparse import ArgumentParser, Namespace
import yaml
import torch.nn.functional as F
import warnings
from render import render_sets
from modules.sky_model import create_sky_model
from modules.sky_losses import SkyLossManager
warnings.filterwarnings('ignore')

lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

# Focal Loss for object segmentation
class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight, ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)

        if self.reduction == 'mean':
            mask = (target != self.ignore_index)
            if mask.sum() > 0:
                return focal_loss[mask].mean()
            else:
                return torch.tensor(0.0, device=input.device)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

# Class to convert object IDs to RGB colors
class ID2RGBConverter:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.color_map = self._generate_color_map()

    def _generate_color_map(self):
        # Generate 256 random RGB colors, each color is a tuple (0-255 range)
        return np.random.randint(0, 256, size=(256, 3), dtype=np.uint8)

    def convert(self, obj: int):
        if obj == 0:
            return 0, np.array([0, 0, 0], dtype=np.uint8)  # Predefine class 0 as black
        if 0 <= obj <= 255:
            return obj, self.color_map[obj]  # Get color from the fixed color map
        else:
            raise ValueError("ID out of range, should be between 0 and 255")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    assert os.path.exists(os.path.join(ROOT, '.gitignore'))
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = Path(__file__).resolve().parent

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')

def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    modules = __import__('scene')
    model_config = dataset.model_config
    gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
    scene = Scene(dataset, gaussians, shuffle=pipe.shuffle, logger=logger, weed_ratio=pipe.weed_ratio)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    # -----------------------Sky Model Integration-----------------------------------
    # Extract the number of images from the dataset configuration
    train_cameras = scene.getTrainCameras()
    n_images = len(train_cameras)
    max_uid = max([cam.uid for cam in train_cameras]) + 1  # +1 because UIDs are 0-indexed
    
    # Use the maximum value to ensure all UIDs are valid indices
    n_embeddings = max(n_images, max_uid)
    
    print(f"[DEBUG Sky Model Init] n_cameras={n_images}, max_uid={max_uid-1}, n_embeddings={n_embeddings}")
    
    # simple sky model
    sky_model = create_sky_model(
        model_type="neural",
        n_images=n_embeddings,  # Use n_embeddings to accommodate all camera UIDs
        head_mlp_layer_width=64,  # Default MLP layer width
        enable_appearance_embedding=True,  # Enable appearance embedding
        appearance_embedding_dim=16,  # Default embedding dimension
        device=torch.device("cuda")  # Use CUDA device
    )
    # environment light model
    # sky_model = create_sky_model(
    #     model_type="envlight",  
    #     resolution=1024,
    #     device=torch.device("cuda")
    # )
    sky_loss_manager = SkyLossManager({
        "opacity_loss_type": "safe_bce",
        "opacity_loss_weight": opt.lambda_sky_opa,
        "regularization_weight": 0.01
    })
    sky_params = sky_model.get_param_groups()
    # Use higher learning rate for sky model to ensure it learns quickly
    optimizer_sky = torch.optim.Adam([{'params': list(sky_model.parameters()), 'lr': 0.005, 'name': 'sky_model'}])
    # -------------------------------------------------------------------------------
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    
    # Vid2Sim loss weight functions
    normal_loss_weight = get_expon_lr_func(getattr(opt, 'normal_l1_weight_init', 0.1), getattr(opt, 'normal_l1_weight_final', 0.01), max_steps=opt.iterations)
    geo_loss_weight = get_expon_lr_func(getattr(opt, 'geo_consist_weight_init', 0.1), getattr(opt, 'geo_consist_weight_final', 0.01), max_steps=opt.iterations)
    
    viewpoint_stack = None
    
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    densify_cnt = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    modules = __import__('gaussian_renderer')
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.add_prefilter, keep_alive = network_gui.receive()
                if custom_cam != None:
                    net_image = getattr(modules, 'render')(custom_cam, gaussians, pipe, scene.background)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        
        # Pick a random Camera
        
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if gaussians.explicit_gs:
            gaussians.set_gs_mask(viewpoint_cam.camera_center, viewpoint_cam.resolution_scale)
            visible_mask = gaussians._gs_mask
        else:
            gaussians.set_anchor_mask(viewpoint_cam.camera_center, viewpoint_cam.resolution_scale)
            from gaussian_renderer.render import prefilter_voxel
            visible_mask = prefilter_voxel(viewpoint_cam, gaussians).squeeze() if pipe.add_prefilter else gaussians._anchor_mask    

        # -----------------------Sky Render-----------------------------------
        # render_pkg = getattr(modules, 'render_with_sky')(viewpoint_cam, gaussians, pipe, scene.background, visible_mask, training=True, object_mask=None, sky_model=sky_model)
        render_pkg = getattr(modules, 'render_with_sky')(viewpoint_cam, gaussians, pipe, scene.background.cuda(), visible_mask, training=True, object_mask=None, sky_model=sky_model)
        #render_pkg = getattr(modules, 'render')(viewpoint_cam, gaussians, pipe, scene.background, visible_mask)
        image, scaling, alpha, semantics = render_pkg["render"], render_pkg["scaling"], render_pkg["render_alphas"], render_pkg["render_semantics"]

        gt_image = viewpoint_cam.original_image.cuda()
        alpha_mask = viewpoint_cam.alpha_mask.cuda()
        if hasattr(viewpoint_cam, 'validity_mask') and viewpoint_cam.validity_mask is not None:
            validity_mask = viewpoint_cam.validity_mask.cuda()
        else:
            validity_mask = torch.ones_like(alpha_mask)
        final_loss_mask = alpha_mask * validity_mask
        
        # Get sky mask for sky region loss
        has_sky_mask = hasattr(viewpoint_cam, 'sky_mask') and viewpoint_cam.sky_mask is not None
        if has_sky_mask:
            sky_mask = viewpoint_cam.sky_mask.cuda()  # 1 for sky, 0 for non-sky
        else:
            sky_mask = torch.zeros_like(alpha_mask)

        losses = dict()

        # Photometric loss - separate for foreground and sky
        gt_mask = alpha_mask.float()  # Use alpha mask as ground truth mask
        
        # Foreground loss (masked region)
        masked_image = image * gt_mask
        masked_gt_image = gt_image * gt_mask
        
        Ll1 = l1_loss(masked_image, masked_gt_image)
        
        if USE_FUSED_SSIM:
            ssim_val = fused_ssim(masked_image.unsqueeze(0).float(), masked_gt_image.unsqueeze(0).float())
            ssim_loss = 1.0 - ssim_val
        else:
            ssim_loss = (1.0 - ssim(masked_image, masked_gt_image))
            
        losses["image_loss"] = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        
        # Sky loss - only compute if we have sky mask and sky regions exist
        if has_sky_mask and sky_mask.sum() > 100:  # At least 100 sky pixels
            # Sky region: where sky_mask=1 and alpha_mask=0 (background)
            sky_region_mask = sky_mask * (1.0 - alpha_mask) * validity_mask
            if sky_region_mask.sum() > 100:
                sky_image = image * sky_region_mask
                sky_gt_image = gt_image * sky_region_mask
                sky_l1 = l1_loss(sky_image, sky_gt_image)
                # Weight sky loss lower than foreground initially
                sky_loss_weight = getattr(opt, 'lambda_sky_recon', 0.5)
                losses["sky_recon_loss"] = sky_loss_weight * sky_l1

        # Scaling loss (Vid2Sim style)
        if opt.lambda_dreg > 0:
            if scaling.shape[0] > 0:
                # Vid2Sim style: minimum scale loss for visible gaussians
                if hasattr(render_pkg, 'visibility_filter') and render_pkg["visibility_filter"].sum() > 0:
                    visible_scaling = scaling[render_pkg["visibility_filter"]]
                    sorted_scale, _ = torch.sort(visible_scaling, dim=-1)
                    min_scale_loss = sorted_scale[..., 0]
                    scaling_reg = min_scale_loss.mean()
                else:
                    scaling_reg = scaling.prod(dim=1).mean()
            else:
                scaling_reg = torch.tensor(0.0, device="cuda")
            losses["scaling_loss"] = opt.lambda_dreg * scaling_reg

        # Object loss - Enhanced with debugging and fixes
        if opt.lambda_object_loss > 0:
            try:
                # Get ground truth object IDs and convert to indices
                gt_object_ids = gaussians.id_encoder.label_to_index(viewpoint_cam.object_mask.cuda()).long()
                
                # Debug information (log occasionally)
                if iteration % 1000 == 0:
                    unique_gt_labels = torch.unique(viewpoint_cam.object_mask.cuda())
                    unique_gt_indices = torch.unique(gt_object_ids)
                    num_classes = semantics.shape[-1]
                    
                    if logger:
                        logger.info(f"[ITER {iteration}] Object Loss Debug:")
                        logger.info(f"  GT unique labels: {unique_gt_labels.tolist()}")
                        logger.info(f"  GT unique indices: {unique_gt_indices.tolist()}")
                        logger.info(f"  Semantics shape: {semantics.shape}")
                        logger.info(f"  Num classes: {num_classes}")
                        logger.info(f"  GT object mask shape: {viewpoint_cam.object_mask.shape}")
                        logger.info(f"  Alpha mask shape: {alpha_mask.shape}")
                
                # Ensure semantics has the correct shape [H, W, num_classes]
                if len(semantics.shape) == 4:  # [1, H, W, num_classes]
                    semantics_2d = semantics.squeeze(0)  # [H, W, num_classes]
                else:  # [H, W, num_classes]
                    semantics_2d = semantics
                
                # Apply alpha mask to focus on valid regions
                valid_mask = final_loss_mask.squeeze().bool()  # [H, W]
                
                if valid_mask.sum() > 0:
                    # Get valid regions only - use proper 2D indexing
                    valid_semantics = semantics_2d[valid_mask]  # [N_valid, num_classes]
                    valid_gt_ids = gt_object_ids[valid_mask]     # [N_valid]
                    
                    # Class balancing - compute class weights for object classes only
                    non_bg_gt_ids = valid_gt_ids[valid_gt_ids > 0]
                    
                    class_weights = torch.ones(semantics_2d.shape[-1], device=valid_gt_ids.device)
                    class_weights[0] = 0 # background weight is not used due to ignore_index, but set to 0 for clarity.

                    if non_bg_gt_ids.numel() > 0:
                        unique_classes, class_counts = torch.unique(non_bg_gt_ids, return_counts=True)
                        total_object_pixels = non_bg_gt_ids.numel()
                        num_object_classes = len(unique_classes)

                        # Create class weights (inverse frequency) for object classes
                        for cls, count in zip(unique_classes, class_counts):
                            if cls > 0 and cls < semantics_2d.shape[-1]:
                                weight = total_object_pixels / (count.float() * num_object_classes)
                                class_weights[cls] = weight
                    
                    # Cap maximum weight to prevent extreme values
                    class_weights = torch.clamp(class_weights, max=10.0)
                    
                    # Use weighted CrossEntropyLoss or FocalLoss, ignoring background class 0
                    object_loss_params = getattr(opt, 'object_loss_params', {})
                    loss_type = object_loss_params.get('loss_type', 'cross_entropy')

                    if loss_type == 'focal_loss':
                        gamma = object_loss_params.get('focal_loss_gamma', 2.0)
                        object_criterion = FocalLoss(
                            weight=class_weights,
                            gamma=gamma,
                            ignore_index=0,
                            reduction='mean'
                        )
                    else: # 'cross_entropy' is the default
                        object_criterion = torch.nn.CrossEntropyLoss(
                            weight=class_weights,
                            ignore_index=0,
                            reduction='mean'
                        )
                    
                    # Compute loss on valid regions
                    object_loss = object_criterion(valid_semantics, valid_gt_ids)
                    
                    # Progressive loss weight (start small, increase gradually)
                    if iteration < getattr(opt, 'object_loss_warmup_iter', 2000):
                        warmup_factor = min(1.0, iteration / getattr(opt, 'object_loss_warmup_iter', 2000))
                        object_weight = opt.lambda_object_loss * warmup_factor
                    else:
                        object_weight = opt.lambda_object_loss
                    
                    losses["object_loss"] = object_weight * object_loss
                    
                    # Log detailed loss information
                    if iteration % 1000 == 0 and logger:
                        logger.info(f"  Object loss: {object_loss.item():.6f}")
                        logger.info(f"  Object weight: {object_weight:.6f}")
                        logger.info(f"  Class weights: {class_weights.tolist()}")
                        logger.info(f"  Valid pixels: {valid_mask.sum().item()}/{valid_mask.numel()}")
                        logger.info(f"  Valid semantics shape: {valid_semantics.shape}")
                        logger.info(f"  Valid GT IDs shape: {valid_gt_ids.shape}")
                
                else:
                    # No valid regions, set object loss to zero
                    losses["object_loss"] = torch.tensor(0.0, device="cuda")
                    if iteration % 1000 == 0 and logger:
                        logger.warning(f"[ITER {iteration}] No valid pixels for object loss computation")
                
                # Zero penalty (encourage non-background predictions in object regions)
                if opt.lambda_zero_penalty > 0:
                    # Focus on non-background regions in GT
                    non_bg_mask = (gt_object_ids > 0) & valid_mask
                    if non_bg_mask.sum() > 0:
                        prob_zero_class = semantics_2d[..., 0][non_bg_mask]  
                        losses["zero_penalty"] = opt.lambda_zero_penalty * prob_zero_class.mean()
                    else:
                        losses["zero_penalty"] = torch.tensor(0.0, device="cuda")
                
            except Exception as e:
                # Fallback to simple loss if there are issues
                if logger:
                    logger.error(f"[ITER {iteration}] Object loss computation error: {str(e)}")
                    # Additional debug info for the error
                    logger.error(f"  Semantics shape: {semantics.shape}")
                    logger.error(f"  Alpha mask shape: {alpha_mask.shape}")
                    logger.error(f"  GT object IDs shape: {gt_object_ids.shape}")
                
                # Simple fallback loss using the original approach but with correct dimensions
                try:
                    # Ensure proper shape for CrossEntropyLoss
                    if len(semantics.shape) == 4:  # [1, H, W, num_classes]
                        semantics_for_loss = semantics.permute(0, 3, 1, 2)  # [1, num_classes, H, W]
                        gt_ids_for_loss = gt_object_ids.unsqueeze(0)  # [1, H, W]
                    else:  # [H, W, num_classes]
                        semantics_for_loss = semantics.permute(2, 0, 1).unsqueeze(0)  # [1, num_classes, H, W]
                        gt_ids_for_loss = gt_object_ids.unsqueeze(0)  # [1, H, W]
                    
                    object_loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='mean')(
                        semantics_for_loss, gt_ids_for_loss
                    )
                    losses["object_loss"] = opt.lambda_object_loss * 0.1 * object_loss  # Reduced weight as fallback
                    
                    if logger:
                        logger.info(f"[ITER {iteration}] Using fallback object loss: {object_loss.item():.6f}")
                        
                except Exception as fallback_e:
                    if logger:
                        logger.error(f"[ITER {iteration}] Fallback object loss also failed: {str(fallback_e)}")
                    losses["object_loss"] = torch.tensor(0.0, device="cuda")  # Last resort

        #--------------sky loss (original, to be checked)---------------
        # # Sky opacity loss - Enhanced with DriveStudio approach
        # if opt.lambda_sky_opa > 0:
        #     # Check if we have sky masks available
        #     if hasattr(viewpoint_cam, 'sky_mask') and viewpoint_cam.sky_mask is not None:
        #         # Use DriveStudio approach with sky masks
        #         sky_masks = viewpoint_cam.sky_mask.squeeze().cuda()  # 1 for sky, 0 for non-sky
        #         gt_occupied_mask = (1.0 - sky_masks).float()  # Convert to occupied mask
        #         pred_occupied_mask = alpha.squeeze()
                
        #         # Safe binary cross entropy loss
        #         pred_occupied_mask = torch.clamp(pred_occupied_mask, 1e-6, 1-1e-6)
        #         loss_sky_opa = F.binary_cross_entropy(pred_occupied_mask, gt_occupied_mask, reduction="mean")
        #     else:
        #         # Fallback to original ObjectGS approach
        #         o = alpha.clamp(1e-6, 1-1e-6)
        #         sky = alpha_mask.float()
        #         loss_sky_opa = (-(1-sky) * torch.log(1 - o)).mean()
            
        #     losses["sky_opa_loss"] = opt.lambda_sky_opa * loss_sky_opa
        #-------------------------------------------------------------
        #--------------sky loss (original, to be checked)---------------
        if opt.lambda_sky_opa > 0 and hasattr(viewpoint_cam, 'sky_mask') and viewpoint_cam.sky_mask is not None:
            # sky_mask is already on cuda from line 239
            sky_losses = sky_loss_manager.compute_losses(
                pred_opacity=render_pkg["render_alphas"],
                sky_masks=sky_mask.squeeze(),
                alpha_masks=alpha_mask.squeeze(),
                validity_mask=validity_mask.squeeze()
            )
            for loss_name, loss_value in sky_losses.items():
                losses[loss_name] = loss_value
        #-------------------------------------------------------------
                 
        # Opacity entropy loss
        if opt.lambda_opacity_entropy > 0:
            o = alpha.clamp(1e-6, 1 - 1e-6)
            loss_opacity_entropy = -(o*torch.log(o)).mean()
            losses["opacity_entropy_loss"] = opt.lambda_opacity_entropy * loss_opacity_entropy

        # Normal loss
        if opt.lambda_normal > 0 and iteration > opt.normal_start_iter:
            assert gaussians.render_mode=="RGB+ED" or gaussians.render_mode=="RGB+D"
            normals = render_pkg["render_normals"].squeeze(0).permute((2, 0, 1))
            normals_from_depth = render_pkg["render_normals_from_depth"] * render_pkg["render_alphas"].permute((1, 2, 0)).detach()
            if len(normals_from_depth.shape) == 4:
                normals_from_depth = normals_from_depth.squeeze(0)
            normals_from_depth = normals_from_depth.permute((2, 0, 1))
            normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
            losses["normal_loss"] = opt.lambda_normal * (normal_error * final_loss_mask).mean()

        # Distortion loss
        if opt.lambda_dist and iteration > opt.dist_start_iter:
            losses["distort_loss"] = opt.lambda_dist * (render_pkg["render_distort"].squeeze(3) * final_loss_mask).mean()
        
        # Depth loss
        if iteration > opt.start_depth and depth_l1_weight(iteration) > 0 and viewpoint_cam.invdepthmap is not None:
            assert gaussians.render_mode=="RGB+ED" or gaussians.render_mode=="RGB+D"
            render_depth = render_pkg["render_depth"]
            invDepth = torch.where(render_depth > 0.0, 1.0 / render_depth, torch.zeros_like(render_depth))            
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            final_depth_mask = depth_mask * validity_mask
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * final_depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            losses["depth_loss"] = Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # Vid2Sim geometric supervision losses
        vid2sim_geo_start = getattr(opt, 'geo_sup_from_iter', 1000)
        vid2sim_geo_end = getattr(opt, 'geo_sup_until_iter', opt.iterations)
        
        if (iteration > vid2sim_geo_start and iteration < vid2sim_geo_end and 
            hasattr(viewpoint_cam, 'gt_depth') and viewpoint_cam.gt_depth is not None):
            
            try:
                # Get ground truth depth and prepare masks
                gt_inv_depth = viewpoint_cam.gt_depth.to("cuda").float()
                gt_depth = 1.0 / (gt_inv_depth + 1e-9)
                gt_depth = torch.clip(gt_depth, 0.0, 150.0)
                
                # Get rendered depth
                if "render_depth" in render_pkg:
                    render_depth = render_pkg["render_depth"]
                    depth = torch.clip(render_depth, 0.0, 150.0)
                    depth = depth.squeeze(0)
                    inv_depth = 1.0 / (depth + 1e-9)
                    
                    # Create valid depth mask
                    inf_depth_mask = gt_inv_depth != 0
                    
                    # Compute ground truth normals from depth
                    gt_normal = render_normal(viewpoint_cam, gt_depth)
                    
                    # Get rendered normals
                    if "render_normals" in render_pkg:
                        normals = render_pkg["render_normals"].squeeze(0)
                        if normals.shape[0] == 3:  # Ensure correct shape (3, H, W)
                            normals = normals
                        else:
                            normals = normals.permute(2, 0, 1)
                        
                        # Prepare combined mask
                        combined_mask = final_loss_mask.float() * inf_depth_mask
                        
                        # Only proceed if we have valid masked region
                        if combined_mask.sum() > 100:  # At least 100 valid pixels
                            
                            # 1. NCC-based depth loss
                            if getattr(opt, 'use_ncc_depth_loss', True):
                                vid2sim_depth_weight = getattr(opt, 'vid2sim_depth_weight', 0.1)
                                if vid2sim_depth_weight > 0:
                                    ncc_depth_loss = patch_based_ncc_loss(inv_depth, gt_inv_depth, combined_mask)
                                    losses["vid2sim_depth_loss"] = vid2sim_depth_weight * normal_loss_weight(iteration) * ncc_depth_loss
                            
                            # 2. Angular normal loss
                            if getattr(opt, 'use_angular_normal_loss', True):
                                vid2sim_normal_weight = getattr(opt, 'vid2sim_normal_weight', 0.1)
                                if vid2sim_normal_weight > 0:
                                    angular_normal_loss = angular_loss(normals, gt_normal, combined_mask)
                                    losses["vid2sim_normal_loss"] = vid2sim_normal_weight * normal_loss_weight(iteration) * angular_normal_loss
                            
                            # 3. Geometric consistency loss
                            if getattr(opt, 'use_geo_consist_loss', True):
                                vid2sim_geo_weight = getattr(opt, 'vid2sim_geo_weight', 0.1)
                                if vid2sim_geo_weight > 0:
                                    geo_consistency_loss = geo_consist_loss(normals, gt_depth, combined_mask)
                                    losses["vid2sim_geo_consist_loss"] = vid2sim_geo_weight * geo_loss_weight(iteration) * geo_consistency_loss
                                    
                        elif iteration % 1000 == 0:  # Log warning every 1000 iterations
                            if logger:
                                logger.warning(f"Insufficient valid depth mask pixels at iteration {iteration}: {combined_mask.sum()}")
                                
            except Exception as e:
                if iteration % 1000 == 0:  # Log error every 1000 iterations to avoid spam
                    if logger:
                        logger.error(f"Error in Vid2Sim geometric supervision at iteration {iteration}: {str(e)}")
                    else:
                        print(f"Error in Vid2Sim geometric supervision at iteration {iteration}: {str(e)}")
                        
        elif iteration == vid2sim_geo_start and logger:
            # Log when geometric supervision starts
            if hasattr(viewpoint_cam, 'gt_depth') and viewpoint_cam.gt_depth is not None:
                logger.info(f"Starting Vid2Sim geometric supervision at iteration {iteration}")
            else:
                logger.warning(f"Vid2Sim geometric supervision requested but no gt_depth available at iteration {iteration}")
    
        total_loss = sum(losses.values())
        total_loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                psnr_log = psnr(image, gt_image).mean().double()
                anchor_prim = len(gaussians.get_anchor)
                
                # Enhanced progress bar with Vid2Sim loss info
                progress_info = {
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",
                    "psnr": f"{psnr_log:.{3}f}",
                    "GS_num": f"{anchor_prim}",
                    "prefilter": f"{pipe.add_prefilter}"
                }
                
                # Add object loss info if available
                if "object_loss" in losses and opt.lambda_object_loss > 0:
                    obj_loss_val = losses["object_loss"].item() if isinstance(losses["object_loss"], torch.Tensor) else losses["object_loss"]
                    progress_info["Obj Loss"] = f"{obj_loss_val:.{4}f}"
                
                # Add Vid2Sim loss info if active
                vid2sim_geo_start = getattr(opt, 'geo_sup_from_iter', 1000)
                vid2sim_geo_end = getattr(opt, 'geo_sup_until_iter', opt.iterations)
                if vid2sim_geo_start <= iteration < vid2sim_geo_end:
                    progress_info["Vid2Sim"] = "ON"
                
                progress_bar.set_postfix(progress_info)
                progress_bar.update(10)
                
                # Log detailed loss breakdown every 100 iterations
                if iteration % 100 == 0:
                    loss_details = []
                    for name, loss_val in losses.items():
                        if isinstance(loss_val, torch.Tensor):
                            loss_details.append(f"{name}: {loss_val.item():.6f}")
                        else:
                            loss_details.append(f"{name}: {loss_val:.6f}")
                    # if logger:
                        # logger.info(f"[ITER {iteration}] Loss breakdown: {', '.join(loss_details)}")
                    
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, losses, total_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, getattr(modules, 'render_with_sky'), (pipe, scene.background), wandb, logger)
            
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration % pipe.vis_step == 0 or iteration == 1 or (iteration % 100 == 0 and iteration < 1000):
                viewpoint_cam = scene.getTrainCameras().copy()[10]

                if gaussians.explicit_gs:
                    gaussians.set_gs_mask(viewpoint_cam.camera_center, viewpoint_cam.resolution_scale)
                    visible_mask = gaussians._gs_mask
                else:
                    gaussians.set_anchor_mask(viewpoint_cam.camera_center, viewpoint_cam.resolution_scale)
                    from gaussian_renderer.render import prefilter_voxel
                    visible_mask = prefilter_voxel(viewpoint_cam, gaussians).squeeze() if pipe.add_prefilter else gaussians._anchor_mask    

                vis_render_pkg = getattr(modules, 'render_with_sky')(viewpoint_cam, gaussians, pipe, scene.background, visible_mask, sky_model = sky_model)
                vis_image, alpha = vis_render_pkg["render"], vis_render_pkg["render_alphas"]
                gt_image = viewpoint_cam.original_image.cuda()
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                vis_image = vis_image * alpha_mask
                gt_image = gt_image * alpha_mask         

                other_img = []
                resolution = (int(viewpoint_cam.image_width/1.0), int(viewpoint_cam.image_height/1.0))
                vis_img = F.interpolate(vis_image.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
                # vis_gt_img = F.interpolate(gt_image.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
                # vis_alpha = F.interpolate(alpha.repeat(3, 1, 1).unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]


                if iteration > opt.start_depth and viewpoint_cam.invdepthmap is not None:
                    vis_depth = visualize_depth(invDepth) 
                    gt_depth = visualize_depth(mono_invdepth)
                    vis_depth = F.interpolate(vis_depth.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
                    vis_gt_depth = F.interpolate(gt_depth.unsqueeze(0), size=(resolution[1], resolution[0]), mode='bilinear', align_corners=False)[0]
                    other_img.append(vis_depth)
                    other_img.append(vis_gt_depth)
                
                grid = torchvision.utils.make_grid([
                    vis_img, 
                    # vis_gt_img, 
                    # vis_alpha,
                ] + other_img, nrow=1)

                vis_path = os.path.join(scene.model_path, "vis")
                os.makedirs(vis_path, exist_ok=True)
                torchvision.utils.save_image(grid, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}.png"))


            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(opt, render_pkg, image.shape[2], image.shape[1])
                densify_cnt += 1 

                # densification
                if opt.densification and iteration > opt.update_from and densify_cnt > 0 and densify_cnt % opt.update_interval == 0:
                    if dataset.pretrained_checkpoint != "":
                        gaussians.roll_back()
                    gaussians.run_densify(opt, iteration)
            
            elif iteration == opt.update_until:
                if dataset.pretrained_checkpoint != "":
                    gaussians.roll_back()
                gaussians.clean()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                #--------------sky model optimizer step----------------
                optimizer_sky.step()
                #-----------------------------------------------------
                gaussians.optimizer.zero_grad(set_to_none = True)
                #--------------sky model optimizer step----------------
                optimizer_sky.zero_grad()
                #-----------------------------------------------------

            if iteration >= opt.iterations - pipe.no_prefilter_step:
                pipe.add_prefilter = False

            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, losses, total_loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        for key, value in losses.items():
            tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/{key}', value, iteration)
        tb_writer.add_scalar(f'{dataset_name}/total_loss', total_loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    if wandb is not None:
        wandb.log({f"{dataset_name}_loss_patches_{key}":value.item() for key, value in losses.items()})
        wandb.log({f"{dataset_name}_total_loss":total_loss.item()})
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx] for idx in range(0, len(scene.getTrainCameras()), 100)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                cnt = 0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    alpha_mask = viewpoint.alpha_mask.cuda()
                    if hasattr(viewpoint, 'validity_mask') and viewpoint.validity_mask is not None:
                        validity_mask = viewpoint.validity_mask.cuda()
                    else:
                        validity_mask = torch.ones_like(alpha_mask)
                    final_eval_mask = alpha_mask * validity_mask
                    image = image * final_eval_mask
                    gt_image = gt_image * final_eval_mask
                    
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        if wandb:
                            gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    cnt += 1 
                
                l1_test /= cnt
                psnr_test /= cnt    
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
         
        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', len(scene.gaussians.get_anchor), iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        render_image = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        render_mask = tf.to_tensor(render).unsqueeze(0)[:, 3:4, :, :].cuda()
        render_image = render_image * render_mask
        gt_image = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        gt_mask = tf.to_tensor(gt).unsqueeze(0)[:, 3:4, :, :].cuda()
        gt_image = gt_image * gt_mask
        renders.append(render_image)
        gts.append(gt_image)
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, eval_name, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / eval_name

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        base_method_dir = test_dir / method
        method_dir = base_method_dir 
        if os.path.exists(method_dir):
            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

            logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
            logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
            logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
            logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
            logger.info("  GS_NUMS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(visible_count).float().mean(), ".5"))
            print("")
            
            full_dict[scene_dir][method].update({
                "PSNR": torch.tensor(psnrs).mean().item(),
                "SSIM": torch.tensor(ssims).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
                "GS_NUMS": torch.tensor(visible_count).float().mean().item(),
                })

            per_view_dict[scene_dir][method].update({
                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                "GS_NUMS": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}
                })

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path')
    parser.add_argument('--scene_name', type=str, help='Override scene name in config', default=None)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[80000,90000,100000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[80000,90000,100000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.scene_name is not None:
        try:
            cfg["model_params"]["exp_name"] = os.path.join(cfg["model_params"]["exp_name"], args.scene_name)
            cfg["model_params"]["source_path"] = os.path.join(cfg["model_params"]["source_path"], args.scene_name)
        except:
            print("OverrideError: Cannot override 'exp_name' and 'source_path' in 'model_params'. Exiting.")
            sys.exit(1)

    lp, op, pp = parse_cfg(cfg)
    args.save_iterations.append(op.iterations)

    # enable logging
    cur_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    lp.model_path = os.path.join("outputs", lp.dataset_name, lp.exp_name, cur_time)
    os.makedirs(lp.model_path, exist_ok=True)
    shutil.copy(args.config, os.path.join(lp.model_path, "config.yaml"))

    logger = get_logger(lp.model_path)

    if args.test_iterations[0] == -1:
        args.test_iterations = [i for i in range(10000, op.iterations + 1, 10000)]
        # args.test_iterations = [i for i in range(5000, op.iterations + 1, 5000)]
    if len(args.test_iterations) == 0 or args.test_iterations[-1] != op.iterations:
        args.test_iterations.append(op.iterations)

    if args.save_iterations[0] == -1:
        args.save_iterations = [i for i in range(10000, op.iterations + 1, 10000)]
        # args.save_iterations = [i for i in range(5000, op.iterations + 1, 5000)]
    if len(args.save_iterations) == 0 or args.save_iterations[-1] != op.iterations:
        args.save_iterations.append(op.iterations)

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    # try:
    #     saveRuntimeCode(os.path.join(lp.model_path, 'backup'))
    # except:
    #     logger.info(f'save code failed~')
    
    exp_name = lp.exp_name if lp.dataset_name=="" else lp.dataset_name+"_"+lp.exp_name
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Horizon-GS",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + lp.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp, op, pp, exp_name, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, wandb, logger)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    if lp.eval:
        visible_count = render_sets(lp, op, pp, -1, skip_train=True, skip_test=False, logger=logger)
    else:
        visible_count = render_sets(lp, op, pp, -1, skip_train=False, skip_test=True, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    eval_name = 'test' if lp.eval else 'train'
    evaluate(lp.model_path, eval_name, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
