#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt, mask=None):
    if mask is not None:
        return (torch.abs(network_output - gt) * mask).mean()
    return torch.abs(network_output - gt).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ncc_loss(pred, gt, mask):
    """
    Compute Normalized Cross-Correlation (NCC) loss over a masked region.

    Args:
        pred (torch.Tensor): predicted map, shape (H, W)
        gt (torch.Tensor): ground truth map, shape (H, W)
        mask (torch.Tensor): binary mask, shape (H, W)

    Returns:
        torch.Tensor: scalar NCC loss (1 - NCC)
    """
    eps = 1e-5

    # Ensure all tensors are float32 for precision
    pred = pred.float()
    gt = gt.float()
    mask = mask.float()

    # Apply mask
    masked_pred = pred * mask
    masked_gt = gt * mask

    # Mean over masked region
    mask_sum = torch.sum(mask) + eps
    mean_pred = torch.sum(masked_pred) / mask_sum
    mean_gt = torch.sum(masked_gt) / mask_sum

    # Centered values
    pred_centered = masked_pred - mean_pred
    gt_centered = masked_gt - mean_gt

    # Variance with masking
    var_pred = torch.sum((pred_centered ** 2) * mask) / mask_sum
    var_gt = torch.sum((gt_centered ** 2) * mask) / mask_sum

    # Clamp variance to avoid division by zero
    std_pred = torch.sqrt(var_pred.clamp(min=eps))
    std_gt = torch.sqrt(var_gt.clamp(min=eps))

    # Compute NCC
    ncc_num = torch.sum(pred_centered * gt_centered * mask)
    ncc_den = std_pred * std_gt * mask_sum
    ncc = ncc_num / (ncc_den + eps)

    # Return NCC loss
    return 1.0 - ncc

def patch_based_ncc_loss(pred, gt, mask, patch_size=3, stride=1):
    H, W = pred.shape
    # Add batch and channel dimensions: (1, 1, H, W)
    pred = pred * mask 
    gt = gt * mask

    pred = pred.float().unsqueeze(0).unsqueeze(0)
    gt = gt.float().unsqueeze(0).unsqueeze(0)
    mask = mask.float().unsqueeze(0).unsqueeze(0)

    unfold = torch.nn.Unfold(kernel_size=patch_size, padding=patch_size // 2, stride=stride)
    pred_patches = unfold(pred)[0].permute(1,0)  # Shape: (1, K*K, h*w)
    gt_patches = unfold(gt)[0].permute(1,0)      # Shape: (1, K*K, h*w)

    # Subtract the mean from predictions and ground truth
    pred_centered = pred_patches - pred_patches.mean()
    gt_centered = gt_patches - gt_patches.mean()

    # Calculate standard deviations of centered predictions and ground truth
    pred_std = torch.sqrt(torch.mean(pred_centered ** 2, dim=1))
    gt_std = torch.sqrt(torch.mean(gt_centered ** 2, dim=1))

    # Calculate the NCC
    ncc = torch.sum(pred_centered * gt_centered, dim=1) / (pred_std * gt_std + 1e-5)
    ncc_loss = 1 - (ncc / patch_size**2)
    return ncc_loss.mean()


def angular_loss(pred_normals, gt_normals, mask=None):
    """
    Computes the angular loss between predicted and ground truth normals.
    
    Args:
        pred_normals (torch.Tensor): Predicted normals of shape (3, H, W).
        gt_normals (torch.Tensor): Ground truth normals of shape (3, H, W).
        mask (torch.Tensor, optional): Binary mask of shape (H, W). Defaults to None.
    
    Returns:
        torch.Tensor: Scalar loss value.
    """
    pred_normals = F.normalize(pred_normals, p=2, dim=0)
    gt_normals = F.normalize(gt_normals, p=2, dim=0)

    cosine_similarity = torch.sum(pred_normals * gt_normals, dim=0)  # Shape: (H, W)
    cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
    loss = 1 - cosine_similarity  # Shape: (H, W)
    
    if mask is not None:
        loss = loss * mask  # Apply mask to the loss
    
    if mask is not None:
        loss = loss.sum() / (mask.sum() + 1e-8)
    else:
        loss = loss.mean()
    
    return loss

def shift_right(pred_normals):
    pred_normals_right = F.pad(pred_normals[:, :, :-1], (1, 0), mode='replicate')  # (3, H, W)
    return pred_normals_right

def shift_down(pred_normals):
    pred_normals_down = F.pad(pred_normals[:, :-1, :], (0, 0, 1, 0), mode='replicate')  # (3, H, W)
    return pred_normals_down

def geo_consist_loss(pred_normals, depth, mask=None):
    pred_normals = F.normalize(pred_normals, p=2, dim=0)
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=depth.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=depth.device).unsqueeze(0).unsqueeze(0)
    depth = depth.unsqueeze(0).unsqueeze(0)
    
    grad_x = F.conv2d(depth, sobel_x, padding=1)
    grad_y = F.conv2d(depth, sobel_y, padding=1)
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    
    epsilon = 1e-6 
    grad_min = grad_magnitude.min()
    grad_max = grad_magnitude.max()
    grad_magnitude_normalized = 1.0 - (grad_magnitude - grad_min) / (grad_max - grad_min + epsilon)
    grad_magnitude_normalized = grad_magnitude_normalized.squeeze(0).squeeze(0)  # Shape: (H, W)
    
    if mask is not None:
        grad_magnitude_normalized = grad_magnitude_normalized * mask  # Shape: (H, W)
   
    pred_normals_right = shift_right(pred_normals)
    pred_normals_down = shift_down(pred_normals)
    dot_right = torch.sum(pred_normals * pred_normals_right, dim=0)
    dot_down = torch.sum(pred_normals * pred_normals_down, dim=0)
    
    dot_right = torch.clamp(dot_right, -1.0, 1.0)
    dot_down = torch.clamp(dot_down, -1.0, 1.0)
    
    angle_right = 1 - dot_right  # Shape: (H, W)
    angle_down = 1 - dot_down    # Shape: (H, W)
    angle_loss = (angle_right + angle_down) / 2.0  # Shape: (H, W)
    weighted_loss = angle_loss * grad_magnitude_normalized  # Shape: (H, W)
    loss = weighted_loss.sum() / (grad_magnitude_normalized.sum() + epsilon)
    
    return loss
