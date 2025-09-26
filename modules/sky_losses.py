"""
Sky-related loss functions for ObjectGS.
Based on OmniRe/DriveStudio implementation.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any


def binary_cross_entropy(pred: torch.Tensor, gt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Binary cross entropy loss."""
    return F.binary_cross_entropy(pred, gt, reduction=reduction)


def safe_binary_cross_entropy(pred: torch.Tensor, gt: torch.Tensor, 
                            limit: float = 0.1, reduction: str = "mean") -> torch.Tensor:
    """
    Safe binary cross entropy loss with clamping to avoid numerical instability.
    
    Args:
        pred: Predicted values
        gt: Ground truth values
        limit: Clamping limit to avoid log(0)
        reduction: Reduction method
    """
    pred = torch.clamp(pred, limit, 1.0 - limit)
    return F.binary_cross_entropy(pred, gt, reduction=reduction)


def sky_opacity_loss(pred_opacity: torch.Tensor, sky_masks: torch.Tensor, 
                    loss_type: str = "bce", limit: float = 0.1) -> torch.Tensor:
    """
    Sky opacity loss to ensure proper sky region handling.
    
    Args:
        pred_opacity: Predicted opacity values, shape (...,)
        sky_masks: Sky mask indicating sky regions (1 for sky, 0 for non-sky), shape (...,)
        loss_type: Type of loss ('bce' or 'safe_bce')
        limit: Limit for safe BCE
        
    Returns:
        Sky opacity loss
    """
    # Convert sky masks to occupied masks (0 for sky, 1 for occupied)
    gt_occupied_mask = (1.0 - sky_masks).float()
    pred_occupied_mask = pred_opacity.squeeze()
    
    if loss_type == "bce":
        return binary_cross_entropy(pred_occupied_mask, gt_occupied_mask, reduction="mean")
    elif loss_type == "safe_bce":
        return safe_binary_cross_entropy(pred_occupied_mask, gt_occupied_mask, 
                                       limit=limit, reduction="mean")
    else:
        raise ValueError(f"Unknown sky opacity loss type: {loss_type}")


def sky_regularization_loss(alpha: torch.Tensor, sky_masks: torch.Tensor) -> torch.Tensor:
    """
    Regularization loss to encourage low opacity in sky regions.
    This is the original ObjectGS sky loss implementation.
    
    Args:
        alpha: Alpha/opacity values
        sky_masks: Sky masks (1 for non-sky, 0 for sky in alpha_mask format)
        
    Returns:
        Sky regularization loss
    """
    o = alpha.clamp(1e-6, 1-1e-6)
    sky = sky_masks.float()  # sky_masks here should be alpha_mask format
    loss_sky_opa = (-(1-sky) * torch.log(1 - o)).mean()
    return loss_sky_opa


class SkyLossManager:
    """Manager for sky-related losses."""
    
    def __init__(self, loss_config: Dict[str, Any]):
        self.config = loss_config
        self.opacity_loss_type = loss_config.get("opacity_loss_type", "safe_bce")
        self.opacity_loss_weight = loss_config.get("opacity_loss_weight", 0.05)
        self.regularization_weight = loss_config.get("regularization_weight", 0.0)
        
    def compute_losses(self, pred_opacity: torch.Tensor, sky_masks: torch.Tensor, 
                      alpha_masks: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute all sky-related losses.
        
        Args:
            pred_opacity: Predicted opacity values
            sky_masks: Sky masks (1 for sky, 0 for non-sky)
            alpha_masks: Alpha masks for regularization (1 for non-sky, 0 for sky)
            
        Returns:
            Dictionary of loss terms
        """
        losses = {}
        
        # Sky opacity loss (main loss from DriveStudio)
        if self.opacity_loss_weight > 0:
            opacity_loss = sky_opacity_loss(
                pred_opacity, sky_masks, 
                loss_type=self.opacity_loss_type
            )
            losses["sky_opacity_loss"] = self.opacity_loss_weight * opacity_loss
            
        # Sky regularization loss (original ObjectGS implementation)
        if self.regularization_weight > 0 and alpha_masks is not None:
            reg_loss = sky_regularization_loss(pred_opacity, alpha_masks)
            losses["sky_regularization_loss"] = self.regularization_weight * reg_loss
            
        return losses
