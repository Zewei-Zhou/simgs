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

def _erode_sky_mask(sky_masks: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    """
    Morphological erosion on sky mask to get a conservative 'sky core' region.
    sky_masks: (H,W), (B,H,W) or (B,H,W,1) with 1=sky, 0=non-sky
    Return: (B,H,W,1)
    """
    # Handle different input dimensions
    if sky_masks.dim() == 2:  # (H,W)
        sky_masks = sky_masks.unsqueeze(0).unsqueeze(-1)  # (1,H,W,1)
    elif sky_masks.dim() == 3:  # (B,H,W)
        sky_masks = sky_masks.unsqueeze(-1)  # (B,H,W,1)
    # If already 4D (B,H,W,1), keep as is
    
    sky = sky_masks.float()                            # (B,H,W,1)
    inv = 1.0 - sky.permute(0, 3, 1, 2)               # (B,1,H,W)
    inv_eroded = F.max_pool2d(inv, kernel_size=kernel, stride=1, padding=kernel // 2)
    sky_core = 1.0 - inv_eroded                       # (B,1,H,W)
    return sky_core.permute(0, 2, 3, 1)               # (B,H,W,1) 

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
    # gt_occupied_mask = (1.0 - sky_masks).float()
    # pred_occupied_mask = pred_opacity.squeeze()

    if sky_masks.dim() == pred_opacity.dim() - 1:
        sky_masks = sky_masks.unsqueeze(-1)
    sky = sky_masks.float()
    O = pred_opacity
    if O.dim() == sky.dim() - 1:
        O = O.unsqueeze(-1)
    T_bg = (1.0 - O).clamp(1e-6, 1 - 1e-6)
    
    if loss_type == "bce":
        return binary_cross_entropy(T_bg, sky, reduction="mean")
    elif loss_type == "safe_bce":
        return safe_binary_cross_entropy(T_bg, sky, limit=limit, reduction="mean")
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
        # edge processing
        self.kernel                = loss_config.get("kernel", 5)              
        self.border_weight         = loss_config.get("border_weight", 0.3)     
        self.nonsky_weight         = loss_config.get("nonsky_weight", 1.0)     
        self.safe_limit            = loss_config.get("safe_limit", 1e-6) 
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
        losses: Dict[str, torch.Tensor] = {}
        device = pred_opacity.device
        
        # Handle different input dimensions for sky_masks
        if sky_masks.dim() == 2:  # (H,W)
            sky_masks = sky_masks.unsqueeze(0).unsqueeze(-1)  # (1,H,W,1)
        elif sky_masks.dim() == 3:  # (B,H,W)
            sky_masks = sky_masks.unsqueeze(-1)  # (B,H,W,1)
        # If already 4D (B,H,W,1), keep as is
        
        sky = sky_masks.float().to(device)                    # (B,H,W,1)

        O = pred_opacity
        if O.dim() == 3:
            O = O.unsqueeze(-1)
        O = O.clamp(0.0, 1.0)
        T_bg = (1.0 - O).clamp(1e-6, 1 - 1e-6)               # (B,H,W,1)

        # sky core/ border/ non-sky masks
        sky_core   = _erode_sky_mask(sky, kernel=self.kernel).to(device)   # (B,H,W,1)
        sky_border = (sky - sky_core).clamp(0.0, 1.0)                       # (B,H,W,1)
        non_sky    = (1.0 - sky)                                            # (B,H,W,1)

        loss_core = sky_opacity_loss(
            pred_opacity=O,
            sky_masks=sky_core,
            loss_type=self.opacity_loss_type,
            limit=self.safe_limit
        )
        loss_border = sky_opacity_loss(
            pred_opacity=O,
            sky_masks=sky_border,
            loss_type=self.opacity_loss_type,
            limit=self.safe_limit
        ) * self.border_weight

        loss_nonsky_opaque = safe_binary_cross_entropy(
            pred=1.0 - T_bg,
            gt=non_sky,
            limit=self.safe_limit,
            reduction="mean"
        ) * self.nonsky_weight

        sky_total = self.opacity_loss_weight * (loss_core + loss_border + loss_nonsky_opaque)
        losses["sky_bce_core"]       = self.opacity_loss_weight * loss_core
        losses["sky_bce_border"]     = self.opacity_loss_weight * loss_border
        losses["sky_nonsky_opaque"]  = self.opacity_loss_weight * loss_nonsky_opaque
        losses["sky_opacity_loss"]          = sky_total
        # Sky opacity loss (main loss from DriveStudio)
        # if self.opacity_loss_weight > 0:
        #     opacity_loss = sky_opacity_loss(
        #         pred_opacity, sky_masks, 
        #         loss_type=self.opacity_loss_type
        #     )
        #     losses["sky_opacity_loss"] = self.opacity_loss_weight * opacity_loss
            
        
        # Sky regularization loss (original ObjectGS implementation)
        if self.regularization_weight > 0 and alpha_masks is not None:
            reg_loss = sky_regularization_loss(pred_opacity, alpha_masks)
            losses["sky_regularization_loss"] = self.regularization_weight * reg_loss
            
        return losses
