"""
Sky-related loss functions for ObjectGS.
Based on OmniRe/DriveStudio implementation.
Modified to support validity_mask for generated images.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional


def binary_cross_entropy(pred: torch.Tensor, gt: torch.Tensor, 
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Binary cross entropy loss, with optional masking.
    Computes mean only over masked-in pixels.
    """
    # Get per-pixel loss
    per_pixel_loss = F.binary_cross_entropy(pred, gt, reduction="none")

    if mask is not None:
        # Ensure mask is broadcastable
        if mask.dim() == per_pixel_loss.dim() - 1:
            mask = mask.unsqueeze(-1)
        
        masked_loss = per_pixel_loss * mask
        
        # Compute mean only over valid pixels
        if mask.sum() > 0:
            return masked_loss.sum() / mask.sum()
        else:
            return torch.tensor(0.0, device=pred.device)
    else:
        # No mask, compute standard mean
        return per_pixel_loss.mean()


def safe_binary_cross_entropy(pred: torch.Tensor, gt: torch.Tensor, 
                            limit: float = 0.1, 
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Safe binary cross entropy loss with clamping, with optional masking.
    Computes mean only over masked-in pixels.
    """
    pred = torch.clamp(pred, limit, 1.0 - limit)
    
    # Get per-pixel loss
    per_pixel_loss = F.binary_cross_entropy(pred, gt, reduction="none")

    if mask is not None:
        # Ensure mask is broadcastable
        if mask.dim() == per_pixel_loss.dim() - 1:
            mask = mask.unsqueeze(-1)
        
        masked_loss = per_pixel_loss * mask
        
        # Compute mean only over valid pixels
        if mask.sum() > 0:
            return masked_loss.sum() / mask.sum()
        else:
            return torch.tensor(0.0, device=pred.device)
    else:
        # No mask, compute standard mean
        return per_pixel_loss.mean()

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
                    loss_type: str = "bce", limit: float = 0.1,
                    validity_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Sky opacity loss to ensure proper sky region handling, supporting validity masking.
    
    Args:
        pred_opacity: Predicted opacity values, shape (...,)
        sky_masks: Sky mask indicating sky regions (1 for sky, 0 for non-sky), shape (...,)
                     This is used as the TARGET for BCE.
        loss_type: Type of loss ('bce' or 'safe_bce')
        limit: Limit for safe BCE
        validity_mask: Mask of "real" pixels (1=real, 0=unreal)
        
    Returns:
        Sky opacity loss
    """
    
    # Handle dimensions
    O = pred_opacity
    if O.dim() == 3:
        O = O.unsqueeze(-1)
        
    if sky_masks.dim() == pred_opacity.dim() - 1:
        sky_masks = sky_masks.unsqueeze(-1)
    
    sky = sky_masks.float()
    T_bg = (1.0 - O).clamp(1e-6, 1 - 1e-6)
    
    # The region of interest is defined by the sky_mask itself (e.g., sky_core)
    # We combine this with the validity_mask
    final_mask = sky # The region mask (e.g., sky_core or sky_border)
    
    if validity_mask is not None:
        # validity_mask should already be 4D from compute_losses
        # But add safety check
        if validity_mask.dim() != final_mask.dim():
            while validity_mask.dim() < final_mask.dim():
                validity_mask = validity_mask.unsqueeze(0) if validity_mask.dim() < 3 else validity_mask.unsqueeze(-1)
        final_mask = final_mask * validity_mask
    
    if loss_type == "bce":
        return binary_cross_entropy(T_bg, sky, mask=final_mask)
    elif loss_type == "safe_bce":
        return safe_binary_cross_entropy(T_bg, sky, limit=limit, mask=final_mask)
    else:
        raise ValueError(f"Unknown sky opacity loss type: {loss_type}")


def sky_regularization_loss(alpha: torch.Tensor, alpha_masks: torch.Tensor,
                            validity_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Regularization loss to encourage low opacity in background regions.
    This is the original ObjectGS sky loss implementation, now with validity masking.
    
    Args:
        alpha: Alpha/opacity values
        alpha_masks: Foreground masks (1 for foreground, 0 for background)
        validity_mask: Mask of "real" pixels (1=real, 0=unreal)
        
    Returns:
        Sky regularization loss
    """
    o = alpha.clamp(1e-6, 1-1e-6)
    foreground = alpha_masks.float()
    background = 1.0 - foreground # 1 for background pixels
    
    # Penalize opacity 'o' in the background
    per_pixel_loss = -background * torch.log(1 - o)

    final_mask = None
    if validity_mask is not None:
        # validity_mask is 4D but per_pixel_loss is 3D, need to adjust
        if validity_mask.dim() > per_pixel_loss.dim():
            # Remove extra dimensions
            while validity_mask.dim() > per_pixel_loss.dim():
                if validity_mask.shape[-1] == 1:
                    validity_mask = validity_mask.squeeze(-1)
                elif validity_mask.shape[0] == 1:
                    validity_mask = validity_mask.squeeze(0)
                else:
                    break
        elif validity_mask.dim() < per_pixel_loss.dim():
            while validity_mask.dim() < per_pixel_loss.dim():
                validity_mask = validity_mask.unsqueeze(0)
        final_mask = validity_mask # Average over all valid pixels
    
    if final_mask is not None:
        masked_loss = per_pixel_loss * final_mask
        if final_mask.sum() > 0:
            return masked_loss.sum() / final_mask.sum()
        else:
            return torch.tensor(0.0, device=alpha.device)
    else:
        # Original behavior: mean over all pixels
        return per_pixel_loss.mean()


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
                       alpha_masks: Optional[torch.Tensor] = None,
                       validity_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all sky-related losses, supporting validity masking.
        
        Args:
            pred_opacity: Predicted opacity values
            sky_masks: Sky masks (1 for sky, 0 for non-sky)
            alpha_masks: Alpha masks for regularization (1 for foreground, 0 for background)
            validity_mask: Mask of "real" pixels (1=real, 0=unreal)
            
        Returns:
            Dictionary of loss terms
        """
        losses: Dict[str, torch.Tensor] = {}
        device = pred_opacity.device
        
        # Ensure validity_mask is on the correct device and adjust dimensions
        if validity_mask is not None:
            validity_mask = validity_mask.to(device)
            # Convert validity_mask to 4D format to match other tensors
            if validity_mask.dim() == 2:  # (H,W)
                validity_mask = validity_mask.unsqueeze(0).unsqueeze(-1)  # (1,H,W,1)
            elif validity_mask.dim() == 3:  # (B,H,W) or (C,H,W)
                if validity_mask.shape[0] == 1:  # (1,H,W)
                    validity_mask = validity_mask.unsqueeze(-1)  # (1,H,W,1)
                else:  # (C,H,W) - take first channel or mean
                    validity_mask = validity_mask[0:1].unsqueeze(-1)  # (1,H,W,1)
            # If already 4D, keep as is
            
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

        # --- Compute Losses with Validity Mask ---
        
        loss_core = sky_opacity_loss(
            pred_opacity=O,
            sky_masks=sky_core,  # Target is sky_core
            loss_type=self.opacity_loss_type,
            limit=self.safe_limit,
            validity_mask=validity_mask  # Pass the validity mask
        )
        
        loss_border = sky_opacity_loss(
            pred_opacity=O,
            sky_masks=sky_border, # Target is sky_border
            loss_type=self.opacity_loss_type,
            limit=self.safe_limit,
            validity_mask=validity_mask  # Pass the validity mask
        ) * self.border_weight

        # For non-sky loss, the region is `non_sky`
        final_nonsky_mask = non_sky
        if validity_mask is not None:
            # validity_mask is already 4D at this point
            final_nonsky_mask = final_nonsky_mask * validity_mask

        loss_nonsky_opaque = safe_binary_cross_entropy(
            pred=1.0 - T_bg,  # Predict 1 (opaque)
            gt=non_sky,       # Target is 1 (non-sky)
            limit=self.safe_limit,
            mask=final_nonsky_mask  # Pass combined mask
        ) * self.nonsky_weight

        sky_total = self.opacity_loss_weight * (loss_core + loss_border + loss_nonsky_opaque)
        losses["sky_bce_core"]       = self.opacity_loss_weight * loss_core
        losses["sky_bce_border"]     = self.opacity_loss_weight * loss_border
        losses["sky_nonsky_opaque"]  = self.opacity_loss_weight * loss_nonsky_opaque
        losses["sky_opacity_loss"]   = sky_total
            
        
        # Sky regularization loss (original ObjectGS implementation)
        if self.regularization_weight > 0 and alpha_masks is not None:
            reg_loss = sky_regularization_loss(
                pred_opacity, 
                alpha_masks, 
                validity_mask=validity_mask # Pass the validity mask
            )
            losses["sky_regularization_loss"] = self.regularization_weight * reg_loss
            
        return losses