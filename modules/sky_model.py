"""
Sky Model for ObjectGS - Based on OmniRe/DriveStudio Implementation
This module provides sky rendering capabilities for urban scene reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional


class SinusoidalEncoder(nn.Module):
    """Sinusoidal positional encoding for directions."""
    
    def __init__(self, n_input_dims: int = 3, min_deg: int = 0, max_deg: int = 6):
        super().__init__()
        self.n_input_dims = n_input_dims
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.n_output_dims = n_input_dims * (max_deg - min_deg + 1) * 2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., n_input_dims)
        Returns:
            Encoded tensor of shape (..., n_output_dims)
        """
        scales = torch.pow(2.0, torch.arange(self.min_deg, self.max_deg + 1, device=x.device, dtype=x.dtype))
        shape = list(x.shape[:-1]) + [-1]
        scaled_x = (x[..., None, :] * scales[:, None]).reshape(shape)
        encoded = torch.cat([torch.sin(scaled_x), torch.cos(scaled_x)], dim=-1)
        return encoded


class MLP(nn.Module):
    """Multi-layer perceptron with skip connections."""
    
    def __init__(self, in_dims: int, out_dims: int, num_layers: int = 3, 
                 hidden_dims: int = 64, skip_connections: Optional[list] = None):
        super().__init__()
        self.skip_connections = skip_connections or []
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layer_in_dims = in_dims
            elif i in self.skip_connections:
                layer_in_dims = hidden_dims + in_dims
            else:
                layer_in_dims = hidden_dims
                
            if i == num_layers - 1:
                layer_out_dims = out_dims
            else:
                layer_out_dims = hidden_dims
                
            layers.append(nn.Linear(layer_in_dims, layer_out_dims))
            
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_connections:
                x = torch.cat([x, input_x], dim=-1)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class SkyModel(nn.Module):
    """
    Neural sky model that learns to render sky appearance based on view directions.
    Based on the OmniRe/DriveStudio implementation.
    """
    
    def __init__(self, 
                 n_images: int,
                 head_mlp_layer_width: int = 64,
                 enable_appearance_embedding: bool = True,
                 appearance_embedding_dim: int = 16,
                 device: torch.device = torch.device("cuda")):
        super().__init__()
        self.device = device
        self.direction_encoding = SinusoidalEncoder(n_input_dims=3, min_deg=0, max_deg=6)
        self.direction_encoding.requires_grad_(False)
        
        self.enable_appearance_embedding = enable_appearance_embedding
        if self.enable_appearance_embedding:
            self.appearance_embedding_dim = appearance_embedding_dim
            self.appearance_embedding = nn.Embedding(n_images, appearance_embedding_dim, dtype=torch.float32)
            
        in_dims = self.direction_encoding.n_output_dims + appearance_embedding_dim \
            if self.enable_appearance_embedding else self.direction_encoding.n_output_dims
            
        self.sky_head = MLP(
            in_dims=in_dims,
            out_dims=3,
            num_layers=3,
            hidden_dims=head_mlp_layer_width,
            skip_connections=[1],
        )
        self.in_test_set = False
    
    def forward(self, viewdirs: torch.Tensor, img_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Render sky color based on view directions.
        
        Args:
            viewdirs: View directions tensor of shape (..., 3)
            img_idx: Image indices for appearance embedding, shape (...,)
        
        Returns:
            RGB sky colors of shape (..., 3)
        """
        self.device = viewdirs.device
        prefix = viewdirs.shape[:-1]
        
        # Encode directions
        dd = self.direction_encoding(viewdirs.reshape(-1, 3)).to(self.device)
        
        if self.enable_appearance_embedding:
            # Add appearance embedding
            if img_idx is not None and not self.in_test_set:
                appearance_embedding = self.appearance_embedding(img_idx.reshape(-1)).reshape(-1, self.appearance_embedding_dim)
            else:
                # Use mean appearance embedding
                appearance_embedding = torch.ones(
                    (*dd.shape[:-1], self.appearance_embedding_dim),
                    device=dd.device,
                    dtype=dd.dtype
                ) * self.appearance_embedding.weight.mean(dim=0)
            dd = torch.cat([dd, appearance_embedding], dim=-1)
            
        rgb_sky = self.sky_head(dd).to(self.device)
        rgb_sky = F.sigmoid(rgb_sky)
        return rgb_sky.reshape(prefix + (3,))
    
    def get_param_groups(self) -> Dict[str, torch.nn.Parameter]:
        """Get parameter groups for optimization."""
        return {"sky_all": list(self.parameters())}


class EnvLight(torch.nn.Module):
    """
    Environment light model using cube map representation.
    Alternative to SkyModel for simpler sky rendering.
    """
    
    def __init__(self, resolution: int = 1024, device: torch.device = torch.device("cuda")):
        super().__init__()
        self.device = device
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], 
                                     dtype=torch.float32, device=device)
        self.base = torch.nn.Parameter(
            0.5 * torch.ones(6, resolution, resolution, 3, requires_grad=True),
        )
        
    def forward(self, viewdirs: torch.Tensor) -> torch.Tensor:
        """
        Render sky color from environment map.
        
        Args:
            viewdirs: View directions tensor of shape (..., 3)
            
        Returns:
            RGB sky colors of shape (..., 3)
        """
        l = viewdirs
        l = (l.reshape(-1, 3) @ self.to_opengl.T).reshape(*l.shape)
        l = l.contiguous()
        prefix = l.shape[:-1]
        
        if len(prefix) != 3:  # reshape to [B, H, W, -1]
            l = l.reshape(1, 1, -1, l.shape[-1])

        try:
            import nvdiffrast.torch as dr
            light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
            light = light.view(*prefix, -1)
        except ImportError:
            # Fallback to simple sampling if nvdiffrast is not available
            print("Warning: nvdiffrast not available, using simplified sky model")
            light = torch.ones(*prefix, 3, device=viewdirs.device) * 0.5
            
        return light
    
    def get_param_groups(self) -> Dict[str, torch.nn.Parameter]:
        """Get parameter groups for optimization."""
        return {"envlight_all": list(self.parameters())}


def create_sky_model(model_type: str = "neural", **kwargs) -> nn.Module:
    """
    Factory function to create sky models.
    
    Args:
        model_type: Type of sky model ('neural' or 'envlight')
        **kwargs: Additional arguments for the model
        
    Returns:
        Sky model instance
    """
    if model_type == "neural":
        return SkyModel(**kwargs)
    elif model_type == "envlight":
        return EnvLight(**kwargs)
    else:
        raise ValueError(f"Unknown sky model type: {model_type}")
