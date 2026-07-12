# visualization/single_channel.py

import torch
import numpy as np
from einops import rearrange
from matplotlib import cm
from typing import Optional

def apply_color_map(
    x: torch.Tensor,
    color_map: str = "inferno",
) -> np.ndarray:
    """Apply matplotlib colormap to tensor values."""
    cmap = cm.get_cmap(color_map)
    
    # Convert to NumPy and apply colormap
    mapped = cmap(x.detach().clip(min=0, max=1).cpu().numpy())[..., :3]
    
    return mapped

def vis_depth_map(depth_tensor: torch.Tensor, 
                  near: Optional[float] = None, 
                  far: Optional[float] = None,
                  color_map: str = "plasma") -> np.ndarray:
    """
    Visualize depth map with automatic range detection.
    
    Args:
        depth_tensor: Input depth tensor, shape (N, H, W) or (N, T, H, W)
        near: Near clipping value (optional, auto-computed if None)
        far: Far clipping value (optional, auto-computed if None)
        color_map: Matplotlib colormap name
        
    Returns:
        Colorized depth array with shape (..., H, W, 3)
    """
    original_shape = depth_tensor.shape
    
    # Auto-compute range if not provided
    if near is None or far is None:
        valid_depths = depth_tensor[depth_tensor > 0][:16_000_000]  # Sample for efficiency
        
        if len(valid_depths) == 0:
            print("No valid depth values found.")
            near, far = 0.0, 1.0
        else:
            if far is None:
                far = valid_depths.quantile(0.99).log().item()
            if near is None:
                near = valid_depths.quantile(0.01).log().item()
    
    # Apply log transform and normalize
    result = depth_tensor.clone()
    result[result > 0] = result[result > 0].log()
    result = torch.clamp(result, near, far)
    result = 1 - (result - near) / (far - near)  # Invert so closer is brighter
    result = torch.clamp(result, 0, 1)
    
    # Apply colormap - returns numpy array with last dim as RGB
    colored = apply_color_map(result, color_map)
    
    return colored

def vis_single_channel(input_tensor: torch.Tensor,
                      normalize_mode: str = "global",  # "global", "per_image", "none"
                      color_map: str = "magma") -> np.ndarray:
    """
    General single-channel visualization function.
    
    Args:
        input_tensor: Input tensor, shape (N, H, W) or (N, T, H, W)
        normalize_mode: How to normalize values
            - "global": normalize across all images using global min/max
            - "per_image": normalize each image independently  
            - "none": no normalization, assumes values in [0,1]
        color_map: Matplotlib colormap name
        
    Returns:
        Colorized array with shape (..., H, W, 3)
    """
    original_shape = input_tensor.shape
    
    # Normalize based on mode
    if normalize_mode == "global":
        min_val = input_tensor.min()
        max_val = input_tensor.max()
        result = (input_tensor - min_val) / (max_val - min_val + 1e-8)
    elif normalize_mode == "per_image":
        result = torch.zeros_like(input_tensor)
        # Flatten to process each image
        if len(original_shape) == 4:  # N, T, H, W
            flat_tensor = input_tensor.view(-1, original_shape[-2], original_shape[-1])
            flat_result = result.view(-1, original_shape[-2], original_shape[-1])
        else:  # N, H, W
            flat_tensor = input_tensor
            flat_result = result
            
        for i in range(len(flat_tensor)):
            img = flat_tensor[i]
            min_val = img.min()
            max_val = img.max()
            flat_result[i] = (img - min_val) / (max_val - min_val + 1e-8)
    else:  # normalize_mode == "none"
        result = input_tensor.clone()
    
    result = torch.clamp(result, 0, 1)
    
    # Apply colormap
    colored = apply_color_map(result, color_map)
    
    return colored
