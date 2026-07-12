import torch
import numpy as np

import cv2
from PIL import Image

def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTHß
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def to_tensor(data) -> torch.Tensor:
    """
    Convert image(s) to PyTorch float32 tensor(s) in [0, 1] range, 
    with shape [C, H, W] for single images or [B, C, H, W] for batches.

    Args:
        data:
            - numpy.ndarray: shape [H, W, C] or [B, H, W, C], dtype uint8 or float.
            - PIL.Image.Image: single image.
            - list of PIL.Image.Image: multiple images (same size).

    Returns:
        torch.Tensor: float32 tensor, shape [C, H, W] (single) or [B, C, H, W] (batch).
    """

    # Handle NumPy input
    if isinstance(data, np.ndarray):
        arr = data
        if arr.ndim == 3:  # Single image
            arr = arr[None, ...]
            single_input = True
        elif arr.ndim == 4:
            single_input = False
        else:
            raise ValueError(f"Unsupported NumPy array shape: {arr.shape}")

    # Handle single PIL image
    elif isinstance(data, Image.Image):
        arr = np.array(data)[None, ...]
        single_input = True

    # Handle list of PIL images
    elif isinstance(data, list) and all(isinstance(img, Image.Image) for img in data):
        arr = np.stack([np.array(img) for img in data], axis=0)
        single_input = False

    else:
        raise TypeError("Input must be a NumPy array, a PIL Image, or a list of PIL Images.")

    # Convert to float32 and normalize
    arr = arr.astype(np.float32) / 255.0

    # Rearrange to [B, C, H, W]
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)

    # If the original input was single, remove batch dimension
    if single_input:
        tensor = tensor.squeeze(0)

    return tensor
