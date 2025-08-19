#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements image manipulation and preprocessing functions.

Common Tasks:
    - Format conversions.
    - Image transformations.
    - Pixel operations.
"""

__all__ = [
    "split",
    "to_array",
    "to_channel_first",
    "to_channel_last",
    "to_tensor",
]

import math
from typing import Union

import numpy as np
import torch

from .utils import imgsz, is_channel_first, is_channel_last


# ----- Splitting -----
def split(image: Union[torch.Tensor, np.ndarray], n: int = 2) -> list[np.ndarray]:
    """Split an image into ``n`` equal parts.

    Args:
        image: Image as a ``numpy.ndarray``of shape :math:`(H, W, C)`
            in :math:`[0, 255]`.
        n: Number of parts to split into (positive integer). Default: ``2``.

    Returns:
        A list of sub-images.

    Raises:
        ValueError: If inputs are invalid (e.g., image shape, n).
    """
    if not isinstance(image, np.ndarray) or len(image.shape) != 3:
        raise ValueError(f"[image] must be a numpy.ndarray of shape (H, W, C), got {image.shape}.")
    if n < 1:
        raise ValueError(f"[n] must be a positive integer, got {n}.")

    h, w = imgsz(image)
    if n > h * w:
        raise ValueError(f"[n] ({n}) exceeds image pixel count ({h * w}).")

    # Determine orientation
    is_portrait = h > w

    # Determine rows and cols
    if n == 1:
        rows, cols = 1, 1
    elif n == 2:
        # Explicitly set grid for N=2 based on orientation
        rows = 2 if is_portrait else 1
        cols = 1 if is_portrait else 2
    else:
        # General case: start with approximate square grid
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)
        # Adjust to ensure rows * cols = n, prioritizing orientation
        candidates = []
        for r in range(1, n + 1):
            c = math.ceil(n / r)
            if r * c == n:
                candidates.append((r, c))
        if not candidates:
            raise ValueError(f"Cannot find valid rows and cols for n={n}")
        # Select grid based on orientation
        if is_portrait:
            # Prefer more rows (taller sub-images)
            rows, cols = max(candidates, key=lambda x: x[0] / x[1])
        else:
            # Prefer more cols (wider sub-images)
            rows, cols = max(candidates, key=lambda x: x[1] / x[0])

    # Compute sub-images and adjust bboxes
    sub_h      = h // rows
    sub_w      = w // cols
    sub_images = []

    for i in range(rows):
        for j in range(cols):
            if len(sub_images) >= n:
                break
            # Compute sub-image boundaries
            y_start   = i * sub_h
            y_end     = min((i + 1) * sub_h, h)
            x_start   = j * sub_w
            x_end     = min((j + 1) * sub_w, w)
            sub_image = image[y_start:y_end, x_start:x_end]
            if sub_image.size == 0:
                continue
            sub_images.append(sub_image)

    # Pad with empty sub-images/bboxes if needed
    while len(sub_images) < n:
        sub_images.append(np.zeros_like(sub_images[0]))

    return sub_images


# ----- Format Conversion -----
def to_channel_first(image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Converts an image to channel-first format.

    Args:
        image: Image as a ``torch.Tensor`` or ``numpy.ndarray`` of arbitrary shape.
    
    Returns:
        Channel-first image as a
            ``torch.Tensor`` (i.e., of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`)
            or ``numpy.ndarray`` (i.e., of shape :math:`(C, H, W)` in :math:`[0, 255]`).
    
    Raises:
        ValueError: If ``image`` dimensions are not 3 or 4.
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if is_channel_first(image):
        return image
    if not 3 <= image.ndim <= 4:
        raise ValueError(f"[image]'s number of dimensions must be between 3 and 4, got {image.ndim}.")
    
    if isinstance(image, torch.Tensor):
        image = image.clone()
        if image.ndim == 3:
            image = image.permute(2, 0, 1)     # [H, W, C] -> [C, H, W]
        elif image.ndim == 4:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    elif isinstance(image, np.ndarray):
        image = np.copy(image)  # Changed from copy.deepcopy for efficiency
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))     # [H, W, C] -> [C, H, W]
        elif image.ndim == 4:
            image = np.transpose(image, (0, 3, 1, 2))  # [B, H, W, C] -> [B, C, H, W]
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
    
    return image


def to_channel_last(image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Converts an image to channel-last format.

    Args:
        image: Image as a ``torch.Tensor`` or ``numpy.ndarray`` in 3D or 4D format.
    
    Returns:
        Channel-last image as a
            ``torch.Tensor`` (i.e., of shape :math:`(B, H, W, C)` in :math:`[0.0, 1.0]`)
            or ``numpy.ndarray`` (i.e., of shape :math:`(H, W, C)` in :math:`[0, 255]`).
    
    Raises:
        ValueError: If ``image`` dimensions are not 3 or 4.
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if is_channel_last(image):
        return image
    if not 3 <= image.ndim <= 4:
        raise ValueError(f"[image]'s number of dimensions must be between 3 and 4, got {image.ndim}.")
    
    if isinstance(image, torch.Tensor):
        image = image.clone()
        if image.ndim == 3:
            image = image.permute(1, 2, 0)     # [C, H, W] -> [H, W, C]
        elif image.ndim == 4:
            image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    elif isinstance(image, np.ndarray):
        image = np.copy(image)  # Changed from copy.deepcopy for efficiency
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))     # [C, H, W] -> [H, W, C]
        elif image.ndim == 4:
            image = np.transpose(image, (0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
    
    return image


# ----- Type Conversion -----
def to_array(image: torch.Tensor) -> np.ndarray:
    """Converts an image from ``torch.Tensor`` to ``numpy.ndarray``.
    
    Args:
        image: Image as a ``torch.Tensor`` of shape :math:`(B, C, H, W)`
            in :math:`[0.0, 1.0]`.
    
    Returns:
        Image as a ``numpy.ndarray`` of shape :math:`(H, W, C)` in :math:`[0, 255]`.
    
    Raises:
        ValueError: If ``image`` dimensions are not 4.
        
    Recommend order:
        image = (tensor.squeeze().detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    """
    # Check shape
    if image.ndim != 4:
        raise ValueError(f"[image]'s number of dimensions must be 4, got {image.ndim}.")
    
    image = (image.squeeze().detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return image
    

def to_tensor(image: np.ndarray, normalize: bool = False) -> torch.Tensor:
    """Converts an image from ``numpy.ndarray`` to ``torch.Tensor`` with optional
    normalization.

    Args:
        image: Image as a ``numpy.ndarray`` of shape :math:`(H, W, C)`
            in :math:`[0, 255]`.
        normalize: If ``True``, normalize to :math:`[0.0, 1.0]`. Default: ``False``.

    Returns:
        Image as a ``torch.Tensor`` of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`.
    
    Raises:
        TypeError: If ``image`` is not a ``numpy.ndarray``.
        
    Recommend order:
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div(255.0).unsqueeze(0).to(device)
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"[image] must be a numpy.ndarray, got {type(image)}.")
    
    if normalize:
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div(255.0).unsqueeze(0)
    else:
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().unsqueeze(0)
    return image
