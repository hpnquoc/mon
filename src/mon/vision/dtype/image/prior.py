#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Data Type.

This module implements the basic functionalities for image data.
"""

from __future__ import annotations

__all__ = [
    "BoundaryAwarePrior",
    "BrightnessAttentionMap",
    "ImageLocalMean",
    "ImageLocalStdDev",
    "ImageLocalVariance",
    "atmospheric_prior",
    "blur_spot_prior",
    "boundary_aware_prior",
    "bright_spot_prior",
    "brightness_attention_map",
    "image_local_mean",
    "image_local_stddev",
    "image_local_variance",
]

import cv2
import kornia
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from mon.vision.dtype.image import base
from mon import core


# region Image Prior

def atmospheric_prior(
    image      : np.ndarray,
    kernel_size: _size_2_t = 15,
    p          : float     = 0.0001
) -> np.ndarray:
    """Get the atmosphere light in RGB image.

    Args:
        image: An RGB image of type `numpy.ndarray` in [H, W, C]
            format with data in the range [0, 255].
        kernel_size: Window for the dark channel. Default: ``15``.
        p: Percentage of pixels for estimating the atmosphere light.
            Default: ``0.0001``.
    
    Returns:
        A 3-element array containing atmosphere light ``([0, L-1])`` for each
        channel.
    """
    image      = image.transpose(1, 2, 0)
    # Reference CVPR09, 4.4
    dark       = dark_channel_prior_02(image=image, kernel_size=kernel_size)
    m, n       = dark.shape
    flat_i     = image.reshape(m * n, 3)
    flat_dark  = dark.ravel()
    search_idx = (-flat_dark).argsort()[:int(m * n * p)]  # find top M * N * p indexes
    # Return the highest intensity for each channel
    return np.max(flat_i.take(search_idx, axis=0), axis=0)


def blur_spot_prior(image: np.ndarray, threshold: int = 250) -> bool:
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding for bright spot detection
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Apply Laplacian filter for edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Calculate maximum intensity and variance
    laplacian_var = laplacian.var()
    # Check blur condition based on variance of Laplacian image
    is_blur = True if laplacian_var < threshold else False
    return is_blur


def boundary_aware_prior(
    image      : torch.Tensor | np.ndarray,
    eps        : float = 0.05,
    as_gradient: bool  = False,
    normalized : bool  = False,
) -> torch.Tensor | np.ndarray:
    """Get the boundary prior from an RGB or grayscale image.
    
    Args:
        image: An RGB image of type:
            - `torch.Tensor` in [B, C, H, W] format with data in
                the range [0.0, 1.0].
            - `numpy.ndarray` in [H, W, C] format with data in the
                range [0, 255].
        eps: Threshold to remove weak edges. Default: ``0.05``.
        as_gradient: If ``True``, return the gradient image. Default: ``False``.
        normalized: If ``True``, L1 norm of the kernel is set to ``1``.
            Default: ``False``.
        
    Returns:
        A boundary aware prior as a binary image.
    """
    if isinstance(image, torch.Tensor):
        gradient = kornia.filters.sobel(image, normalized=normalized, eps=1e-6)
        g_max    = torch.max(gradient)
        gradient = gradient / g_max
        boundary = (gradient > eps).float()
    elif isinstance(image, np.ndarray):
        if base.is_image_colored(image):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        from mon.vision.filtering import sobel_filter
        gradient = sobel_filter(image, kernel_size=3)
        g_max    = np.max(gradient)
        gradient = gradient / g_max
        boundary = (gradient > eps).float()
        return boundary
    else:
        raise ValueError(f"Unsupported input type: {type(image)}.")
    
    # return boundary, gradient
    if as_gradient:
        return gradient
    else:
        return boundary


def bright_spot_prior(image: np.ndarray) -> bool:
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding for bright spot detection
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Calculate maximum intensity and variance
    binary_var = binary.var()
    # Check bright spot condition based on variance of binary image
    is_bright = True if 5000 < binary_var < 8500 else False
    return is_bright


def brightness_attention_map(
    image        : torch.Tensor | np.ndarray,
    gamma        : float     = 2.5,
    denoise_ksize: _size_2_t = None,
) -> torch.Tensor:
    """Get the Brightness Attention Map (BAM) prior from an RGB image.
    
    This is a self-attention map extracted from the V-channel of a low-light
    image. This map is multiplied to convolutional activations of all layers in
    the enhancement network. Brighter regions are given lower weights to avoid
    over-saturation, while preserving image details and enhancing the contrast
    in the dark regions effectively.
    
    Equation: `I_{attn} = (1 - I_{V})^{\gamma}`, where `\gamma \geq 1`.
    
    Args:
        image: An RGB image of type:
            - `torch.Tensor` in [B, C, H, W] format with data in
                the range [0.0, 1.0].
            - `numpy.ndarray` in [H, W, C] format with data in the
                range [0, 255].
        gamma: A parameter controls the curvature of the map.
        denoise_ksize: Window size for de-noising operation. Default: ``None``.
        
    Returns:
        An `numpy.ndarray` brightness enhancement map as prior.
    """
    if isinstance(image, torch.Tensor):
        if denoise_ksize:
            image = kornia.filters.median_blur(image, denoise_ksize)
            # image = kornia.filters.bilateral_blur(image, denoise_ksize, 0.1, (1.5, 1.5))
        hsv = kornia.color.rgb_to_hsv(image)
        v   = base.get_image_channel(image=hsv, index=(2, 3), keep_dim=True)  # hsv[:, 2:3, :, :]
        bam = torch.pow((1 - v), gamma)
    elif isinstance(image, np.ndarray):
        if denoise_ksize:
            image = cv2.medianBlur(image, denoise_ksize)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if hsv.dtype != np.float64:
            hsv  = hsv.astype("float64")
            hsv /= 255.0
        v   = base.get_image_channel(image=hsv, index=(2, 3), keep_dim=True)  # hsv[:, :, 2:3]
        bam = np.power((1 - v), gamma)
    else:
        raise ValueError(f"Unsupported input type: {type(image)}.")
    return bam


def image_local_mean(image: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
    """Calculate the local mean of an image using a sliding window.
    
    Args:
        image: The input image tensor of shape [B, C, H, W].
        patch_size: The size of the sliding window. Default: ``5``.
    """
    padding = patch_size // 2
    image   = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    return patches.mean(dim=(4, 5))


def image_local_variance(image: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
    """Calculate the local variance of an image using a sliding window.
    
    Args:
        image: The input image tensor of shape [B, C, H, W].
        patch_size: The size of the sliding window. Default: ``5``.
    """
    padding = patch_size // 2
    image   = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    mean    = patches.mean(dim=(4, 5))
    return ((patches - mean.unsqueeze(4).unsqueeze(5)) ** 2).mean(dim=(4, 5))


def image_local_stddev(
    image     : torch.Tensor,
    patch_size: int   = 5,
    eps       : float = 1e-9,
) -> torch.Tensor:
    """Calculate the local standard deviation of an image using a sliding window.
    
    Args:
        image: The input image tensor of shape [B, C, H, W].
        patch_size: The size of the sliding window. Default: ``5``.
        eps: A small value to avoid sqrt by zero. Default: ``1e-9``.
    """
    padding        = patch_size // 2
    image          = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches        = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    mean           = patches.mean(dim=(4, 5), keepdim=True)
    squared_diff   = (patches - mean) ** 2
    local_variance = squared_diff.mean(dim=(4, 5))
    local_stddev   = torch.sqrt(local_variance + eps)
    return local_stddev


class BoundaryAwarePrior(nn.Module):
    """Get the boundary prior from an RGB or grayscale image.
    
    Args:
        eps: Threshold weak edges. Default: ``0.05``.
        normalized: If ``True``, L1 norm of the kernel is set to ``1``.
            Default: ``True``.
    """
    
    def __init__(self, eps: float = 0.05, normalized: bool = False):
        super().__init__()
        self.eps        = eps
        self.normalized = normalized
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return boundary_aware_prior(image, self.eps, self.normalized)


class BrightnessAttentionMap(nn.Module):
    """Get the Brightness Attention Map (BAM) prior from an RGB image.
    
    This is a self-attention map extracted from the V-channel of a low-light
    image. This map is multiplied to convolutional activations of all layers in
    the enhancement network. Brighter regions are given lower weights to avoid
    over-saturation, while preserving image details and enhancing the contrast
    in the dark regions effectively.
    
    Equation: `I_{attn} = (1 - I_{V})^{\gamma}`, where `\gamma \geq 1`.
    
    Args:
        gamma: A parameter controls the curvature of the map.
        denoise_ksize: Window size for de-noising operation. Default: ``None``.
    """
    
    def __init__(
        self,
        gamma        : float     = 2.5,
        denoise_ksize: _size_2_t = None
    ):
        super().__init__()
        self.gamma         = gamma
        self.denoise_ksize = denoise_ksize
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return brightness_attention_map(image, self.gamma, self.denoise_ksize)


class ImageLocalMean(nn.Module):
    """Calculate the local mean of an image using a sliding window.
    
    Args:
        patch_size: The size of the sliding window. Default: ``5``.
    """
    
    def __init__(self, patch_size: int = 5):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, image):
        return image_local_mean(image, self.patch_size)


class ImageLocalVariance(nn.Module):
    """Calculate the local variance of an image using a sliding window.
    
    Args:
        patch_size: The size of the sliding window. Default: ``5``.
    """
    
    def __init__(self, patch_size: int = 5):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, image):
        return image_local_variance(image, self.patch_size)


class ImageLocalStdDev(nn.Module):
    """Calculate the local standard deviation of an image using a sliding window.
    
    Args:
        patch_size: The size of the sliding window. Default: ``5``.
        eps: A small value to avoid sqrt by zero. Default: ``1e-9``.
    """
    
    def __init__(self, patch_size: int = 5, eps: float = 1e-9):
        super().__init__()
        self.patch_size = patch_size
        self.eps        = eps
    
    def forward(self, image):
        return image_local_stddev(image, self.patch_size, self.eps)

# endregion
