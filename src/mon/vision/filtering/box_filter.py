#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements box filters."""

__all__ = [
    "BoxFilter",
    "box_filter",
    "box_filter_conv",
]

import cv2
import numpy as np
import torch

from mon import nn
from mon.nn import functional as F
from mon.vision import types


# ----- Utils -----
def diff_x(image: torch.Tensor, radius: int) -> torch.Tensor:
    """Computes difference along the x-axis of an image.

    Args:
        image: Image as ``torch.Tensor`` in [B, C, H, W], range [0.0, 1.0].
        radius: Radius of the kernel.
    
    Returns:
        Tensor with x-axis differences.
    
    Raises:
        ValueError: If image does not have 4 dimensions.
    
    References:
        - https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
    """
    if image.ndim != 4:
        raise ValueError("[image] must have 4 dimensions")
    left   = image[:, :, radius        : 2 * radius + 1]
    middle = image[:, :, 2 * radius + 1:               ] - image[: , : ,                : -2 * radius - 1]
    right  = image[:, :, -1            :               ] - image[: , : , -2 * radius - 1:     -radius - 1]
    output = torch.cat([left, middle, right], dim=2)
    return output


def diff_y(image: torch.Tensor, radius: int) -> torch.Tensor:
    """Computes difference along the y-axis of an image.

    Args:
        image: Image as ``torch.Tensor`` in [B, C, H, W], range [0.0, 1.0].
        radius: Radius of the kernel.
    
    Returns:
        Tensor with y-axis differences.
    
    Raises:
        ValueError: If image does not have 4 dimensions.
    
    References:
        - https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
    """
    if image.ndim != 4:
        raise ValueError("[image] must have 4 dimensions")
    left   = image[:, :, :,         radius:2 * radius + 1]
    middle = image[:, :, :, 2 * radius + 1:              ] - image[:, :, :,                :-2 * radius - 1]
    right  = image[:, :, :,             -1:              ] - image[:, :, :, -2 * radius - 1:    -radius - 1]
    output = torch.cat([left, middle, right], dim=3)
    return output


# ----- Box Filter -----
def box_filter(
    image      : torch.Tensor,
    kernel_size: int = None,
    radius     : int = None,
    **kwargs
) -> torch.Tensor:
    """Performs box filtering on an image.

    Args:
        image: Image as ``torch.Tensor`` in [B, C, H, W], range [0.0, 1.0].
        kernel_size: Size of the kernel (e.g., 3, 5, 7, 9).
        radius: Radius of the kernel (kernel_size = radius * 2 + 1).
    
    Returns:
        Filtered image.
    
    Raises:
        ValueError: If neither ``kernel_size`` nor ``radius`` is provided, or image dimensions are invalid.
        TypeError: If image type is neither ``torch.Tensor`` nor ``numpy.ndarray``.
    
    Notes:
        For ``torch.Tensor``, kwargs are ignored, and filtering uses diff_x/diff_y.
        For ``numpy.ndarray``, kwargs include:
            - ddepth: Output image depth. Default is ``-1`` (same as input).
            - anchor: Kernel anchor point. Default is ``(-1, -1)`` (center).
            - normalize: Normalize kernel if True. Default is ``False``.
            - borderType: Border mode (e.g., cv2.BORDER_DEFAULT). Default is ``cv2.BORDER_DEFAULT``.
    
    References:
        - https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
    """
    if kernel_size is None and radius is None:
        raise ValueError("Either [kernel_size] or [radius] must be provided.")
    if isinstance(image, torch.Tensor):
        if image.ndim != 4:
            raise ValueError("[image] must have 4 dimensions")
        radius = radius or int((kernel_size - 1) / 2)
        return diff_y(diff_x(image.cumsum(dim=2), radius).cumsum(dim=3), radius)
    elif isinstance(image, np.ndarray):
        ddepth      = kwargs.get("ddepth",     -1)
        anchor      = kwargs.get("anchor",     (-1, -1))
        normalize   = kwargs.get("normalize",  False)
        borderType  = kwargs.get("borderType", cv2.BORDER_DEFAULT)
        kernel_size = kernel_size or 2 * radius + 1
        kernel_size = types.image_size(kernel_size)
        return cv2.boxFilter(
            src        = image,
            ddepth     = ddepth,
            ksize      = kernel_size,
            anchor     = anchor,
            normalize  = normalize,
            borderType = borderType
        )
    else:
        raise TypeError(f"[image] must be torch.Tensor or numpy.ndarray, got {type(image)}.")
    

def box_filter_conv(
    image      : torch.Tensor,
    kernel_size: int = None,
    radius     : int = None
) -> torch.Tensor:
    """Performs box filtering on an image using convolution.

    Args:
        image: Image as ``torch.Tensor`` in [B, C, H, W] format.
        kernel_size: Size of the kernel (e.g., 3, 5, 7, 9).
        radius: Radius of the kernel (kernel_size = radius * 2 + 1, e.g., 1, 2, 3, 4).
    
    Returns:
        Filtered image.
    
    Raises:
        ValueError: If neither ``kernel_size`` nor ``radius`` is provided.
    """
    if kernel_size is None and radius is None:
        raise ValueError("Either [kernel_size] or [radius] must be provided.")
    kernel_size = kernel_size or 2 * radius + 1
    b, c, h, w  = image.shape
    kernel      = torch.ones(b, 1, kernel_size, kernel_size, device=image.device)
    kernel     /= kernel_size ** 2
    output      = [F.conv2d(image[:, i:i + 1, :, :], kernel, padding=kernel_size // 2)
                   for i in range(image.size(1))]
    output      = torch.cat(output, dim=1)
    return output


class BoxFilter(nn.Module):
    """Applies box filtering to an image.

    Args:
        kernel_size: Size of the kernel (e.g., 3, 5, 7, 9).
        radius: Radius of the kernel (kernel_size = radius * 2 + 1, e.g., 1, 2, 3, 4).
    
    Raises:
        ValueError: If neither ``kernel_size`` nor ``radius`` is provided.
    """
    
    def __init__(self, kernel_size: int = None, radius: int = None):
        super().__init__()
        if kernel_size is None and radius is None:
            raise ValueError("Either [kernel_size] or [radius] must be provided.")
        self.kernel_size = kernel_size or 2 * radius + 1
        self.radius      = int((self.kernel_size - 1) / 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Filters the image using box filtering.

        Args:
            image: Image as ``torch.Tensor`` in [B, C, H, W] format.
        
        Returns:
            Filtered image.
        """
        return box_filter(image, self.kernel_size, self.radius)
