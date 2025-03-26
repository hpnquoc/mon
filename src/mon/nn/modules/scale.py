#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sampling Layers.

This module implements upsampling and downsampling layers.
"""

from __future__ import annotations

__all__ = [
    "Downsample",
    "DownsampleConv2d",
    "Interpolate",
    "Scale",
    "Upsample",
    "UpsampleConv2d",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
]

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.upsampling import *


# region Utils

def get_image_size(input: Any) -> tuple[int, int]:
    """Gets the size of an image as ``(height, width)``.

    Args:
        input: Image or data to measure.
    Returns:
        Tuple of ``(height, width)`` in pixels.
    """
    from mon.vision.dtype import image as I
    return I.get_image_size(input)
    
# endregion


# region Downsampling

class Downsample(nn.Module):
    """Downsamples multi-channel 1D, 2D, or 3D data.
    
    Args:
        size: Output spatial sizes. Default is ``None``.
        scale_factor: Multiplier for spatial size. Default is ``None``.
        mode: Interpolation algorithm. One of: ``'nearest'``, ``'linear'``, ``'bilinear'``,
            ``'bicubic'``, or ``'trilinear'``. Default is ``nearest``.
        align_corners: If ``True``, the corner pixels of the input and output
            tensors are aligned, and thus preserving the values of those pixels.
            This only has effect when ``mode`` is ``'linear'``, ``'bilinear'``,
            ``'bicubic'``, or ``'trilinear'``. Default is ``False``.
        recompute_scale_factor: Recompute scale factor if ``True``. Default is ``False``.
            - If ``True``, then ``scale_factor`` must be passed in and ``scale_factor``
                is used to compute the output ``size``. The computed output ``size``
                will be used to infer new scales for the interpolation. Note that when
                ``scale_factor`` is floating-point, it may differ from the recomputed
                ``scale_factor`` due to rounding and precision issues.
            - If ``False``, then `size` or `scale_factor` will be used directly for interpolation.
    """
    
    def __init__(
        self,
        size                  : Any  = None,
        scale_factor          : Any  = None,
        mode                  : str  = "nearest",
        align_corners         : bool = False,
        recompute_scale_factor: bool = False
    ):
        super().__init__()
        self.size                   = size
        self.scale_factor           = self._invert_scale_factor(scale_factor)
        self.mode                   = mode
        self.align_corners          = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def _invert_scale_factor(self, scale_factor: Any) -> Any:
        """Inverts scale factor for downsampling."""
        if isinstance(scale_factor, tuple):
            return tuple(1.0 / factor for factor in scale_factor)
        return 1.0 / scale_factor if scale_factor else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Downsamples the input tensor.

        Args:
            input: Tensor to downsample.
            
        Returns:
            Downsampled tensor.
        """
        if self.size and self.size == list(input.shape[2:]):
            return input
        if self.scale_factor and isinstance(self.scale_factor, tuple) and all(s == 1.0 for s in self.scale_factor):
            return input
        return F.interpolate(
            input                  = input,
            size                   = self.size,
            scale_factor           = self.scale_factor,
            mode                   = self.mode,
            align_corners          = self.align_corners,
            recompute_scale_factor = self.recompute_scale_factor
        )


class DownsampleConv2d(nn.Module):
    """Downsamples 2D data using a convolutional layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv         = nn.Sequential(nn.Conv2d(in_channels, out_channels, 4, 2, 1))
        self.in_channels  = in_channels
        self.out_channels = out_channels
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Downsamples input tensor via convolution.

        Args:
            input: Tensor of shape ``(B, L, C)`` to downsample.
        
        Returns:
            Downsampled tensor of shape ``(B, H*W, C)``.
        """
        x       = input
        b, l, c = x.shape
        h       = int(math.sqrt(l))
        w       = int(math.sqrt(l))
        x       = x.transpose(1, 2).contiguous().view(b, c, h, w)
        x       = self.conv(x).flatten(2).transpose(1, 2).contiguous()
        return x
    
    def flops(self, h: int, w: int) -> int:
        """Calculates FLOPs for the downsampling operation.

        Args:
            h: Input height.
            w: Input width.
            
        Returns:
            Total FLOPs as an integer.
        """
        return h // 2 * w // 2 * self.in_channels * self.out_channels * 4 * 4

# endregion


# region Upsampling

class UpsampleConv2d(nn.Module):
    """Upsamples 2D data using a transposed convolutional layer.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv       = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.in_channels  = in_channels
        self.out_channels = out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Upsamples input tensor via transposed convolution.

        Args:
            input: Tensor of shape ``(B, L, C)`` to upsample.
            
        Returns:
            Upsampled tensor of shape ``(B, H*W, C)``.
        """
        b, l, c = input.shape
        h       = w = int(math.sqrt(l))
        x       = input.transpose(1, 2).view(b, c, h, w)
        x       = self.deconv(x).flatten(2).transpose(1, 2)
        return x

    def flops(self, h: int, w: int) -> int:
        """Calculates FLOPs for the upsampling operation.

        Args:
            h: Input height.
            w: Input width.
            
        Returns:
            Total FLOPs as an integer.
        """
        return h * w * self.in_channels * self.out_channels * 16

# endregion


# region Misc

class Scale(nn.Module):
    """Applies a learnable scale parameter to input data.

    Args:
        scale: Initial scale factor value. Default is ``1.0``.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Adds scale factor to input tensor.

        Args:
            input: Tensor to scale.
        
        Returns:
            Scaled tensor.
        """
        return input + self.scale


class Interpolate(nn.Module):
    """Interpolates input tensor to a specified size.

    Args:
        size: Target output size as ``(height, width)``.
    """

    def __init__(self, size: _size_2_t):
        super().__init__()
        self.size = get_image_size(size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Resizes input tensor to target size.

        Args:
            input: Tensor to interpolate.
            
        Returns:
            Interpolated tensor.
        """
        return F.interpolate(input, self.size)

# endregion
