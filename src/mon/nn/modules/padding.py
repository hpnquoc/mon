#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements padding layers."""

from __future__ import annotations

__all__ = [
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "ZeroPad2d",
    "get_same_padding",
    "get_symmetric_padding",
    "pad_same",
    "to_same_padding",
]

import math

from torch.nn.common_types import _size_2_t
from torch.nn.modules.padding import *


# region Helper Function

def get_same_padding(
    x          : int,
    kernel_size: int,
    stride     : int,
    dilation   : int
) -> int:
    """Calculates TensorFlow-like 'same' padding for 1D convolution.

    Args:
        x: Input size (e.g., height or width).
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution.
        dilation: Dilation of the convolution.

    Returns:
        Padding value for one dimension to achieve ``'same'`` output size.
    """
    return max(
        (math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0
    )


def get_symmetric_padding(
    kernel_size: int,
    stride     : int = 1,
    dilation   : int = 1
) -> int:
    """Calculates symmetric padding for a convolution.

    Args:
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution. Default is ``1``.
        dilation: Dilation of the convolution. Default is ``1``.

    Returns:
        Symmetric padding value for one dimension.
    """
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


def to_same_padding(
    kernel_size: _size_2_t,
    padding    : _size_2_t = None
) -> int | list[int] | None:
    """Converts padding to symmetric 'same' style if None.

    Args:
        kernel_size: Size of the convolutional kernel.
        padding: Padding for the convolution. Default is ``None``.

    Returns:
        Symmetric padding ``(kernel_size // 2)`` if padding is ``None``, else padding.
    """
    if padding is None:
        if isinstance(kernel_size, int):
            return kernel_size // 2
        if isinstance(kernel_size, (tuple, list)):
            return [k // 2 for k in kernel_size]
    return padding


import torch
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t

def pad_same(
    input      : torch.Tensor,
    kernel_size: _size_2_t,
    stride     : _size_2_t,
    dilation   : _size_2_t = (1, 1),
    value      : float     = 0
) -> torch.Tensor:
    """Pads input tensor with ``'same'`` padding for convolution.

    Args:
        input: Input tensor [..., H, W].
        kernel_size: Size of the convolution kernel (H, W).
        stride: Stride of the convolution (H, W).
        dilation: Dilation of the convolution (H, W). Default is ``(1, 1)``.
        value: Padding value. Default is ``0``.

    Returns:
        Padded tensor with same spatial size post-convolution.
    """
    ih, iw = input.size()[-2:]
    pad_h  = get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w  = get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
    if pad_h > 0 or pad_w > 0:
        input = F.pad(input, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return input

# endregion
