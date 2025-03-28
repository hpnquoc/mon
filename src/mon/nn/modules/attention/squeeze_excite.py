#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Squeeze and Excite layers."""

from __future__ import annotations

__all__ = [
    "SqueezeExcitation",
    "SqueezeExciteC",
    "SqueezeExciteL",
]

import torch
from torch import nn
from torchvision.ops.misc import SqueezeExcitation


class SqueezeExciteC(nn.Module):
    """Squeeze and Excite layer using Conv2d from 'Squeeze and Excitation' paper.

    Args:
        channels: Number of input channels as ``int``.
        reduction_ratio: Channel reduction ratio as ``int``. Default is ``16``.
        bias: Adds bias to convolutions if ``True``. Default is ``False``.

    References:
        - https://amaarora.github.io/2020/07/24/SeNet.html#squeeze-and-excitation-block-in-pytorch
        - https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """
    
    def __init__(
        self,
        channels       : int,
        reduction_ratio: int  = 16,
        bias           : bool = False,
    ):
        super().__init__()
        self.avg_pool   = nn.AdaptiveAvgPool2d(1)  # squeeze
        self.excitation = nn.Sequential(
            nn.Conv2d(
                in_channels  = channels,
                out_channels = channels  // reduction_ratio,
                kernel_size  = 1,
                bias         = bias,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels  = channels  // reduction_ratio,
                out_channels = channels,
                kernel_size  = 1,
                bias         = bias,
            ),
            nn.Sigmoid()
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies squeeze and excite attention.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            channel attention applied.
        """
        return input * self.excitation(self.avg_pool(input))


class SqueezeExciteL(nn.Module):
    """Squeeze and Excite layer using Linear from 'Squeeze and Excitation' paper.

    Args:
        channels: Number of input channels as ``int``.
        reduction_ratio: Channel reduction ratio as ``int``. Default is ``16``.
        bias: Adds bias to linear layers if ``True``. Default is ``False``.

    References:
        - https://amaarora.github.io/2020/07/24/SeNet.html#squeeze-and-excitation-block-in-pytorch
        - https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """
    
    def __init__(
        self,
        channels       : int,
        reduction_ratio: int  = 16,
        bias           : bool = False,
    ):
        super().__init__()
        self.avg_pool   = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(
                in_features  = channels,
                out_features = channels // reduction_ratio,
                bias         = bias
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features  = channels // reduction_ratio,
                out_features = channels,
                bias         = bias
            ),
            nn.Sigmoid()
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies squeeze and excite attention.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            channel attention applied.
        """
        b, c, _, _ = input.shape
        y = self.avg_pool(input).view(b, c)      # [B, C, 1, 1] -> [B, C]
        y = self.excitation(y).view(b, c, 1, 1)  # [B, C] -> [B, C, 1, 1]
        return input * y
