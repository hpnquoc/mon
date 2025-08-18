#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements custom upsampling and downsampling layers."""

__all__ = [
    "UpsampleConv2d",
]

import math

import torch
import torch.nn as nn


class UpsampleConv2d(nn.Module):
    """Upsamples 2D data using a transposed convolutional layer.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv       = torch.nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.in_channels  = in_channels
        self.out_channels = out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Upsamples input tensor via transposed convolution.

        Args:
            input: Tensor as a ``torch.Tensor`` with shape [B, L, C].

        Returns:
            Upsampled tensor as a ``torch.Tensor`` with shape [B, 4*L, C].
        """
        b, l, c = input.shape
        h       = w = int(math.sqrt(l))
        x       = input.transpose(1, 2).view(b, c, h, w)
        x       = self.deconv(x).flatten(2).transpose(1, 2)
        return x

    def flops(self, h: int, w: int) -> float:
        """Calculates FLOPs for the upsampling operation.

        Args:
            h: Input height as ``int``.
            w: Input width as ``int``.

        Returns:
            Total FLOPs as ``float``.
        """
        return h * w * self.in_channels * self.out_channels * 16
