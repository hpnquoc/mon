#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements pixel attention layers."""

from __future__ import annotations

__all__ = [
    "PAM",
    "PixelAttentionModule",
]

import torch
from torch import nn
from torch.nn.common_types import _size_2_t


# region Pixel Attention Module

class PixelAttentionModule(nn.Module):
    """Pixel Attention Module for spatial feature enhancement.

    Args:
        channels: Number of input channels as ``int``.
        reduction_ratio: Channel reduction ratio as ``int``.
        kernel_size: Size of the convolution kernel as ``int`` or ``tuple[int, int]``.
    """

    def __init__(
        self,
        channels       : int,
        reduction_ratio: int,
        kernel_size    : _size_2_t
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(
                in_channels  = channels,
                out_channels = channels // reduction_ratio,
                kernel_size  = kernel_size
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels  = channels // reduction_ratio,
                out_channels = 1,
                kernel_size  = kernel_size
            )
        )
        self.act = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies pixel attention to the input.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            pixel attention applied.
        """
        return input * self.act(self.fc(input))


PAM = PixelAttentionModule

# endregion
