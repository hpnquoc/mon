#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mixture of Experts (MoE) Network.

This module implements the Mixture of Experts (MoE) network.
"""

from __future__ import annotations

__all__ = [
    "LayeredFeatureAggregation",
]

from typing import Any, Sequence

import torch
from torch import nn
from torch.nn.common_types import _size_2_t

from mon import core


# region Utils

def get_image_size(input: Any) -> tuple[int, int]:
    """Retrieves the size of an image.

    Args:
        input: Image data in any compatible format.

    Returns:
        Tuple of (height, width) in pixels.
    """
    from mon.vision.dtype import image as I
    return I.get_image_size(input)
    
# endregion


# region Layer

class LayeredFeatureAggregation(nn.Module):
    """Layered Feature Aggregation (LFA) fuses decoder layer features.

    Args:
        in_channels: List of input channel counts for each feature.
        out_channels: Number of output channels.
        size: Target size for upsampling. Default is ``None`` (no resizing).
    """

    def __init__(
        self,
        in_channels : list[int],
        out_channels: int,
        size        : _size_2_t = None
    ):
        super().__init__()
        self.in_channels  = core.to_int_list(in_channels)
        self.out_channels = out_channels
        self.num_experts  = len(self.in_channels)

        if not self.num_experts:
            raise ValueError("[in_channels] must not be empty")

        if size is not None:
            self.size    = get_image_size(size)
            self.resize  = nn.Upsample(size=self.size, mode="bilinear", align_corners=False)
            self.linears = nn.ModuleList([
                nn.Conv2d(in_c, self.out_channels, 1) for in_c in self.in_channels
            ])
        else:
            self.size    = None
            self.resize  = None
            self.linears = None

        self.conv    = nn.Conv2d(self.out_channels * self.num_experts, self.out_channels, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        """Aggregates layered features with attention.

        Args:
            input: Sequence of feature tensors ``[B, C_i, H, W]``.

        Returns:
            Aggregated feature tensor ``[B, C_out, H, W]``.

        Raises:
            ValueError: If number of input tensors mismatches ``num_experts``.
        """
        if len(input) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} input tensors, but got [{len(input)}]")

        r = [
            self.linears[i](self.resize(inp)) if self.resize else self.linears[i](inp) if self.linears else inp
            for i, inp in enumerate(input)
        ]
        o_s = torch.cat(r, dim=1)  # [B, C_out * num_experts, H, W]
        w   = self.softmax(self.conv(o_s))  # [B, C_out, H, W]
        o_w = torch.stack([r[i] * w[:, i:i+1] for i in range(len(r))], dim=1)  # [B, num_experts, C_out, H, W]
        return torch.sum(o_w, dim=1)  # [B, C_out, H, W]

# endregion
