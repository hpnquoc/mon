#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Metric.

This module implements image metrics.
"""

from __future__ import annotations

__all__ = [
    "GoodLookingImageMetric",
]

import torch
from torch import nn


# region Good-Looking Image Metric

class GoodLookingImageMetric(nn.Module):
    """A good-looking image is one with a low well-exposedness, but high
    contrast and saturation.
    
    References:
        https://github.com/VinAIResearch/PSENet-Image-Enhancement/blob/main/source/iqa.py
    """
    
    def __init__(self, exposed_level: float = 0.5, pool_size: int = 25):
        super().__init__()
        self.exposed_level = exposed_level
        self.pool_size     = pool_size
        self.mean_pool     = nn.Sequential(nn.ReflectionPad2d(self.pool_size // 2), nn.AvgPool2d(self.pool_size, stride=1))
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        eps         = 1 / 255.0
        max_rgb     = torch.max(images, dim=1, keepdim=True)[0]
        min_rgb     = torch.min(images, dim=1, keepdim=True)[0]
        saturation  = (max_rgb - min_rgb + eps) / (max_rgb + eps)
        mean_rgb    = self.mean_pool(images).mean(dim=1, keepdim=True)
        exposedness = torch.abs(mean_rgb - self.exposed_level) + eps
        contrast    = self.mean_pool(images * images).mean(dim=1, keepdim=True) - mean_rgb ** 2
        return torch.mean((saturation * contrast) / exposedness, dim=[1], keepdim=True)
    
# endregion
