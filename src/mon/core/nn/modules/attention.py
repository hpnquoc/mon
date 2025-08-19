#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements attention layers."""

__all__ = [
    "SimAM",
]

import torch
import torch.nn as nn


# ----- Parameter-Free Attention -----
class SimAM(nn.Module):
    """Implement Simple, Parameter-Free Attention Module (SimAM).

    Args:
        e_lambda: Regularization parameter for energy. Default: ``1e-4``.

    References:
        - Code: https://github.com/ZjjConan/SimAM
    """

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid  = nn.Sigmoid()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        b, c, h, w = input.shape
        n          = w * h - 1
        d          = (input - input.mean(dim=[2, 3], keepdim=True)).pow(2)  # [B, C, H, W]
        v          = d.sum(dim=[2, 3], keepdim=True) / n   # [B, C, 1, 1]
        e_inv      = d / (4 * (v + self.e_lambda)) + 0.5   # [B, C, H, W]
        return input * self.sigmoid(e_inv)
