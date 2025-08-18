#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements custom activation layers."""

__all__ = [
    "SimpleGate",
    "Sine",
]

import torch
import torch.nn as nn


class SimpleGate(nn.Module):
    """Simple gate activation unit from 'Simple Baselines for Image Restoration'.

    References:
        - https://arxiv.org/pdf/2204.04676.pdf
    """
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies simple gate activation by chunking and multiplication.

        Args:
            input: Input tensor as a ``torch.Tensor`` with shape [B, C, H, W],
                where ``C`` is even.
    
        Returns:
            Output tensor as a ``torch.Tensor`` with shape [B, C/2, H, W] after chunking
            and multiplication.
        """
        x1, x2 = input.chunk(chunks=2, dim=1)
        return x1 * x2


class Sine(nn.Module):
    """Sine activation unit.

    Args:
        w0: Frequency scaling factor as ``float``. Default is ``1.0``.

    References:
        - Code: https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py
    """

    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies sine activation.

        Args:
            input: Input tensor as a ``torch.Tensor`` of any shape.

        Returns:
            Output tensor as a ``torch.Tensor`` with same shape as input.
        """
        return torch.sin(self.w0 * input)

    def forward_with_intermediate(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies sine activation and returns intermediate value.

        Args:
            input: Input tensor as a ``torch.Tensor`` of any shape.

        Returns:
            Tuple of (sine output ``torch.Tensor``,
                      intermediate value ``torch.Tensor``) with same shape as input.
        """
        intermediate = self.w0 * input  # Corrected: Removed undefined self.linear
        return torch.sin(intermediate), intermediate
