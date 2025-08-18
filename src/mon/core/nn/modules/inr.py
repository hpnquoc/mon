#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Implicit Neural Representations (INR) layers and networks."""

__all__ = [

]

import numpy as np
import torch
import torch.nn as nn


# ----- Linear Layer -----
class SineLayer(nn.Module):
    """Applies linear transformation with sine activation.

    Args:
        in_features: Number of input channels as ``int``.
        out_features: Number of output channels as ``int``.
        w0: Sine frequency factor as ``float``. Default is ``30.0``.
        is_first: First layer flag for weight initialization as ``bool``. Default is ``False``.
        bias: Uses bias in linear layer if ``True``. Default is ``True``.
        init_weights: Initializes weights if ``True``. Default is ``True``.

    References:
        - https://github.com/vishwa91/wire/blob/main/modules/siren.py
    """

    def __init__(
        self,
        in_features : int,
        out_features: int,
        w0          : float = 30.0,
        is_first    : bool  = False,
        bias        : bool  = True,
        init_weights: bool  = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.w0          = w0
        self.is_first    = is_first
        self.linear      = nn.Linear(in_features, out_features, bias=bias)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        """Initializes linear layer weights based on layer position."""
        with torch.no_grad():
            bound = 1 / self.in_features if self.is_first else np.sqrt(6 / self.in_features) / self.w0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with linear layer and sine.

        Args:
            x: Input tensor as a ``torch.Tensor``.

        Returns:
            Sine-transformed tensor as a ``torch.Tensor``.
        """
        return torch.sin(self.w0 * self.linear(x))

    def forward_with_intermediate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transforms input and returns intermediate result.

        Args:
            x: Input tensor as a ``torch.Tensor``.

        Returns:
            Tuple of (sine-transformed tensor as a ``torch.Tensor``, intermediate tensor as a ``torch.Tensor``).
        """
        intermediate = self.w0 * self.linear(x)
        return torch.sin(intermediate), intermediate
