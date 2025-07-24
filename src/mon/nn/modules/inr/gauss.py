#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Gaussian network with Gauss layers."""

__all__ = [
    "GAUSS",
    "GaussLayer",
]

import torch

from mon.nn.modules.inr import core


# ----- Gauss's Activation Layer -----
class GaussLayer(torch.nn.Module):
    """Applies linear transformation with Gaussian activation.

    Args:
        in_features: Number of input channels as ``int``.
        out_features: Number of output channels as ``int``.
        scale: Gaussian scale factor as ``float``. Default is ``10.0``.
        bias: Uses bias in linear layer if ``True``. Default is ``True``.

    References:
        - https://github.com/vishwa91/wire/blob/main/modules/gauss.py
    """

    def __init__(
        self,
        in_features : int,
        out_features: int,
        scale       : float = 10.0,
        bias        : bool  = True,
    ):
        super().__init__()
        self.scale  = scale
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with linear layer and Gaussian.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            Gaussian-transformed tensor as ``torch.Tensor``.
        """
        return torch.exp(-(self.scale * self.linear(x))**2)


# ----- GAUSS -----
class GAUSS(torch.nn.Module):
    """Implements Gaussian network with Gauss layers.

    Args:
        in_features: Number of input channels as ``int``.
        out_features: Number of output channels as ``int``.
        hidden_dim: Number of channels in hidden layers as ``int``.
        hidden_layers: Number of hidden layers as ``int``.
        scale: Gaussian scale factor as ``float``. Default is ``30.0``.
        bias: Uses bias in layers if ``True``. Default is ``True``.

    References:
        - https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_features  : int,
        out_features : int,
        hidden_dim   : int,
        hidden_layers: int,
        scale        : float = 30.0,
        bias         : bool  = True,
    ):
        super().__init__()
        self.net = torch.nn.Sequential(
            GaussLayer(in_features, hidden_dim, scale, bias=bias),
            *[GaussLayer(hidden_dim, hidden_dim, scale, bias=bias) for _ in range(hidden_layers)],
            torch.nn.Linear(hidden_dim, out_features)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates output from image coordinates.

        Args:
            x: Input image tensor as ``torch.Tensor`` for size reference.

        Returns:
            Output tensor as ``torch.Tensor`` from network.
        """
        from mon import vision
        s, _   = vision.image_size(x)
        coords = core.create_coords(s).to(x.device)
        return self.net(coords)
