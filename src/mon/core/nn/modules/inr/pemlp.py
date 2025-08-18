#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements positional encoding MLP network."""

__all__ = [
    "PEMLP",
    "PositionalEncodingLayer",
]

import torch

from mon.nn.modules.inr import core


# ----- Positional Encoding Layer -----
class PositionalEncodingLayer(torch.nn.Module):
    """Applies positional encoding with sine and cosine functions.

    Args:
        in_features: Number of input channels as ``int``.
        N_freqs: Number of frequency bands as ``int``.
        logscale: Uses logarithmic frequency scale if ``True``. Default is ``True``.

    References:
        - https://github.com/liuzhen0212/FINER/blob/main/models.py
    """

    def __init__(
        self,
        in_features: int,
        N_freqs    : int,
        logscale   : bool = True
    ):
        super().__init__()
        self.N_freqs      = N_freqs
        self.in_features  = in_features
        self.funcs        = [torch.sin, torch.cos]
        self.out_features = in_features * (len(self.funcs) * N_freqs + 1)
        self.freq_bands   = (
            2 ** torch.linspace(0, N_freqs - 1, N_freqs)
            if logscale
            else torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes input with positional frequency bands.

        Args:
            x: Input tensor as a ``torch.Tensor`` of shape [..., in_channels].

        Returns:
            Encoded tensor as a ``torch.Tensor`` of shape [..., out_channels].
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(freq * x))
        return torch.cat(out, dim=-1)
    

# ----- PEMLP -----
class PEMLP(torch.nn.Module):
    """Implements positional encoding MLP network.

    Args:
        in_features: Number of input channels as ``int``.
        out_features: Number of output channels as ``int``.
        hidden_dim: Number of channels in hidden layers as ``int``.
        hidden_layers: Number of hidden layers as ``int``.
        N_freqs: Number of frequency bands for encoding as ``int``. Default is ``10``.
    """
    
    def __init__(
        self,
        in_features  : int,
        out_features : int,
        hidden_dim   : int,
        hidden_layers: int,
        N_freqs      : int = 10,
    ):
        super().__init__()
        self.encoding = PositionalEncodingLayer(in_features=in_features, N_freqs=N_freqs)
        
        layers  = [torch.nn.Linear(self.encoding.out_features, hidden_dim), torch.nn.ReLU(True)]
        layers += [torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(True)] * hidden_layers
        layers.append(torch.nn.Linear(hidden_dim, out_features))
        
        self.net = torch.nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates output from encoded image coordinates.

        Args:
            x: Input image tensor as a ``torch.Tensor`` for size reference.

        Returns:
            Output tensor as a ``torch.Tensor`` from network.
        """
        from mon import vision
        s, _   = vision.image_size(x)
        coords = core.create_coords(s).to(x.device)
        return self.net(self.enconding(coords))
