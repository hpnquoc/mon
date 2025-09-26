#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Positional Encoding (PE) MLP.

References:
    - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
"""

__all__ = [
    "PositionalEncoding",
    "PE_MLP",
]

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    r"""Implements the Positional Encoding (PE) as described in the NeRF paper.
    Given an input :math:`x`, the encoding is defined as:
    .. math::
        \gamma(x) = \left( x, \sin(2^0 \pi x), \cos(2^0 \pi x), \ldots,
        \sin(2^{L-1} \pi x), \cos(2^{L-1} \pi x) \right)
    
    where :math:`L` is the number of frequency bands.
    
    Args:
        in_features: Size of each input sample.
        N_freqs: Number of frequency bands.
        logscale: If ``True``, frequency bands are spaced logarithmically. Default: ``True``.
        
    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """

    def __init__(
        self,
        in_features: int,
        N_freqs    : int,
        logscale   : bool = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_features  = in_features
        self.N_freqs      = N_freqs
        self.funcs        = [torch.sin, torch.cos]
        self.out_features = in_features * (len(self.funcs) * N_freqs + 1)
        
        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, -1)
    

class PE_MLP(nn.Module):
    """Implements the Positional Encoding (PE) MLP.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        hidden_dim: Hidden channel dimensions.
        hidden_layers: Number of hidden layers.
        N_freqs: Number of frequency bands for positional encoding. Default: ``10``.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
    
    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_features  : int,
        out_features : int,
        hidden_dim   : int,
        hidden_layers: int,
        N_freqs      : int  = 10,
        bias         : bool = True,
    ):
        super().__init__()
        self.encoding = PositionalEncoding(in_features=in_features, N_freqs=N_freqs)
        
        # First layer
        self.net = []
        self.net.append(nn.Linear(self.encoding.out_features, hidden_dim, bias=bias))
        self.net.append(nn.ReLU(True))
        # Hidden layers
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            self.net.append(nn.ReLU(True))
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features, bias=bias)
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(self.encoding(coords))
