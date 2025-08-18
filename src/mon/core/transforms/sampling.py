#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements resizing transformations."""

__all__ = [
    "pair_downsample",
]

import torch
import torch.nn.functional as F


def pair_downsample(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample an image tensor into a pair to half resolution.
    
    Args:
        image: Image as a ``torch.Tensor`` of shape :math:`(B, C, H, W)`.

    Returns:
        Two downsampled images, each of shape :math:`(B, C, H/2, W/2)`.

    Notes:
        Averages diagonal pixels in non-overlapping patches:
            ---------------------        ---------------------
            | A1 | B1 | A2 | B2 |        | A1+D1/2 | A2+D2/2 |
            | C1 | D1 | C2 | D2 |        | A3+D3/2 | A4+D4/2 |
            ---------------------  ===>  ---------------------
            | A3 | B3 | A4 | B4 |        | B1+C1/2 | B2+C2/2 |
            | C3 | D3 | C4 | D4 |        | B3+C3/2 | B4+C4/2 |
            ---------------------        ---------------------

    References:
        - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing
    """
    c       = image.shape[1]
    filter1 = torch.Tensor([[[[0, 0.5], [0.5, 0]]]]).to(image.dtype).to(image.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = torch.Tensor([[[[0.5, 0], [0, 0.5]]]]).to(image.dtype).to(image.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = F.conv2d(image, filter1, stride=2, groups=c)
    output2 = F.conv2d(image, filter2, stride=2, groups=c)
    return output1, output2
