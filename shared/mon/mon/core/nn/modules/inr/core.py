#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements core INR layers and functions."""

__all__ = [
    "create_coords",
    "create_patches",
    "ff_embedding",
]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- Utils -----
def create_coords(size: int) -> torch.Tensor:
    """Creates a coordinate grid for INF.

    Args:
        size: Size of the square coordinates grid.

    Returns:
        A ``torch.Tensor`` of shape :math:`(size, size, 2)` with normalized
        coords.
    """
    h, w   = size, size
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w)))
    return torch.from_numpy(coords).float()


def create_patches(image: torch.Tensor, kernel_size: int = 1) -> torch.Tensor:
    """Creates a tensor where the channel contains patch information.

    Args:
        image: Image as a ``torch.Tensor`` of shape :math:`(B, C, H, W)` in
            range :math:`[0, 1]`.
        kernel_size: Size of square patches. Default: ``1``.

    Returns:
        A ``torch.Tensor`` with patches in channels of shape :math:`(B, H', W', K^2)`.
    """
    if image.ndim != 4:
        raise ValueError(f"``image`` must be a torch.Tensor of shape (B, C, H, W), got {image.shape}.")
    
    b, c, h, w = image.shape
    kernel     = torch.zeros(kernel_size ** 2, c, kernel_size, kernel_size, device=image.device)
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i + j * kernel_size, 0, i, j] = 1
    
    pad       = nn.ReflectionPad2d(kernel_size // 2)
    im_padded = pad(image)
    extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
    return torch.movedim(extracted, 0, -1)


def ff_embedding(p: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
    """Applies Fourier feature embedding to input tensor.

    Args:
        p: Input tensor to embed.
        B: Projection matrix as a ``torch.Tensor``. Default: ``None``.

    Returns:
        An embedded ``torch.Tensor`` with sine and cosine features.
    """
    if B is None:
        return p
    x_proj = (2 * np.pi * p) @ B.T
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
