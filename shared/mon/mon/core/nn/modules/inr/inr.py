#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Implicit Neural Representations (INR) layers and networks."""

__all__ = [
    "INRLayer",
    "create_coords",
    "create_patches",
    "ff_embedding",
    "interpolate_image",
]

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- Utils -----
def create_coords(size: int) -> torch.Tensor:
    """Creates a coordinate grid.

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
    """Extracts patches into channels of a tensor.

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
            kernel[i * kernel_size + j, :, i, j] = 1
    
    im_padded = nn.ReflectionPad2d(kernel_size // 2)(image)
    extracted = F.conv2d(im_padded, kernel, padding=0)
    return torch.movedim(extracted, 1, -1)


def interpolate_image(image: torch.Tensor, size: int) -> torch.Tensor:
    """Resizes image to a new square resolution.

    Args:
        image: Image as a ``torch.Tensor`` of shape :math:`(B, C, H, W)` in
            range :math:`[0, 1]`.
        size: Target square size.

    Returns:
        A resized tensor as a ``torch.Tensor`` of shape :math:`(B, C, size, size)`.
    """
    return F.interpolate(image, size=(size, size), mode="bicubic")


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


class INRLayer(nn.Module):
    """Implements a general INR layer which automatically selects the nonlinearity.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        nonlinear: Nonlinearity type. Default: ``"sine"``.
        w0: Frequency scaling factor. Default: ``30.0``.
        is_first: First layer flag. Default: ``False``.
        is_last: Last layer flag, forces "sigmoid", as ``bool``. Default: ``False``.
    """
    
    INR_AF = Literal["sigmoid", "tanh", "relu", "sine", "gauss", "finer"]
    
    def __init__(
        self,
        in_features : int,
        out_features: int,
        bias        : bool   = True,
        nonlinear   : INR_AF = "sine",
        w0          : float  = 30.0,
        is_first    : bool   = False,
        is_last     : bool   = False,
        **kwargs
    ):
        super().__init__()
        if is_last:
            nonlinear = "sigmoid"
        
        layer_args = {
            "in_features" : in_features,
            "out_features": out_features,
            "bias"        : bias
        } | kwargs
        
        if nonlinear == "sigmoid":
            self.nonlinear = SigmoidLayer(**layer_args)
        elif nonlinear == "tanh":
            self.nonlinear = TanhLayer(**layer_args)
        elif nonlinear == "relu":
            self.nonlinear = ReLULayer(**layer_args)
        elif nonlinear == "sine":
            self.nonlinear = SineLayer(**layer_args, is_first=is_first, w0=w0, init_weights=not is_last)
        elif nonlinear == "gauss":
            self.nonlinear = GaussLayer(**layer_args)
        elif nonlinear == "finer":
            self.nonlinear = FINERLayer(**layer_args, is_first=is_first, w0=w0)
        else:
            raise ValueError(f"``nonlinear`` must be supported type, got {nonlinear}.")
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(input)
