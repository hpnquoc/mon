#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "create_coords",
    "create_patches",
    "ff_embedding",
    "filter_up",
    "interpolate_image",
]

import numpy as np
import torch

from mon.core import nn
from mon.core.dtypes.image import FastGuidedFilter
from mon.core.nn import functional as F


def create_coords(size: int) -> torch.Tensor:
    """Creates a coordinates grid.

    Args:
        size: The size of the coordinates grid.
    """
    h, w   = size, size
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w)))
    coords = torch.from_numpy(coords).float()
    return coords


def create_patches(image: torch.Tensor, kernel_size: int = 1) -> torch.Tensor:
    """Creates a tensor where the channel contains patch information."""
    b, c, h, w = image.shape
    kernel     = torch.zeros((kernel_size ** 2, c, kernel_size, kernel_size)).to(image.device)
    for i in range(kernel_size):
        for j in range(kernel_size):
            # kernel[int(torch.sum(kernel).item()), 0, i, j] = 1
            kernel[i + j * kernel_size, 0, i, j] = 1

    pad       = nn.ReflectionPad2d(kernel_size // 2)
    im_padded = pad(image)
    extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
    return torch.movedim(extracted, 0, -1)


def interpolate_image(image: torch.Tensor, size: int) -> torch.Tensor:
    """Reshapes the image based on new resolution."""
    # return F.interpolate(image, size=(down_size, down_size), mode="bicubic")
    return F.interpolate(image, size=(size, size), mode="area")


def ff_embedding(p: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
    if B is None:
        return p
    else:
        x_proj    = (2. * np.pi * p) @ B.T
        embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
        return embedding


def filter_up(
    x_lr: torch.Tensor,
    y_lr: torch.Tensor,
    x_hr: torch.Tensor,
    kernel_size: int = 7
) -> torch.Tensor:
    """Applies the guided filter to upscale the predicted image. """
    guided_filter = FastGuidedFilter(kernel_size)
    y_hr = guided_filter(x_lr, y_lr, x_hr)
    y_hr = torch.clip(y_hr, 0, 1)
    return y_hr
