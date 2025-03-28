#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements various image channel priors."""

from __future__ import annotations

__all__ = [
    "bright_channel_prior",
    "dark_channel_prior",
]

import cv2
import kornia
import numpy as np
import torch
from torch.nn.common_types import _size_2_t

from mon import core


def bright_channel_prior(
    image      : torch.Tensor | np.ndarray,
    kernel_size: _size_2_t
) -> torch.Tensor | np.ndarray:
    """Gets bright channel prior from an RGB image.

    Args:
        image: RGB image as ``torch.Tensor`` [B, C, H, W] or ``numpy.ndarray`` [H, W, C].
        kernel_size: Window size as int or tuple.

    Returns:
        Bright channel prior as ``torch.Tensor`` or ``numpy.ndarray``.
    """
    kernel_size = core.to_2tuple(kernel_size)
    if isinstance(image, torch.Tensor):
        bright_channel = torch.max(image, dim=1)[0]
        kernel         = torch.ones(kernel_size[0], kernel_size[0])
        bcp            = kornia.morphology.erosion(bright_channel, kernel)
    elif isinstance(image, np.ndarray):
        bright_channel = np.max(image, axis=2)
        kernel         = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        bcp            = cv2.erode(bright_channel, kernel)
    else:
        raise ValueError(f"[image] must be ``torch.Tensor`` or ``numpy.ndarray``, "
                         f"but got [{type(image)}].")
    return bcp


def dark_channel_prior(
    image      : torch.Tensor | np.ndarray,
    kernel_size: int
) -> torch.Tensor | np.ndarray:
    """Gets dark channel prior from an RGB image.

    Args:
        image: RGB image as ``torch.Tensor`` [B, C, H, W] or ``numpy.ndarray`` [H, W, C].
        kernel_size: Window size as ``int``.

    Returns:
        Dark channel prior as ``torch.Tensor`` or ``numpy.ndarray``.
    """
    kernel_size = core.to_2tuple(kernel_size)
    if isinstance(image, torch.Tensor):
        dark_channel = torch.min(image, dim=1)[0]
        kernel       = torch.ones(kernel_size[0], kernel_size[1])
        dcp          = kornia.morphology.erosion(dark_channel, kernel)
    elif isinstance(image, np.ndarray):
        dark_channel = np.min(image, axis=2)
        kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        dcp          = cv2.erode(dark_channel, kernel)
    else:
        raise ValueError(f"[image] must be ``torch.Tensor`` or ``numpy.ndarray``, "
                         f"but got [{type(image)}].")
    return dcp
