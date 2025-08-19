#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements attention priors (saliency and focus maps).

This category includes methods that highlight important regions in an image, often used
in deep learning to guide processing or interpretation (e.g., attention maps in neural
networks).
"""

__all__ = [
    "BrightnessAttentionMap",
    "brightness_attention_map",
]

from typing import Union

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn

from mon.core.dtypes.image import utils


def brightness_attention_map(
    image: Union[torch.Tensor, np.ndarray],
    gamma: float = 2.5,
    ksize: int   = None,
) -> torch.Tensor:
    """Get the Brightness Attention Map (BAM) prior from an RGB image.

    This is a self-attention map extracted from the V-channel of a low-light
    image, multiplied to convolutional activations of all layers in the enhancement
    network. Brighter regions are given lower weights to avoid over-saturation,
    while preserving image details and enhancing contrast in dark regions effectively.

    Args:
        image: An RGB image as a
            ``torch.Tensor`` (i.e., of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`)
            or ``numpy.ndarray`` (i.e., of shape :math:`(H, W, C)` in :math:`[0, 255]`).
        gamma: Parameter controlling the curvature of the map. Default: ``2.5``.
        ksize: Window size for denoising operation. Default: ``None``.

    Returns:
        Brightness enhancement map as a ``torch.Tensor`` or ``numpy.ndarray``
        matching the ``image`` type and format.
    """
    if isinstance(image, torch.Tensor):
        if ksize:
            image = kornia.filters.median_blur(image, ksize)
            # image = kornia.filters.bilateral_blur(image, denoise_ksize, 0.1, (1.5, 1.5))
        hsv = kornia.color.rgb_to_hsv(image)
        v   = utils.channel(image=hsv, index=(2, 3), keep_dim=True)  # hsv[:, 2:3, :, :]
        bam = torch.pow((1 - v), gamma)
    elif isinstance(image, np.ndarray):
        if ksize:
            image = cv2.medianBlur(image, ksize)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if hsv.dtype != np.float32:
            hsv  = hsv.astype("float32")
            hsv /= 255.0
        v   = utils.channel(image=hsv, index=(2, 3), keep_dim=True)  # hsv[:, :, 2:3]
        bam = np.power((1 - v), gamma)
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
    return bam


class BrightnessAttentionMap(nn.Module):
    """Get the Brightness Attention Map (BAM) prior from an RGB image.

    This is a self-attention map extracted from the V-channel of a low-light
    image, multiplied to convolutional activations of all layers in the enhancement
    network. Brighter regions are given lower weights to avoid over-saturation,
    while preserving image details and enhancing contrast in dark regions effectively.

    Args:
        gamma: Parameter controlling the curvature of the map. Default: ``2.5``.
        ksize: Window size for denoising operation. Default: ``None``.
    """
    
    def __init__(self, gamma: float = 2.5, ksize: int = None):
        super().__init__()
        self.gamma = gamma
        self.ksize = ksize
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return brightness_attention_map(image, self.gamma, self.ksize)
