#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements image feature priors (structural and semantic priors).

This category involves identifying and isolating structural or semantic features
(e.g., edges, boundaries) in an image.
"""

__all__ = [
    "BoundaryAwarePrior",
    "boundary_aware_prior",
]

from typing import Union

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn

from mon.core.dtypes.image import filtering, utils


def boundary_aware_prior(
    image      :  Union[torch.Tensor, np.ndarray],
    eps        : float = 0.05,
    as_gradient: bool  = False,
    normalized : bool  = False,
) ->  Union[torch.Tensor, np.ndarray]:
    """Get the boundary prior from an RGB or grayscale image.

    Args:
        image: An RGB or grayscale image as a
            ``torch.Tensor`` (i.e., of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`)
            or ``numpy.ndarray`` (i.e., of shape :math:`(H, W, C)` in :math:`[0, 255]`).
        eps: Threshold to remove weak edges. Default: ``0.05``.
        as_gradient: If ``True``, returns the gradient image instead of binary boundary.
            Default: ``False``.
        normalized: L1 norm of the kernel is set to 1 if ``True``. Default: ``False``.

    Returns:
        The boundary prior with similar type and format as the input ``image``.

    Raises:
        ValueError: If ``image`` type is not supported.
    """
    if isinstance(image, torch.Tensor):
        gradient = kornia.filters.sobel(image, normalized=normalized, eps=1e-6)
        g_max    = torch.max(gradient)
        gradient = gradient / g_max
        boundary = (gradient > eps).float()
    elif isinstance(image, np.ndarray):
        if utils.is_color(image):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gradient = filtering.sobel_filter(image, kernel_size=3)
        g_max    = np.max(gradient)
        gradient = gradient / g_max
        boundary = (gradient > eps).float()
        return boundary
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
    
    # return boundary, gradient
    if as_gradient:
        return gradient
    else:
        return boundary


class BoundaryAwarePrior(nn.Module):
    """Get the boundary prior from an RGB or grayscale image.

    Args:
        eps: Threshold to remove weak edges. Default: ``0.05``.
        as_gradient: If ``True``, returns the gradient image instead of binary boundary.
            Default: ``False``.
        normalized: L1 norm of the kernel is set to 1 if ``True``. Default: ``False``.
    """
    
    def __init__(self, eps: float = 0.05, as_gradient: bool = False, normalized: bool = False):
        super().__init__()
        self.eps        = eps
        self.as_gradient = as_gradient
        self.normalized = normalized
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return boundary_aware_prior(image, self.eps, self.as_gradient, self.normalized)
