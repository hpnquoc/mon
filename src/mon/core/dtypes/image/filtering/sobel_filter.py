#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Sobel filter/operator."""

__all__ = [
    "sobel_filter",
]

import cv2
import numpy as np

from mon.core.dtypes.image.utils import is_color


# ----- Sobel Filter -----
def sobel_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Applies Sobel filter to detect edges in an image.

    Args:
        image: RGB image as a ``numpy.ndarray`` of shape :math:`(H, W, C)` in :math:`[0, 255]`.
        kernel_size: Size of the Sobel kernel. Default: ``3``.
    
    Returns:
        Grayscale image with edge magnitudes.
    """
    if is_color(image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    return sobel_combined
