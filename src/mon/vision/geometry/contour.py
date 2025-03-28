#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements geometry functions for contours/segments."""

from __future__ import annotations

__all__ = [
    "contour_voc_to_yolo",
    "contour_yolo_to_voc",
    "convert_contour",
    "denormalize_contour",
    "normalize_contour",
]

import numpy as np

from mon.globals import ShapeCode


# region Conversion

def normalize_contour(contour: np.ndarray, height: int, width: int) -> np.ndarray:
    """Normalizes contour points to the range [0.0, 1.0].

    Args:
        contour: Contour points as numpy.ndarray in [N, 2] format.
        height: Image height in pixels.
        width: Image width in pixels.
    
    Returns:
        Normalized contour points in [N, 2] format.
    """
    contour  = contour.copy()
    x, y, *_ = contour.T
    x_norm   = x / width
    y_norm   = y / height
    contour  = np.stack((x_norm, y_norm), axis=-1)
    return contour


def denormalize_contour(contour: np.ndarray, height: int, width: int) -> np.ndarray:
    """Denormalizes contour points from range [0.0, 1.0] to pixel coordinates.

    Args:
        contour: Normalized contour points as numpy.ndarray in [N, 2] format.
        height: Image height in pixels.
        width: Image width in pixels.
    Returns:
        Denormalized contour points in [N, 2] format.
    """
    contour = contour.copy()
    x_norm, y_norm, *_ = contour.T
    x       = x_norm * width
    y       = y_norm * height
    contour = np.stack((x, y), axis=-1)
    return contour


contour_voc_to_yolo = normalize_contour
contour_yolo_to_voc = denormalize_contour


def convert_contour(
    contour: np.ndarray,
    code   : ShapeCode | int,
    height : int,
    width  : int
) -> np.ndarray:
    """Convert bounding box."""
    code = ShapeCode.from_value(value=code)
    match code:
        case ShapeCode.VOC2YOLO:
            return contour_voc_to_yolo(contour, height, width)
        case ShapeCode.YOLO2VOC:
            return contour_yolo_to_voc(contour, height, width)
        case _:
            return contour

# endregion
