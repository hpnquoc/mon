#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements geometry functions for contours/segments."""

__all__ = [
    "convert_contour",
    "convert_contour_voc_to_yolo",
    "convert_contour_yolo_to_voc",
    "denormalize_contour",
    "normalize_contour",
]

import numpy as np

from mon.constants import ShapeCode


# ----- Normalize -----
def normalize_contour(contour: np.ndarray, height: int, width: int) -> np.ndarray:
    """Normalize contour points to [0.0, 1.0].

    Args:
        contour: Contour points as ``np.ndarray`` in [N, 2] format.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Normalized contour points as ``np.ndarray`` in [N, 2] format.
    """
    x, y, *_ = contour.T
    x_norm   = x / width
    y_norm   = y / height
    return np.stack((x_norm, y_norm), axis=-1)


def denormalize_contour(contour: np.ndarray, height: int, width: int) -> np.ndarray:
    """Denormalize contour points to pixel coordinates.

    Args:
        contour: Normalized points as ``np.ndarray`` in [N, 2], range [0.0, 1.0].
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Denormalized contour points as ``np.ndarray`` in [N, 2].
    """
    x_norm, y_norm, *_ = contour.T
    x = x_norm * width
    y = y_norm * height
    return np.stack((x, y), axis=-1)


# ----- Conversion -----
convert_contour_voc_to_yolo = normalize_contour
convert_contour_yolo_to_voc = denormalize_contour


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
            return convert_contour_voc_to_yolo(contour, height, width)
        case ShapeCode.YOLO2VOC:
            return convert_contour_yolo_to_voc(contour, height, width)
        case _:
            return contour
