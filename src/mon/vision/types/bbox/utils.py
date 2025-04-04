#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements general-purpose utilities for bounding box.

Common Tasks:
    - Property accessors
    - Validation checks
    - Miscellaneous
"""

__all__ = [
    "bbox_area",
    "bbox_center",
    "bbox_corners",
    "bbox_corners_pts",
    "enclosing_bbox",
]

import numpy as np


# ----- Access -----
def bbox_area(bbox: np.ndarray) -> np.ndarray:
    """Compute area of bounding box(es).

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4] or [N, 4], XYXY format.

    Returns:
        Area(s) as ``np.ndarray`` in [1] or [N] shape.

    Raises:
        ValueError: If ``bbox`` is not 1D or 2D.
    """
    if bbox.ndim == 1:
        bbox = np.expand_dims(bbox, 0)
    if bbox.ndim != 2:
        raise ValueError(f"[bbox] must be 1D or 2D, got {bbox.ndim}D.")
    x1 = bbox[..., 0]
    y1 = bbox[..., 1]
    x2 = bbox[..., 2]
    y2 = bbox[..., 3]
    return (x2 - x1) * (y2 - y1)


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    """Compute center(s) of bounding box(es).

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4] or [N, 4], XYXY format.

    Returns:
        Center(s) as ``np.ndarray`` in [1, 2] or [N, 2], [cx, cy] format.

    Raises:
        ValueError: If bbox is not 1D or 2D.
    """
    if bbox.ndim == 1:
        bbox = np.expand_dims(bbox, 0)
    if bbox.ndim != 2:
        raise ValueError(f"[bbox] must be 1D or 2D, got {bbox.ndim}D.")
    x1 = bbox[..., 0]
    y1 = bbox[..., 1]
    x2 = bbox[..., 2]
    y2 = bbox[..., 3]
    cx = x1 + (x2 - x1) / 2.0
    cy = y1 + (y2 - y1) / 2.0
    return np.stack((cx, cy), -1)


def bbox_corners(bbox: np.ndarray) -> np.ndarray:
    """Get corner(s) of bounding box(es).

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4] or [N, 4], XYXY format

    Returns:
        Corners as ``np.ndarray`` in [N, 8], [x1, y1, x2, y2, x3, y3, x4, y4] format

    Raises:
        ValueError: If ``bbox`` is not 1D or 2D
    """
    if bbox.ndim == 1:
        bbox = np.expand_dims(bbox, 0)
    if bbox.ndim != 2:
        raise ValueError(f"[bbox] must be 1D or 2D, got {bbox.ndim}D.")
    x1   = bbox[..., 0]
    y1   = bbox[..., 1]
    x2   = bbox[..., 2]
    y2   = bbox[..., 3]
    w    = x2 - x1
    h    = y2 - y1
    c_x1 = x1
    c_y1 = y1
    c_x2 = x1 + w
    c_y2 = y1
    c_x3 = x2
    c_y3 = y2
    c_x4 = x1
    c_y4 = y1 + h
    return np.hstack((c_x1, c_y1, c_x2, c_y2, c_x3, c_y3, c_x4, c_y4))


def bbox_corners_pts(bbox: np.ndarray) -> np.ndarray:
    """Get corner(s) of bounding box(es) as points.

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4] or [N, 4], XYXY format.

    Returns:
        Corners as ``np.ndarray`` in
        [N, 4, 2], [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] format.

    Raises:
        ValueError: If ``bbox`` is not 1D or 2D.
    """
    if bbox.ndim == 1:
        bbox = np.expand_dims(bbox, 0)
    if bbox.ndim != 2:
        raise ValueError(f"[bbox] must be 1D or 2D, got {bbox.ndim}D.")
    x1   = bbox[..., 0]
    y1   = bbox[..., 1]
    x2   = bbox[..., 2]
    y2   = bbox[..., 3]
    w    = x2 - x1
    h    = y2 - y1
    c_x1 = x1
    c_y1 = y1
    c_x2 = x1 + w
    c_y2 = y1
    c_x3 = x2
    c_y3 = y2
    c_x4 = x1
    c_y4 = y1 + h
    return np.array([[c_x1, c_y1], [c_x2, c_y2], [c_x3, c_y3], [c_x4, c_y4]], np.int32)


def enclosing_bbox(bbox: np.ndarray) -> np.ndarray:
    """Get enclosing box(es) for rotated corners.

    Args:
        bbox: Box(es) as ``np.ndarray`` in [..., 8], [x1, y1, x2, y2, x3, y3, x4, y4] format.

    Returns:
        Box(es) as ``np.ndarray`` in [..., 4], XYXY format.

    Raises:
        ValueError: If bbox last dimension is not 8.
    """
    if bbox.shape[-1] < 8:
        raise ValueError(f"[bbox] last dimension must be 8, got {bbox.shape[-1]}.")
    x_ = bbox[:, [0, 2, 4, 6]]
    y_ = bbox[:, [1, 3, 5, 7]]
    x1 = np.min(x_, 1).reshape(-1, 1)
    y1 = np.min(y_, 1).reshape(-1, 1)
    x2 = np.max(x_, 1).reshape(-1, 1)
    y2 = np.max(y_, 1).reshape(-1, 1)
    return np.hstack((x1, y1, x2, y2, bbox[:, 8:]))
