#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements general-purpose utilities for bounding box.

Common Tasks:
    - Property accessors
    - Validation checks
    - Miscellaneous
"""

__all__ = [
    "check_valid_bbox",
    "is_bbox_coco",
    "is_bbox_cxcywhn",
    "is_bbox_normalized",
    "is_bbox_voc",
    "is_bbox_xywh",
    "is_bbox_xyxy",
    "is_bbox_yolo",
]

import numpy as np
import torch


# ----- Accessing -----


# ----- Validation -----
def check_valid_bbox(bbox: torch.Tensor | np.ndarray | list | tuple) -> torch.Tensor | np.ndarray:
    """Check if a bounding box is valid.

    Args:
        bbox: Box(es) as ``np.ndarray`` or ``torch.Tensor`` in [4] or [N, 4].

    Returns:
        ``True`` if valid, ``False`` otherwise.
    """
    if isinstance(bbox, list | tuple):
        bbox = np.array(bbox, dtype=np.float32)
    if not isinstance(bbox, torch.Tensor | np.ndarray):
        raise ValueError(f"[bbox] must be a torch.Tensor or numpy.ndarray, got {type(bbox)}.")
    if bbox.ndim == 1:
        if isinstance(bbox, np.ndarray):
            bbox = bbox.reshape(1, -1)
        else:  # torch.Tensor
            bbox = bbox.unsqueeze(0)
    if bbox.ndim != 2 or bbox.shape[1] < 4:
        raise ValueError(f"[bbox] must be a 2D array [N, M] with M >= 4 or a 1D array [M] with M >= 4, got {bbox.shape}.")
    return bbox


def is_bbox_normalized(bbox: np.ndarray) -> bool:
    """Check if a bounding box is normalized to range [0.0, 1.0].

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4] or [N, 4].

    Returns:
        ``True`` if normalized, ``False`` otherwise.
    """
    bbox = check_valid_bbox(bbox)
    return np.all((bbox[:, :4] >= 0) & (bbox[:, :4] <= 1))


def is_bbox_cxcywhn(bbox: np.ndarray, height: int, width: int) -> bool:
    """Check if a bounding box is in CXCYWHN format.

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4] or [N, 4].
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        ``True`` if in CXCYWHN format, ``False`` otherwise.
    """
    bbox = check_valid_bbox(bbox)
    return (
        np.all((bbox[:, :4] >= 0) & (bbox[:, :4] <= 1))
        and np.all((bbox[:, 2:4] > 0))  # Width and height must be positive
    )


def is_bbox_xyxy(bbox: np.ndarray, height: int, width: int) -> bool:
    """Check if a bounding box is in XYXY format.

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4] or [N, 4].
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        ``True`` if in XYXY format, ``False`` otherwise.
    """
    bbox = check_valid_bbox(bbox)
    if is_bbox_cxcywhn(bbox, height, width):
        return False

    # Extract first bbox for format checking
    x, y, w, h = bbox[0, :4]
    if w > x and h > y:  # VOC: x_max > x_min, y_max > y_min
        return True
    else:
        return False


def is_bbox_xywh(bbox: np.ndarray, height: int, width: int) -> bool:
    """Check if a bounding box is in XYWH format.

    Args:
        bbox: Box(es) as ``np.ndarray`` in [4] or [N, 4].
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        ``True`` if in XYWH format, ``False`` otherwise.
    """
    bbox = check_valid_bbox(bbox)
    if is_bbox_cxcywhn(bbox, height, width):
        return False

    # Extract first bbox for format checking
    x, y, w, h = bbox[0, :4]
    if w + x > x and h + y > y:  # VOC: w=x_max, h=y_max, so x_min+w > x_min
        return True  # COCO: w=width, h=height
    else:
        return False


is_bbox_coco = is_bbox_xywh
is_bbox_voc  = is_bbox_xyxy
is_bbox_yolo = is_bbox_cxcywhn
