#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements bounding box manipulation and preprocessing functions.

Common Tasks:
    - Format conversions.
    - Transformations.
"""

__all__ = [
    "bbox_center_distance",
    "bbox_ciou",
    "bbox_coco_to_voc",
    "bbox_coco_to_yolo",
    "bbox_cxcywhn_to_xywh",
    "bbox_cxcywhn_to_xyxy",
    "bbox_cxcywhn_to_xyxyn",
    "bbox_diou",
    "bbox_giou",
    "bbox_iou",
    "bbox_voc_to_coco",
    "bbox_voc_to_yolo",
    "bbox_xywh_to_cxcywhn",
    "bbox_xywh_to_xyxy",
    "bbox_xywh_to_xyxyn",
    "bbox_xyxy_to_cxcywhn",
    "bbox_xyxy_to_xywh",
    "bbox_xyxy_to_xywh",
    "bbox_xyxy_to_xyxyn",
    "bbox_xyxyn_to_cxcywhn",
    "bbox_xyxyn_to_xywh",
    "bbox_xyxyn_to_xyxy",
    "bbox_yolo_to_coco",
    "bbox_yolo_to_voc",
    "convert_bbox",
]

import numpy as np

from mon.constants import ShapeCode


# ----- Calculation -----
def bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of bounding boxes.

    Args:
        bbox1: Boxes as ``np.ndarray`` in [4] or [N, 4], XYXY format.
        bbox2: Boxes as ``np.ndarray`` in [4] or [M, 4], XYXY format.

    Returns:
        Pairwise IoU values as ``np.ndarray`` in [N, M].

    Raises:
        ValueError: If ``bbox1`` or ``bbox2`` is not 1D or 2D.
    """
    # Ensure 2D arrays
    if bbox1.ndim == 1:
        bbox1 = np.expand_dims(bbox1, 0)
    if bbox2.ndim == 1:
        bbox2 = np.expand_dims(bbox2, 0)
    if bbox1.ndim != 2:
        raise ValueError(f"[bbox1] must be 1D or 2D, but got {bbox1.ndim}D.")
    if bbox2.ndim != 2:
        raise ValueError(f"[bbox2] must be 1D or 2D, but got {bbox2.ndim}D.")
    
    # Expand the dimensions of the bboxes to calculate pairwise IoU values.
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)
    
    # IoU calculation.
    xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2 = np.minimum(bbox1[..., 3], bbox2[..., 3])
    
    # Intersection area
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    
    # Union area
    union = ((bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
           + (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh)
    iou   = wh / union
    return iou


def bbox_giou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute generalized IoU between two sets of boxes.

    Args:
        bbox1: Boxes as ``np.ndarray`` in [4] or [N, 4], XYXY format.
        bbox2: Boxes as ``np.ndarray`` in [4] or [M, 4], XYXY format.

    Returns:
        Pairwise GIoU values as ``np.ndarray`` in [N, M].

    Raises:
        ValueError: If bbox1 or bbox2 is not 1D or 2D.

    References:
        - https://arxiv.org/pdf/1902.09630.pdf
    """
    # Ensure 2D arrays
    if bbox1.ndim == 1:
        bbox1 = np.expand_dims(bbox1, 0)
    if bbox2.ndim == 1:
        bbox2 = np.expand_dims(bbox2, 0)
    if bbox1.ndim != 2:
        raise ValueError(f"[bbox1] must be 1D or 2D, but got {bbox1.ndim}D.")
    if bbox2.ndim != 2:
        raise ValueError(f"[bbox2] must be 1D or 2D, but got {bbox2.ndim}D.")
    
    # Expand dimensions for pairwise computation
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)

    # Intersection coordinates
    xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2 = np.minimum(bbox1[..., 3], bbox2[..., 3])

    # Intersection area
    w   = np.maximum(0.0, xx2 - xx1)
    h   = np.maximum(0.0, yy2 - yy1)
    wh  = w * h

    # Union area
    union = (
        (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1]) +
        (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh
    )

    # IoU
    iou = wh / union

    # Enclosing box coordinates
    xxc1 = np.minimum(bbox1[..., 0], bbox2[..., 0])
    yyc1 = np.minimum(bbox1[..., 1], bbox2[..., 1])
    xxc2 = np.maximum(bbox1[..., 2], bbox2[..., 2])
    yyc2 = np.maximum(bbox1[..., 3], bbox2[..., 3])

    # Enclosing area
    wc   = xxc2 - xxc1
    hc   = yyc2 - yyc1
    area_enclose = wc * hc

    # GIoU
    giou = iou - (area_enclose - union) / area_enclose
    # giou = (giou + 1.0) / 2.0  # Commented out: GIoU typically in [-1, 1], not [0, 1]
    return giou


def bbox_diou(bbox1: np.ndarray,  bbox2: np.ndarray) -> np.ndarray:
    """Compute distance IoU between two sets of boxes.

    Args:
        bbox1: Boxes as ``np.ndarray`` in [4] or [N, 4], XYXY format.
        bbox2: Boxes as ``np.ndarray`` in [4] or [M, 4], XYXY format.

    Returns:
        Pairwise DIoU values as ``np.ndarray`` in [N, M].

    Raises:
        ValueError: If bbox1 or bbox2 is not 1D or 2D.

    References:
        - https://arxiv.org/pdf/1902.09630.pdf
    """
    # Ensure 2D arrays
    if bbox1.ndim == 1:
        bbox1 = np.expand_dims(bbox1, 0)
    if bbox2.ndim == 1:
        bbox2 = np.expand_dims(bbox2, 0)
    if bbox1.ndim != 2:
        raise ValueError(f"[bbox1] must be 1D or 2D, got {bbox1.ndim}D.")
    if bbox2.ndim != 2:
        raise ValueError(f"[bbox2] must be 1D or 2D, got {bbox2.ndim}D.")

    # Expand dimensions for pairwise computation
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)

    # Intersection coordinates
    xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2 = np.minimum(bbox1[..., 3], bbox2[..., 3])

    # Intersection area
    w   = np.maximum(0.0, xx2 - xx1)
    h   = np.maximum(0.0, yy2 - yy1)
    wh  = w * h

    # Union area
    union = (
        (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1]) +
        (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh
    )

    # IoU
    iou = wh / union

    # Center distances
    centerx1 = (bbox1[..., 0] + bbox1[..., 2]) / 2.0
    centery1 = (bbox1[..., 1] + bbox1[..., 3]) / 2.0
    centerx2 = (bbox2[..., 0] + bbox2[..., 2]) / 2.0
    centery2 = (bbox2[..., 1] + bbox2[..., 3]) / 2.0
    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    # Enclosing box diagonal
    xxc1 = np.minimum(bbox1[..., 0], bbox2[..., 0])
    yyc1 = np.minimum(bbox1[..., 1], bbox2[..., 1])
    xxc2 = np.maximum(bbox1[..., 2], bbox2[..., 2])
    yyc2 = np.maximum(bbox1[..., 3], bbox2[..., 3])
    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2

    # DIoU
    diou = iou - inner_diag / outer_diag
    # diou = (diou + 1) / 2.0  # Commented: DIoU typically in [-1, 1], not [0, 1]
    return diou


def bbox_ciou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute complete IoU between two sets of boxes.

    Args:
        bbox1: Boxes as ``np.ndarray`` in [4] or [N, 4], XYXY format.
        bbox2: Boxes as ``np.ndarray`` in [4] or [M, 4], XYXY format.

    Returns:
        Pairwise CIoU values as ``np.ndarray`` in [N, M].

    Raises:
        ValueError: If bbox1 or bbox2 is not 1D or 2D.

    References:
        - https://arxiv.org/pdf/1902.09630.pdf
    """
    # Ensure 2D arrays
    if bbox1.ndim == 1:
        bbox1 = np.expand_dims(bbox1, 0)
    if bbox2.ndim == 1:
        bbox2 = np.expand_dims(bbox2, 0)
    if bbox1.ndim != 2:
        raise ValueError(f"[bbox1] must be 1D or 2D, got {bbox1.ndim}D.")
    if bbox2.ndim != 2:
        raise ValueError(f"[bbox2] must be 1D or 2D, got {bbox2.ndim}D.")

    # Expand dimensions for pairwise computation
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)

    # Intersection coordinates
    xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2 = np.minimum(bbox1[..., 3], bbox2[..., 3])

    # Intersection area
    w   = np.maximum(0.0, xx2 - xx1)
    h   = np.maximum(0.0, yy2 - yy1)
    wh  = w * h

    # Union area
    union = (
        (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1]) +
        (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh
    )

    # IoU
    iou = wh / union

    # Center distances
    centerx1 = (bbox1[..., 0] + bbox1[..., 2]) / 2.0
    centery1 = (bbox1[..., 1] + bbox1[..., 3]) / 2.0
    centerx2 = (bbox2[..., 0] + bbox2[..., 2]) / 2.0
    centery2 = (bbox2[..., 1] + bbox2[..., 3]) / 2.0
    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    # Enclosing box diagonal
    xxc1 = np.minimum(bbox1[..., 0], bbox2[..., 0])
    yyc1 = np.minimum(bbox1[..., 1], bbox2[..., 1])
    xxc2 = np.maximum(bbox1[..., 2], bbox2[..., 2])
    yyc2 = np.maximum(bbox1[..., 3], bbox2[..., 3])
    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2

    # Aspect ratio term
    w1 = bbox1[..., 2] - bbox1[..., 0]
    h1 = bbox1[..., 3] - bbox1[..., 1]
    w2 = bbox2[..., 2] - bbox2[..., 0]
    h2 = bbox2[..., 3] - bbox2[..., 1]
    h2 += 1.0  # Prevent division by zero
    h1 += 1.0  # Prevent division by zero
    arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v      = (4 / (np.pi ** 2)) * (arctan ** 2)
    S      = 1 - iou
    alpha  = v / (S + v)

    # CIoU
    ciou = iou - inner_diag / outer_diag - alpha * v
    # ciou = (ciou + 1) / 2.0  # Commented: CIoU typically in [-1, 1], not [0, 1]
    return ciou


def bbox_center_distance(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Measure center distance(s) between two sets of boxes.

    Args:
        bbox1: Boxes as ``np.ndarray`` in [4] or [N, 4], XYXY format.
        bbox2: Boxes as ``np.ndarray`` in [4] or [M, 4], XYXY format.

    Returns:
        Pairwise center distances as ``np.ndarray`` in [N, M].

    Raises:
        ValueError: If bbox1 or bbox2 is not 1D or 2D.

    Notes:
        Coarse implementation, not recommended alone for association due to instability.
    """
    # Ensure 2D arrays
    if bbox1.ndim == 1:
        bbox1 = np.expand_dims(bbox1, 0)
    if bbox2.ndim == 1:
        bbox2 = np.expand_dims(bbox2, 0)
    if bbox1.ndim != 2:
        raise ValueError(f"[bbox1] must be 1D or 2D, but got {bbox1.ndim}D.")
    if bbox2.ndim != 2:
        raise ValueError(f"[bbox2] must be 1D or 2D, but got {bbox2.ndim}D.")

    # Expand dimensions for pairwise computation
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)

    # Center coordinates
    centerx1 = (bbox1[..., 0] + bbox1[..., 2]) / 2.0  # Fixed: Use bbox1 only
    centery1 = (bbox1[..., 1] + bbox1[..., 3]) / 2.0  # Fixed: Use bbox1 only
    centerx2 = (bbox2[..., 0] + bbox2[..., 2]) / 2.0  # Fixed: Use bbox2 only
    centery2 = (bbox2[..., 1] + bbox2[..., 3]) / 2.0  # Fixed: Use bbox2 only

    # Squared Euclidean distance
    ct_dist2 = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    # Euclidean distance
    ct_dist = np.sqrt(ct_dist2)

    # Normalize and invert to [0, 1] (smaller distance = higher value)
    ct_dist_max = np.max(ct_dist)
    if ct_dist_max > 0:  # Avoid division by zero
        ct_dist = ct_dist / ct_dist_max
        ct_dist = ct_dist_max - ct_dist  # Invert: max distance = 0, min = max
    else:
        ct_dist = np.ones_like(ct_dist)  # All distances 0 -> all 1

    return ct_dist


# ----- Convert -----
def bbox_cxcywhn_to_xywh(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from CXCYWHN to XYWH format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], CXCYWHN format, normalized.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4], XYWH format, pixel coordinates.
    """
    cx_norm, cy_norm, w_norm, h_norm, *_ = bbox.T
    w = w_norm * width
    h = h_norm * height
    x = (cx_norm * width)  - (w / 2.0)
    y = (cy_norm * height) - (h / 2.0)
    return np.stack((x, y, w, h), axis=-1)


def bbox_cxcywhn_to_xyxy(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from CXCYWHN to XYXY format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], CXCYWHN format, normalized.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4], XYXY format, pixel coordinates.
    """
    cx_norm, cy_norm, w_norm, h_norm, *_ = bbox.T
    x1 = width  * (cx_norm - w_norm / 2)
    y1 = height * (cy_norm - h_norm / 2)
    x2 = width  * (cx_norm + w_norm / 2)
    y2 = height * (cy_norm + h_norm / 2)
    return np.stack((x1, y1, x2, y2), axis=-1)


def bbox_cxcywhn_to_xyxyn(bbox: np.ndarray) -> np.ndarray:
    """Convert boxes from CXCYWHN to XYXYN format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], CXCYWHN format, normalized.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4], XYXYN format, normalized.
    """
    cx_norm, cy_norm, w_norm, h_norm, *_ = bbox.T
    x1 = (cx_norm - w_norm / 2)
    y1 = (cy_norm - h_norm / 2)
    x2 = (cx_norm + w_norm / 2)
    y2 = (cy_norm + h_norm / 2)
    return np.stack((x1, y1, x2, y2), axis=-1)


def bbox_xywh_to_cxcywhn(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYWH to CXCYWHN format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYWH format, pixel coordinates.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4], CXCYWHN format, normalized.
    """
    x, y, w, h, *_ = bbox.T
    cx      = x + (w / 2.0)
    cy      = y + (h / 2.0)
    cx_norm = cx / width
    cy_norm = cy / height
    w_norm  = w / width
    h_norm  = h / height
    return np.stack((cx_norm, cy_norm, w_norm, h_norm), axis=-1)


def bbox_xywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """Convert boxes from XYWH to XYXY format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYWH format, pixel coordinates.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4], XYXY format, pixel coordinates.
    """
    x, y, w, h, *_ = bbox.T
    x2 = x + w
    y2 = y + h
    return np.stack((x, y, x2, y2), axis=-1)


def bbox_xywh_to_xyxyn(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYWH to XYXYN format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYWH format, pixel coordinates.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4], XYXYN format, normalized.
    """
    x, y, w, h, *_ = bbox.T
    x2      = x + w
    y2      = y + h
    x1_norm = x / width
    y1_norm = y / height
    x2_norm = x2 / width
    y2_norm = y2 / height
    return np.stack((x1_norm, y1_norm, x2_norm, y2_norm), axis=-1)


def bbox_xyxy_to_cxcywhn(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYXY to CXCYWHN format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYXY format, pixel coordinates.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4], CXCYWHN format, normalized.
    """
    x1, y1, x2, y2, *_ = bbox.T
    w       = x2 - x1
    h       = y2 - y1
    cx      = x1 + (w / 2.0)
    cy      = y1 + (h / 2.0)
    cx_norm = cx / width
    cy_norm = cy / height
    w_norm  = w / width
    h_norm  = h / height
    return np.stack((cx_norm, cy_norm, w_norm, h_norm), axis=-1)


def bbox_xyxy_to_xywh(bbox: np.ndarray) -> np.ndarray:
    """Convert boxes from XYXY to XYWH format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYXY format, pixel coordinates.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4], XYWH format, pixel coordinates.
    """
    x1, y1, x2, y2, *_ = bbox.T
    w = x2 - x1
    h = y2 - y1
    return np.stack((x1, y1, w, h), axis=-1)


def bbox_xyxy_to_xyxyn(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYXY to XYXYN format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYXY format, pixel coordinates.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4], XYXYN format, normalized
    """
    x1, y1, x2, y2, *_ = bbox.T
    x1_norm = x1 / width
    y1_norm = y1 / height
    x2_norm = x2 / width
    y2_norm = y2 / height
    return np.stack((x1_norm, y1_norm, x2_norm, y2_norm), axis=-1)


def bbox_xyxyn_to_cxcywhn(bbox: np.ndarray) -> np.ndarray:
    """Convert boxes from XYXYN to CXCYWHN format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYXYN format, normalized.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4], CXCYWHN format, normalized.
    """
    x1, y1, x2, y2, *_ = bbox.T
    w_norm  = x2 - x1
    h_norm  = y2 - y1
    cx_norm = x1 + (w_norm / 2.0)
    cy_norm = y1 + (h_norm / 2.0)
    return np.stack((cx_norm, cy_norm, w_norm, h_norm), axis=-1)


def bbox_xyxyn_to_xywh(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYXYN to XYWH format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYXYN format, normalized.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4], XYWH format, pixel coordinates.
    """
    x1, y1, x2, y2, *_ = bbox.T
    x1 = x1 * width
    x2 = x2 * width
    y1 = y1 * height
    y2 = y2 * height
    w  = x2 - x1
    h  = y2 - y1
    return np.stack((x1, y1, w, h), axis=-1)


def bbox_xyxyn_to_xyxy(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYXYN to XYXY format.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], XYXYN format, normalized.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4], XYXY format, pixel coordinates.
    """
    x1, y1, x2, y2, *_ = bbox.T
    x1 = x1 * width
    x2 = x2 * width
    y1 = y1 * height
    y2 = y2 * height
    return np.stack((x1, y1, x2, y2), axis=-1)


bbox_coco_to_voc  = bbox_xywh_to_xyxy
bbox_coco_to_yolo = bbox_xywh_to_cxcywhn
bbox_voc_to_coco  = bbox_xyxy_to_xywh
bbox_voc_to_yolo  = bbox_xyxy_to_cxcywhn
bbox_yolo_to_coco = bbox_cxcywhn_to_xywh
bbox_yolo_to_voc  = bbox_cxcywhn_to_xyxy


def convert_bbox(bbox: np.ndarray, code: ShapeCode | int, height: int, width: int) -> np.ndarray:
    """Convert bounding box between formats.

    Args:
        bbox: Boxes as ``np.ndarray`` in [N, 4+], input format varies by code.
        code: Conversion code as ``ShapeCode`` or ``int``.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4], output format varies by code.

    Raises:
        ValueError: If ``code`` is invalid.
    """
    code = ShapeCode.from_value(value=code)
    match code:
        case ShapeCode.SAME:
            return bbox
        case ShapeCode.VOC2COCO | ShapeCode.XYXY2XYWH:
            return bbox_voc_to_coco(bbox)
        case ShapeCode.VOC2YOLO | ShapeCode.XYXY2CXCYN:
            return bbox_voc_to_yolo(bbox, height, width)
        case ShapeCode.COCO2VOC | ShapeCode.XYWH2XYXY:
            return bbox_coco_to_voc(bbox)
        case ShapeCode.COCO2YOLO | ShapeCode.XYWH2CXCYN:
            return bbox_coco_to_yolo(bbox, height, width)
        case ShapeCode.YOLO2VOC | ShapeCode.CXCYN2XYXY:
            return bbox_yolo_to_voc(bbox, height, width)
        case ShapeCode.YOLO2COCO | ShapeCode.CXCYN2XYXY:
            return bbox_yolo_to_coco(bbox, height, width)
        case _:
            raise ValueError(f"[code] invalid: {code}.")
