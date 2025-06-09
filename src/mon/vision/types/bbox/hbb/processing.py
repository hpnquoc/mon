#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements HBBs manipulation and preprocessing functions.

Common Tasks:
    - Format conversions.
    - Transformations.
"""

__all__ = [
    "hbb_area",
    "hbb_center",
    "hbb_center_distance",
    "hbb_ciou",
    "hbb_coco_to_voc",
    "hbb_coco_to_yolo",
    "hbb_corners",
    "hbb_corners_pts",
    "hbb_cxcywhn_to_xywh",
    "hbb_cxcywhn_to_xyxy",
    "hbb_diou",
    "hbb_giou",
    "hbb_iou",
    "hbb_to_2d",
    "hbb_to_3d",
    "hbb_to_array",
    "hbb_to_tensor",
    "hbb_voc_to_coco",
    "hbb_voc_to_yolo",
    "hbb_xywh_to_cxcywhn",
    "hbb_xywh_to_xyxy",
    "hbb_xyxy_to_cxcywhn",
    "hbb_xyxy_to_xywh",
    "hbb_xyxy_to_xywh",
    "hbb_yolo_to_coco",
    "hbb_yolo_to_voc",
    "convert_hbb",
    "denormalize_hbb",
    "enclosing_hbb",
    "normalize_hbb",
    "split_image_and_hbbs",
]

import math

import numpy as np
import torch

from mon.constants import BBoxFormat
from mon.vision.types import image as I
from mon.vision.types.bbox.hbb import utils


# ----- Splitting -----
def split_image_and_hbbs(
    image : np.ndarray,
    bbox  : np.ndarray,
    n     : int = 2
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split an image into ``n`` equal parts and adjust YOLO-format hbbs accordingly.

    Args:
        image: Image as ``numpy.ndarray`` [H, W, C].
        bbox: HBBs as ``np.ndarray`` in [N, 4+], CXCYWHN format (YOLO), normalized.
        n: Number of parts to split into (positive integer). Default is ``2``.

    Raises:
        ValueError: If inputs are invalid (e.g., image shape, bboxes, n).
    """
    if not isinstance(image, np.ndarray) or len(image.shape) != 3:
        raise ValueError(f"[image] must be a 3D numpy array [H, W, C], got {image.shape}.")
    if not isinstance(bbox, np.ndarray)  or bbox.ndim != 2 or bbox.shape[1] < 4:
        raise ValueError(f"[bboxes] must be a 2D numpy array [N, M] with M >= 4, got {bbox.shape}.")
    if n < 1:
        raise ValueError(f"[n] must be a positive integer, got {n}.")

    h, w = I.image_size(image)
    if n > h * w:
        raise ValueError(f"[n] ({n}) exceeds image pixel count ({h * w}).")

    # Determine orientation
    is_portrait = h > w

    # Determine rows and cols
    if n == 1:
        rows, cols = 1, 1
    elif n == 2:
        # Explicitly set grid for N=2 based on orientation
        rows = 2 if is_portrait else 1
        cols = 1 if is_portrait else 2
    else:
        # General case: start with approximate square grid
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)
        # Adjust to ensure rows * cols = n, prioritizing orientation
        candidates = []
        for r in range(1, n + 1):
            c = math.ceil(n / r)
            if r * c == n:
                candidates.append((r, c))
        if not candidates:
            raise ValueError(f"Cannot find valid rows and cols for n={n}")
        # Select grid based on orientation
        if is_portrait:
            # Prefer more rows (taller sub-images)
            rows, cols = max(candidates, key=lambda x: x[0] / x[1])
        else:
            # Prefer more cols (wider sub-images)
            rows, cols = max(candidates, key=lambda x: x[1] / x[0])

    # Compute sub-images and adjust bboxes
    sub_h      = h // rows
    sub_w      = w // cols
    sub_images = []
    sub_bboxes = []

    for i in range(rows):
        for j in range(cols):
            if len(sub_images) >= n:
                break
            # Compute sub-image boundaries
            y_start   = i * sub_h
            y_end     = min((i + 1) * sub_h, h)
            x_start   = j * sub_w
            x_end     = min((j + 1) * sub_w, w)
            sub_image = image[y_start:y_end, x_start:x_end]
            if sub_image.size == 0:
                continue
            sub_images.append(sub_image)

            # Adjust bboxes
            sub_bboxes_i     = []
            sub_h_i, sub_w_i = sub_image.shape[:2]
            for bbox in bbox:
                cx_n, cy_n, w_n, h_n = bbox[:4]
                cx = cx_n * w
                cy = cy_n * h
                x1 = cx - w_n * w / 2
                x2 = cx + w_n * w / 2
                y1 = cy - h_n * h / 2
                y2 = cy + h_n * h / 2

                # Check if bbox intersects sub-image
                if x2 > x_start and x1 < x_end and y2 > y_start and y1 < y_end:
                    # Compute new bbox in sub-image coordinates
                    x1_new   = max(x1, x_start) - x_start
                    x2_new   = min(x2, x_end)   - x_start
                    y1_new   = max(y1, y_start) - y_start
                    y2_new   = min(y2, y_end)   - y_start
                    cx_n_new = (x1_new + x2_new) / 2 / sub_w_i
                    cy_n_new = (y1_new + y2_new) / 2 / sub_h_i
                    w_n_new  = (x2_new - x1_new) / sub_w_i
                    h_n_new  = (y2_new - y1_new) / sub_h_i
                    if w_n_new > 0 and h_n_new > 0:
                        bbox_new = np.concatenate(([cx_n_new, cy_n_new, w_n_new, h_n_new], bbox[4:]))
                        sub_bboxes_i.append(bbox_new)
            sub_bboxes.append(np.array(sub_bboxes_i) if sub_bboxes_i else np.zeros((0, bbox.shape[1]), dtype=np.float32))

    # Pad with empty sub-images/bboxes if needed
    while len(sub_images) < n:
        sub_images.append(np.zeros_like(sub_images[0]))
        sub_bboxes.append(np.zeros((0, bbox.shape[1]), dtype=np.float32))

    return sub_images, sub_bboxes


# ----- IoU Calculation -----
def hbb_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of HBBs.

    Args:
        bbox1: HBBs as ``np.ndarray`` in [4+] or [N, 4+], XYXY format.
        bbox2: HBBs as ``np.ndarray`` in [4+] or [M, 4+], XYXY format.

    Returns:
        Pairwise IoU values as ``np.ndarray`` in [N, M].

    Raises:
        ValueError: If ``bbox1`` or ``bbox2`` is not 1D or 2D.
    """
    # Ensure 2D arrays
    bbox1 = hbb_to_2d(bbox1)
    bbox2 = hbb_to_2d(bbox2)

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


def hbb_giou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute generalized IoU between two sets of boxes.

    Args:
        bbox1: HBBs as ``np.ndarray`` in [4+] or [N, 4+], XYXY format.
        bbox2: HBBs as ``np.ndarray`` in [4+] or [M, 4+], XYXY format.

    Returns:
        Pairwise GIoU values as ``np.ndarray`` in [N, M].

    Raises:
        ValueError: If bbox1 or bbox2 is not 1D or 2D.

    References:
        - https://arxiv.org/pdf/1902.09630.pdf
    """
    # Ensure 2D arrays
    bbox1 = hbb_to_2d(bbox1)
    bbox2 = hbb_to_2d(bbox2)
    
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


def hbb_diou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute distance IoU between two sets of boxes.

    Args:
        bbox1: HBBs as ``np.ndarray`` in [4+] or [N, 4+], XYXY format.
        bbox2: HBBs as ``np.ndarray`` in [4+] or [M, 4+], XYXY format.

    Returns:
        Pairwise DIoU values as ``np.ndarray`` in [N, M].

    Raises:
        ValueError: If bbox1 or bbox2 is not 1D or 2D.

    References:
        - https://arxiv.org/pdf/1902.09630.pdf
    """
    # Ensure 2D arrays
    bbox1 = hbb_to_2d(bbox1)
    bbox2 = hbb_to_2d(bbox2)

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
    cx1 = (bbox1[..., 0] + bbox1[..., 2]) / 2.0
    cy1 = (bbox1[..., 1] + bbox1[..., 3]) / 2.0
    cx2 = (bbox2[..., 0] + bbox2[..., 2]) / 2.0
    cy2 = (bbox2[..., 1] + bbox2[..., 3]) / 2.0
    inner_diag = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

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


def hbb_ciou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute complete IoU between two sets of boxes.

    Args:
        bbox1: HBBs as ``np.ndarray`` in [4+] or [N, 4+], XYXY format.
        bbox2: HBBs as ``np.ndarray`` in [4+] or [M, 4+], XYXY format.

    Returns:
        Pairwise CIoU values as ``np.ndarray`` in [N, M].

    Raises:
        ValueError: If bbox1 or bbox2 is not 1D or 2D.

    References:
        - https://arxiv.org/pdf/1902.09630.pdf
    """
    # Ensure 2D arrays
    bbox1 = hbb_to_2d(bbox1)
    bbox2 = hbb_to_2d(bbox2)

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
    cx1 = (bbox1[..., 0] + bbox1[..., 2]) / 2.0
    cy1 = (bbox1[..., 1] + bbox1[..., 3]) / 2.0
    cx2 = (bbox2[..., 0] + bbox2[..., 2]) / 2.0
    cy2 = (bbox2[..., 1] + bbox2[..., 3]) / 2.0
    inner_diag = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

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


# ----- Properties Calculation -----
def hbb_center_distance(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Measure center distance(s) between two sets of boxes.

    Args:
        bbox1: HBBs as ``np.ndarray`` in [4+] or [N, 4+], XYXY format.
        bbox2: HBBs as ``np.ndarray`` in [4+] or [M, 4+], XYXY format.

    Returns:
        Pairwise center distances as ``np.ndarray`` in [N, M].

    Raises:
        ValueError: If bbox1 or bbox2 is not 1D or 2D.

    Notes:
        Coarse implementation, not recommended alone for association due to instability.
    """
    # Ensure 2D arrays
    bbox1 = hbb_to_2d(bbox1)
    bbox2 = hbb_to_2d(bbox2)

    # Expand dimensions for pairwise computation
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)

    # Center coordinates
    cx1 = (bbox1[..., 0] + bbox1[..., 2]) / 2.0  # Fixed: Use bbox1 only
    cy1 = (bbox1[..., 1] + bbox1[..., 3]) / 2.0  # Fixed: Use bbox1 only
    cx2 = (bbox2[..., 0] + bbox2[..., 2]) / 2.0  # Fixed: Use bbox2 only
    cy2 = (bbox2[..., 1] + bbox2[..., 3]) / 2.0  # Fixed: Use bbox2 only

    # Squared Euclidean distance
    ct_dist2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

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


def hbb_area(bbox: np.ndarray) -> np.ndarray:
    """Compute area of HBBs.

    Args:
        bbox: HBBs as ``np.ndarray`` in [4+] or [N, 4+], XYXY format.

    Returns:
        Area(s) as ``np.ndarray`` in [1] or [N] shape.

    Raises:
        ValueError: If ``bbox`` is not 1D or 2D.
    """
    bbox = hbb_to_2d(bbox)
    x1   = bbox[..., 0]
    y1   = bbox[..., 1]
    x2   = bbox[..., 2]
    y2   = bbox[..., 3]
    return (x2 - x1) * (y2 - y1)


def hbb_center(bbox: np.ndarray) -> np.ndarray:
    """Compute center(s) of HBBs.

    Args:
        bbox: HBBs as ``np.ndarray`` in [4+] or [N, 4+], XYXY format.

    Returns:
        Center(s) as ``np.ndarray`` in [1, 2] or [N, 2], [cx, cy] format.

    Raises:
        ValueError: If bbox is not 1D or 2D.
    """
    bbox = hbb_to_2d(bbox)
    x1   = bbox[..., 0]
    y1   = bbox[..., 1]
    x2   = bbox[..., 2]
    y2   = bbox[..., 3]
    cx   = x1 + (x2 - x1) / 2.0
    cy   = y1 + (y2 - y1) / 2.0
    return np.stack((cx, cy), -1)


def hbb_corners(bbox: np.ndarray) -> np.ndarray:
    """Get corner(s) of HBBs.

    Args:
        bbox: HBBs as ``np.ndarray`` in [4+] or [N, 4+], XYXY format

    Returns:
        Corners as ``np.ndarray`` in [N, 8], [x1, y1, x2, y2, x3, y3, x4, y4] format

    Raises:
        ValueError: If ``bbox`` is not 1D or 2D.
    """
    bbox = hbb_to_2d(bbox)
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


def hbb_corners_pts(bbox: np.ndarray) -> np.ndarray:
    """Get corner(s) of HBBs as points.

    Args:
        bbox: HBBs as ``np.ndarray`` in [4+] or [N, 4+], XYXY format.

    Returns:
        Corners as ``np.ndarray`` in
        [N, 4, 2], [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] format.

    Raises:
        ValueError: If ``bbox`` is not 1D or 2D.
    """
    bbox = hbb_to_2d(bbox)
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


def enclosing_hbb(bbox: np.ndarray) -> np.ndarray:
    """Get enclosing box(es) for rotated corners.

    Args:
        bbox: HBBs as ``np.ndarray`` in [..., 8], [x1, y1, x2, y2, x3, y3, x4, y4] format.

    Returns:
        HBBs as ``np.ndarray`` in [..., 4], XYXY format.

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


# ----- Normalization -----
def normalize_hbb(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Normalize HBBs according to image dimensions.

    Args:
        bbox: HBBs as ``np.ndarray`` in [4+] or [N, 4+].
        height: Image height.
        width: Image width.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"[height] and [width] must be positive integers, got {height}, {width}.")

    bbox = hbb_to_2d(bbox)
    if utils.is_hbb_normalized(bbox):
        return bbox

    b1, b2, b3, b4, *rest = bbox.T
    b1 = b1 / width
    b2 = b2 / height
    b3 = b3 / width
    b4 = b4 / height
    return np.stack((b1, b2, b3, b4, *rest), axis=-1)


def denormalize_hbb(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Denormalize HBBs according to image dimensions.

    Args:
        bbox: HBBs as ``np.ndarray`` in [4+] or [N, 4+].
        height: Image height.
        width: Image width.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"[height] and [width] must be positive integers, got {height}, {width}.")

    bbox = hbb_to_2d(bbox)
    if not utils.is_hbb_normalized(bbox):
        return bbox

    b1, b2, b3, b4, *rest = bbox.T
    b1 = b1 * width
    b2 = b2 * height
    b3 = b3 * width
    b4 = b4 * height
    return np.stack((b1, b2, b3, b4, *rest), axis=-1)


# ----- Conversion -----
def hbb_to_2d(bbox: torch.Tensor | np.ndarray | list | tuple) -> torch.Tensor | np.ndarray:
    """Convert a 1D, 2D, or 3D box(es) to 2D.

    Args:
        bbox: HBBs as ``np.ndarray``, ``torch.Tensor``, or list/tuple of [4+] or [N, 4+].

    Returns:
        HBBs as ``np.ndarray`` or ``torch.Tensor`` in [N, 4+] format.

    Raises:
        TypeError: If ``bbox`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if isinstance(bbox, torch.Tensor):
        if bbox.ndim == 1:                                                      # [4+]
            bbox = bbox.unsqueeze(0)                                            # [4+]       -> [1, 4+]
        elif bbox.ndim == 3:                                                    # [B, N, 4+]
            if bbox.shape[0] == 1:                                              # [1, N, 4+]
                bbox = bbox.squeeze(0)                                          # [1, N, 4+] -> [N, 4+]
    elif isinstance(bbox, np.ndarray):
        if bbox.ndim == 1:                                                      # [4+]
            bbox = np.expand_dims(bbox, axis=0)                                 # [4+]       -> [1, 4+]
        elif bbox.ndim == 3:                                                    # [B, N, 4+]
            if bbox.shape[0] == 1:                                              # [1, N, 4+]
                bbox = np.squeeze(bbox, axis=0)                                 # [1, N, 4+] -> [N, 4+]
    elif isinstance(bbox, list | tuple):
        if all(isinstance(b, list | tuple | int | float) for b in bbox):
            bbox = np.array(bbox, dtype=np.float32)

        if all(isinstance(b, torch.Tensor)   and b.ndim == 1 for b in bbox):    # [[4+], ...]
            bbox = torch.stack(bbox, dim=0)                                     # [[4+], ...]    -> [N, 4+]
        elif all(isinstance(b, torch.Tensor) and b.ndim == 2 for b in bbox):    # [[N, 4+], ...]
            bbox = torch.cat(bbox, dim=0)                                       # [[N, 4+], ...] -> [N*, 4+]
        elif all(isinstance(b, np.ndarray)   and b.ndim == 1 for b in bbox):    # [[4+], ...]
            bbox = np.stack(bbox, axis=0)                                       # [[4+], ...]    -> [N, 4+]
        elif all(isinstance(b, np.ndarray)   and b.ndim == 2 for b in bbox):    # [[N, 4+], ...]
            bbox = np.concatenate(bbox, axis=0)                                 # [[N, 4+], ...] -> [N*, 4+]
        else:
            raise TypeError(f"[image] list/tuple must contain consistent 1D or 2D "
                            f"torch.Tensor or numpy.ndarray, got mixed types or dimensions: "
                            f"{[type(b) for b in bbox]} "
                            f"{[b.shape for b in bbox if b is not None]}.")
    else:
        raise ValueError(f"[bbox] must be a torch.Tensor, numpy.ndarray, "
                         f"or list/tuple, got {type(bbox)}.")

    return bbox


def hbb_to_3d(bbox: torch.Tensor | np.ndarray | list | tuple) -> torch.Tensor | np.ndarray:
    """Convert a 1D, 2D, or 3D box(es) to 3D.

    Args:
        bbox: HBBs as ``np.ndarray``, ``torch.Tensor``, or list/tuple of
            [4+], [N, 4+], or [B, N, 4+].

    Returns:
        HBBs as ``np.ndarray`` or ``torch.Tensor`` in [B, N, 4+] format.

    Raises:
        TypeError: If ``bbox`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if isinstance(bbox, torch.Tensor):
        if bbox.ndim == 1:                                                      # [4+]
            bbox = bbox.unsqueeze(0).unsqueeze(0)                               # [4+]    -> [1, 1, 4+]
        elif bbox.ndim == 2:                                                    # [N, 4+]
            bbox = bbox.unsqueeze(0)                                            # [N, 4+] -> [1, N, 4+]
    elif isinstance(bbox, np.ndarray):
        if bbox.ndim == 1:                                                      # [4+]
            bbox = np.expand_dims(bbox, axis=0)                                 # [4+]    -> [1, 4+]
            bbox = np.expand_dims(bbox, axis=0)                                 # [1, 4+] -> [1, 1, 4+]
        elif bbox.ndim == 2:                                                    # [N, 4+]
            bbox = np.squeeze(bbox, axis=0)                                     # [N, 4+] -> [1, N, 4+]
    elif isinstance(bbox, list | tuple):
        if all(isinstance(b, list | tuple | int | float) for b in bbox):
            bbox = np.array(bbox, dtype=np.float32)

        if all(isinstance(b, torch.Tensor)   and b.ndim == 1 for b in bbox):    # [[4+], ...]
            bbox = torch.stack(bbox, dim=0)                                     # [[4+], ...]       -> [N, 4+]
            bbox = bbox.unsqueeze(0)                                            # [N, 4+]           -> [1, N, 4+]
        elif all(isinstance(b, torch.Tensor) and b.ndim == 2 for b in bbox):    # [[N, 4+], ...]
            bbox = torch.stack(bbox, dim=0)                                     # [[N, 4+], ...]    -> [B, N, 4+]
        elif all(isinstance(b, torch.Tensor) and b.ndim == 3 for b in bbox):    # [[B, N, 4+], ...]
            bbox = torch.cat(bbox, dim=0)                                       # [[B, N, 4+], ...] -> [B*, N, 4+]
        elif all(isinstance(b, np.ndarray)   and b.ndim == 1 for b in bbox):    # [[4+], ...]
            bbox = np.stack(bbox, axis=0)                                       # [[4+], ...]       -> [N, 4+]
            bbox = np.expand_dims(bbox, axis=0)                                 # [N, 4+]           -> [1, N, 4+]
        elif all(isinstance(b, np.ndarray)   and b.ndim == 2 for b in bbox):    # [[N, 4+], ...]
            bbox = np.stack(bbox, axis=0)                                       # [[N, 4+], ...]    -> [B, N, 4+]
        elif all(isinstance(b, np.ndarray)   and b.ndim == 3 for b in bbox):    # [[B, N, 4+], ...]
            bbox = np.concatenate(bbox, axis=0)                                 # [[B, N, 4+], ...] -> [B*, N, 4+]
        else:
            raise TypeError(f"[image] list/tuple must contain consistent 1D or 2D "
                            f"torch.Tensor or numpy.ndarray, got mixed types or dimensions: "
                            f"{[type(b) for b in bbox]} "
                            f"{[b.shape for b in bbox if b is not None]}.")
    else:
        raise ValueError(f"[bbox] must be a torch.Tensor, numpy.ndarray, "
                         f"or list/tuple, got {type(bbox)}.")

    return bbox


def hbb_xywh_to_cxcywhn(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYWH to CXCYWHN format.

    Args:
        bbox: HBBs as ``np.ndarray`` in [N, 4+], XYWH format, pixel coordinates.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        HBBs as ``np.ndarray`` in [N, 4+], CXCYWHN format, normalized.
    """
    bbox = hbb_to_2d(bbox)
    x, y, w, h, *rest = bbox.T
    cx   = x + (w / 2.0)
    cy   = y + (h / 2.0)
    cx_n = cx / width
    cy_n = cy / height
    w_n  = w  / width
    h_n  = h  / height
    return np.stack((cx_n, cy_n, w_n, h_n, *rest), axis=-1)


def hbb_xywh_to_xyxy(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYWH to XYXY format.

    Args:
        bbox: HBBs as ``np.ndarray`` in [N, 4+], XYWH format, pixel coordinates.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        HBBs as ``np.ndarray`` in [N, 4+], XYXY format, pixel coordinates.
    """
    bbox = hbb_to_2d(bbox)
    x, y, w, h, *rest = bbox.T
    x2 = x + w
    y2 = y + h
    return np.stack((x, y, x2, y2, *rest), axis=-1)


def hbb_xyxy_to_cxcywhn(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYXY to CXCYWHN format.

    Args:
        bbox: HBBs as ``np.ndarray`` in [N, 4+], XYXY format, pixel coordinates.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        HBBs as ``np.ndarray`` in [N, 4+], CXCYWHN format, normalized.
    """
    bbox = hbb_to_2d(bbox)
    x1, y1, x2, y2, *rest = bbox.T
    w    = x2 - x1
    h    = y2 - y1
    cx   = x1 + (w / 2.0)
    cy   = y1 + (h / 2.0)
    cx_n = cx / width
    cy_n = cy / height
    w_n  = w  / width
    h_n  = h  / height
    return np.stack((cx_n, cy_n, w_n, h_n, *rest), axis=-1)


def hbb_xyxy_to_xywh(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from XYXY to XYWH format.

    Args:
        bbox: HBBs as ``np.ndarray`` in [N, 4+], XYXY format, pixel coordinates.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        HBBs as ``np.ndarray`` in [N, 4], XYWH format, pixel coordinates.
    """
    bbox = hbb_to_2d(bbox)
    x1, y1, x2, y2, *rest = bbox.T
    w = x2 - x1
    h = y2 - y1
    return np.stack((x1, y1, w, h, *rest), axis=-1)


def hbb_cxcywhn_to_xywh(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from CXCYWHN to XYWH format.

    Args:
        bbox: HBBs as ``np.ndarray`` in [N, 4+], CXCYWHN format, normalized.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        HBBs as ``np.ndarray`` in [N, 4+], XYWH format, pixel coordinates.
    """
    bbox = hbb_to_2d(bbox)
    cx_n, cy_n, w_n, h_n, *rest = bbox.T
    w = w_n * width
    h = h_n * height
    x = (cx_n * width)  - (w / 2.0)
    y = (cy_n * height) - (h / 2.0)
    # Combine processed columns with rest
    return np.stack((x, y, w, h, *rest), axis=-1)


def hbb_cxcywhn_to_xyxy(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert boxes from CXCYWHN to XYXY format.

    Args:
        bbox: HBBs as ``np.ndarray`` in [N, 4+], CXCYWHN format, normalized.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        HBBs as ``np.ndarray`` in [N, 4+], XYXY format, pixel coordinates.
    """
    bbox = hbb_to_2d(bbox)
    cx_n, cy_n, w_n, h_n, *rest = bbox.T
    x1 = width  * (cx_n - w_n / 2)
    y1 = height * (cy_n - h_n / 2)
    x2 = width  * (cx_n + w_n / 2)
    y2 = height * (cy_n + h_n / 2)
    return np.stack((x1, y1, x2, y2, *rest), axis=-1)


hbb_coco_to_voc  = hbb_xywh_to_xyxy
hbb_coco_to_yolo = hbb_xywh_to_cxcywhn
hbb_voc_to_coco  = hbb_xyxy_to_xywh
hbb_voc_to_yolo  = hbb_xyxy_to_cxcywhn
hbb_yolo_to_coco = hbb_cxcywhn_to_xywh
hbb_yolo_to_voc  = hbb_cxcywhn_to_xyxy


def convert_hbb(bbox: np.ndarray, fmt: BBoxFormat, height: int, width: int) -> np.ndarray:
    """Convert HBBs between formats.

    Args:
        bbox: HBBs as ``np.ndarray`` in [N, 4+], input format varies by code.
        fmt: Conversion code as ``BBoxFormat`` or ``int``.
        height: Image height in pixels as ``int``.
        width: Image width in pixels as ``int``.

    Returns:
        HBBs as ``np.ndarray`` in [N, 4+], output format varies by code.

    Raises:
        ValueError: If ``code`` is invalid.
    """
    if len(bbox) == 0:
        return bbox

    fmt = BBoxFormat.from_value(value=fmt)
    if fmt in BBoxFormat.formats():
        return bbox
    match fmt:
        case BBoxFormat.COCO2VOC  | BBoxFormat.XYWH2XYXY:
            return hbb_coco_to_voc(bbox, height, width)
        case BBoxFormat.COCO2YOLO | BBoxFormat.XYWH2CXCYN:
            return hbb_coco_to_yolo(bbox, height, width)
        case BBoxFormat.VOC2COCO  | BBoxFormat.XYXY2XYWH:
            return hbb_voc_to_coco(bbox, height, width)
        case BBoxFormat.VOC2YOLO  | BBoxFormat.XYXY2CXCYN:
            return hbb_voc_to_yolo(bbox, height, width)
        case BBoxFormat.YOLO2VOC  | BBoxFormat.CXCYN2XYXY:
            return hbb_yolo_to_voc(bbox, height, width)
        case BBoxFormat.YOLO2COCO | BBoxFormat.CXCYN2XYXY:
            return hbb_yolo_to_coco(bbox, height, width)
        case _:
            raise ValueError(f"[code] must be one of {BBoxFormat.conversion_codes()}, got {fmt}.")


def hbb_to_array(
    bbox       : torch.Tensor | np.ndarray,
    height     : int,
    width      : int,
    denormalize: bool = False
) -> np.ndarray:
    """Convert HBBs to a NumPy array.

    Args:
        bbox: HBBs as ``np.ndarray`` or ``torch.Tensor``, or list/tuple of [4+], [N, 4+], or [B, N, 4+].
        height: Image height.
        width: Image width.
        denormalize: Denormalize according to image dimensions if ``True``. Default is ``False``.

    Returns:
        HBBs as ``np.ndarray`` in [N, 4+] format.

    Raises:
        TypeError: If ``bbox`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    # Check shape
    if not 2 <= bbox.ndim <= 3:
        raise ValueError(f"[bbox]'s number of dimensions must be between 2 and 3, got {bbox.ndim}.")
    # Remove batch dimension
    bbox = hbb_to_2d(bbox)
    # Detach
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.detach().cpu()
    # Convert to numpy
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.numpy()
    # Denormalize image
    if denormalize:
        bbox = denormalize_hbb(bbox, height, width)#.round().astype(np.uint8)

    return bbox


def hbb_to_tensor(
    bbox     : torch.Tensor | np.ndarray | list | tuple,
    height   : int,
    width    : int,
    normalize: bool         = False,
    device   : torch.device = None
) -> torch.Tensor:
    """Convert HBBs to a PyTorch tensor.

    Args:
        bbox: HBBs as ``np.ndarray``, ``torch.Tensor``, or list/tuple of [4+] or [N, 4+].
        height: Image height.
        width: Image width.
        normalize: Normalize according to image dimensions if ``True``. Default is ``False``.
        device: Device to place tensor on, e.g., ``'cuda'`` or ``None`` for CPU.
            Default is ``None``.

    Returns:
        HBBs as ``torch.Tensor`` in [B, N, 4+] format.

    Raises:
        TypeError: If ``bbox`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    # Add batch dimension
    bbox = hbb_to_3d(bbox)
    # Convert to tensor
    if isinstance(bbox, np.ndarray):
        bbox = torch.from_numpy(bbox).contiguous().float()
    elif isinstance(bbox, torch.Tensor):
        bbox = bbox.float()
    else:
        raise TypeError(f"[bbox] must be a torch.Tensor or numpy.ndarray, got {type(bbox)}.")
    # Ensure float32 for model input.
    if bbox.dtype != torch.float32:
        bbox = bbox.float()
    # Normalize image
    if normalize:
        bbox = normalize_hbb(bbox, height, width)
    # Move to device
    if device is not None:
        bbox = bbox.to(device)
    bbox = bbox.contiguous()

    return bbox
