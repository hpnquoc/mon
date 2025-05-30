#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements input/output operations for bbox label file.

Common Tasks:
    - Load bboxes from disk.
    - Save bboxes to disk.
    - Batch I/O.
    - Metadata handling.
"""

__all__ = [
    "load_bbox",
    "load_bbox_coco",
    "load_bbox_voc",
    "load_bbox_yolo",
]

import numpy as np
import torch

from mon import core
from mon.constants import BBoxFormat
from mon.core import error_console
from mon.vision.types.bbox import processing


# ----- Reading -----
def load_bbox_coco(path: core.Path, verbose: bool = True) -> np.ndarray:
    """Load COCO-format bounding boxes from a ``.json`` file.

    Args:
        path: Label file path (one ``.json`` file for each image).
        verbose: Verbosity. Defaults is ``True``.
    """
    raise NotImplementedError


def load_bbox_voc(path: core.Path, verbose: bool = True) -> np.ndarray:
    """Load VOC-format bounding boxes from a ``.xml`` file.

    Args:
        path: Label file path (one ``.xml`` file for each image).
        verbose: Verbosity. Defaults is ``True``.
    """
    raise NotImplementedError


def load_bbox_yolo(path: core.Path, verbose: bool = True) -> np.ndarray:
    """Load YOLO-format bounding boxes from a ``.txt`` file.

    Each line in the file should contain:
        <class_id> <center_x> <center_y> <width> <height> <confidence (optional)>
    where:
        - ``class_id`` is the class index (0-based).
        - ``x_center``, ``y_center``, ``width``, and ``height`` are normalized values
            relative to the image dimensions.
        - ``confidence`` is an optional value representing the confidence score.

    Args:
        path: Label file path (one ``.txt`` for each image).
        verbose: Verbosity. Defaults is ``True``.
    """
    path = core.Path(path)
    if not path.is_txt_file(exist=True):
        if verbose:
            error_console.print(f"[path] must be a valid .txt file, got {path}.")
        return np.empty((0, 6), dtype=np.float32)

    with open(path, "r") as f:
        ls = f.readlines()
    ls = [l.strip().split(" ") for l in ls]
    ls = [l for l in ls if len(l) >= 5]

    if len(ls) == 0:
        if verbose:
            error_console.print(f"No bounding boxes found in {path}.")
        return np.empty((0, 6), dtype=np.float32)

    ls = np.array(ls, dtype=np.float32)
    c, cx_n, cy_n, w_n, h_n, *rest = ls.T
    return np.stack([cx_n, cy_n, w_n, h_n, c] + rest, axis=-1)


def load_bbox(
    path     : core.Path,
    fmt      : BBoxFormat,
    height   : int          = None,
    width    : int          = None,
    to_tensor: bool         = False,
    normalize: bool         = False,
    device   : torch.device = None,
    verbose  : bool         = False
) -> np.ndarray:
    """Load bounding boxes from a label file.

    Args:
        path: Label file path.
        fmt: Bounding box format of the label file.
        height: Image height. Default is ``None``.
        width: Image width. Default is ``None``.
        to_tensor: Convert to ``torch.Tensor`` if ``True``. Default is ``False``.
        normalize: Normalize bounding boxes to [0.0, 1.0] if ``True``. Default is ``False``.
        device: Device to place tensor on, e.g., ``'cuda'`` or ``None`` for CPU.
            Default is ``None``.
        verbose: Verbosity. Defaults is ``False``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4+], output format varies by code.

    Raises:
        ValueError: If ``format`` is invalid.
    """
    fmt = BBoxFormat.from_value(value=fmt)
    if fmt in BBoxFormat.conversion_codes():
        src_fmt = fmt.value.split("_to_")[0]
        src_fmt = BBoxFormat.from_value(value=src_fmt)
    else:
        fmt     = None
        src_fmt = fmt

    match src_fmt:
        case BBoxFormat.COCO | BBoxFormat.XYWH:
            bbox = load_bbox_coco(path, verbose)
        case BBoxFormat.VOC  | BBoxFormat.XYXY:
            bbox = load_bbox_voc(path, verbose)
        case BBoxFormat.YOLO | BBoxFormat.CXCYN:
            bbox = load_bbox_yolo(path, verbose)
        case _:
            raise ValueError(f"[src_fmt] must be one of {BBoxFormat.formats()}, got {src_fmt}.")

    if (fmt or to_tensor) and (height is None or width is None):
        raise ValueError("[height] and [width] must be provided when converting bounding boxes.")

    if fmt:
        bbox = processing.convert_bbox(bbox=bbox, fmt=fmt, height=height, width=width)

    if to_tensor:
        bbox = processing.bbox_to_tensor(
            bbox      = bbox,
            height    = height,
            width     = width,
            normalize = normalize,
            device    = device
        )

    return bbox


# ----- Writing -----
