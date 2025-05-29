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
    "load_bbox_yolo",
]

import numpy as np

from mon import core
from mon.core import error_console
from mon.constants import BBoxFormat


# ----- Read -----
def load_bbox_coco(path: core.Path, verbose: bool = True) -> np.ndarray:
    """Load COCO-format bounding boxes from a ``.json`` file.

    Args:
        path: Label file path (one ``.json`` file for each image).
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


def load_bbox(path: core.Path, format: BBoxFormat, verbose: bool = True) -> np.ndarray:
    """Load bounding boxes from a label file.

    Args:
        path: Label file path.
        format: Bounding box format of the label file.
        verbose: Verbosity. Defaults is ``True``.

    Returns:
        Boxes as ``np.ndarray`` in [N, 4+], output format varies by code.

    Raises:
        ValueError: If ``code`` is invalid.
    """
    code = BBoxFormat.from_value(value=format)
    match code:
        case BBoxFormat.COCO | BBoxFormat.XYWH:
            return load_bbox_coco(path, verbose)
        case BBoxFormat.YOLO | BBoxFormat.CXCYN:
            return load_bbox_yolo(path, verbose)
        case _:
            raise ValueError(f"[code] must be one of {BBoxFormat.formats()}, got {code}.")


# ----- Write -----
