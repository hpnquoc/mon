#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Mask class and core properties.

Common Tasks:
    - Define the Mask class (e.g., wrapper for ``numpy.ndarray`` or ``torch.Tensor``).
    - Access core properties.
"""

__all__ = [
    "SemanticMask",
]

from typing import Union

import cv2
import numpy as np
import torch

from mon.core.pathlib import Path
from mon.core.data_types import image as I


class SemanticMask(I.Image):
    """Segmentation mask.

    Args:
        data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``. Default is ``None``.
        path: Semantic mask file path. Default is ``None``.
        root: Root directory for the semantic mask. Default is ``None``.
        flags: OpenCV flag to read image. Default is ``cv2.IMREAD_COLOR_BGR``.
        cache: If ``True``, caches image in memory. Default is ``False``.
    """

    def __init__(
        self,
        data : Union[torch.Tensor, np.ndarray] = None,
        path : Path = None,
        root : Path = None,
        flags: int  = cv2.IMREAD_COLOR_BGR,
        cache: bool = False,
    ):
        super().__init__(data=data, path=path, root=root, flags=flags, cache=cache)
