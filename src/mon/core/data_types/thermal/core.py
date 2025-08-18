#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements InfraredMap class and its core properties.

Common Tasks:
    - Define the InfraredMap class (e.g., wrapper for ``numpy.ndarray`` or ``torch.Tensor``).
    - Access core properties.
"""

__all__ = [
    "InfraredMap",
]

from typing import Union

import cv2
import numpy as np
import torch

from mon.core.data_types import image as I
from mon.core.enum import InfraredSource
from mon.core.pathlib import Path


class InfraredMap(I.Image):
    """Infrared map.

    Args:
        data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``. Default is ``None``.
        path: Infrared map file path. Default is ``None``.
        root: Root directory for the infrared map. Default is ``None``.
        source: Source of infrared data. One of ``InfraredSource``.
            Default is ``InfraredSource.INFRARED``.
        flags: OpenCV flag to read image. Default is ``cv2.IMREAD_GRAYSCALE``.
        cache: If ``True``, caches image in memory. Default is ``False``.
    """

    def __init__(
        self,
        data  : Union[torch.Tensor, np.ndarray] = None,
        path  : Path           = None,
        root  : Path           = None,
        source: InfraredSource = InfraredSource.INFRARED,
        flags : int            = cv2.IMREAD_GRAYSCALE,
        cache : bool           = False,
    ):
        source = InfraredSource.from_value(source)
        if source not in InfraredSource:
            raise ValueError(f"[source] must be one of {InfraredSource}, got {source}.")

        super().__init__(data=data, path=path, root=root, flags=flags, cache=cache)
        self.source = source
