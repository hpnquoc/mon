#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements depth-based annotations."""

__all__ = [
    "DepthMapAnnotation",
]

from typing import Literal

import cv2

from mon import core
from mon.globals import DEPTH_DATA_SOURCES
from mon.vision.types import image as I


class DepthMapAnnotation(I.ImageAnnotation):
    """Dense depth map annotation.

    Args:
        path: Path to depth map file as ``core.Path`` or ``str``.
        root: Root dir as ``core.Path`` or ``str``. Default is ``None``.
        source: Source of depth data from ``DEPTH_DATA_SOURCES``. Default is ``None``.
        flags: Flag to read image (e.g., ``cv2.IMREAD_COLOR``).
            Default is ``cv2.IMREAD_COLOR``.

    Raises:
        ValueError: If ``source`` is not in ``DEPTH_DATA_SOURCES``.
    """
    
    def __init__(
        self,
        path  : core.Path | str,
        root  : core.Path | str = None,
        source: Literal[*DEPTH_DATA_SOURCES] = None,
        flags : int = cv2.IMREAD_COLOR,
        *args, **kwargs
    ):
        super().__init__(path=path, root=root, flags=flags, *args, **kwargs)
        if source not in DEPTH_DATA_SOURCES:
            raise ValueError(f"[source] must be one of {DEPTH_DATA_SOURCES}, got {source}.")
        self.source = source
        self.flags  = (cv2.IMREAD_GRAYSCALE if source and "g" in source else cv2.IMREAD_COLOR)
