#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements depth-based annotations."""

__all__ = [
    "DepthMapAnnotation",
]

from typing import Literal

import cv2

from mon import core
from mon.constants import DepthSource
from mon.vision.types import image as I


# ----- Annotation -----
class DepthMapAnnotation(I.ImageAnnotation):
    """Dense depth map annotation.
    
    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``image``.
    
    Args:
        path: Depth map file path.
        root: Root directory. Default is ``None``.
        source: Source of depth data from ``DepthSource``. Default is ``DepthSource.DAv2_ViTB``.
        flags: Flag to read image (e.g., ``cv2.IMREAD_GRAYSCALE``). Default is ``cv2.IMREAD_GRAYSCALE``.

    Raises:
        ValueError: If ``source`` is not in ``DepthSource``.
    """
    
    albumentation_target_type: str = "image"
    
    def __init__(
        self,
        path  : core.Path,
        root  : core.Path   = None,
        source: DepthSource = DepthSource.DAv2_ViTB,
        flags : int         = cv2.IMREAD_GRAYSCALE,
        *args, **kwargs
    ):
        super().__init__(path=path, root=root, flags=flags, *args, **kwargs)
        source = DepthSource.from_value(source)
        if source not in DepthSource:
            raise ValueError(f"[source] must be one of {DepthSource}, got {source}.")
        self.source = source
        # self.flags  = (cv2.IMREAD_COLOR if source and "c" in source else cv2.IMREAD_GRAYSCALE)
