#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements thermal and infrared annotations."""

__all__ = [
    "InfraredAnnotation",
]

from typing import Literal

import cv2

from mon import core
from mon.constants import InfraredSource
from mon.vision.types import image as I


# ----- Annotation -----
class InfraredAnnotation(I.ImageAnnotation):
    """Dense infrared map annotation.
    
    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``image``.
    
    Args:
        path: Infrared map file path.
        root: Root directory. Default is ``None``.
        source: Source of depth data from ``InfraredSource``. Default is ``InfraredSource.INFRARED``.
        flags: Flag to read image (e.g., ``cv2.IMREAD_GRAYSCALE``). Default is ``cv2.IMREAD_GRAYSCALE``.
    """
    
    albumentation_target_type: str = "image"
    
    def __init__(
        self,
        path  : core.Path,
        root  : core.Path      = None,
        source: InfraredSource = InfraredSource.INFRARED,
        flags : int            = cv2.IMREAD_GRAYSCALE,
        *args, **kwargs
    ):
        super().__init__(path=path, root=root, flags=flags, *args, **kwargs)
        source = InfraredSource.from_value(source)
        if source not in InfraredSource:
            raise ValueError(f"[source] must be one of {InfraredSource}, got {source}.")
        self.source = source
