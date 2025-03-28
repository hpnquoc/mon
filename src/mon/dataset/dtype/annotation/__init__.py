#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements annotation types for labels and predictions."""

from __future__ import annotations

from typing import Optional

import mon.dataset.dtype.annotation.base
import mon.dataset.dtype.annotation.bbox
import mon.dataset.dtype.annotation.category
import mon.dataset.dtype.annotation.classlabel
import mon.dataset.dtype.annotation.image
import mon.dataset.dtype.annotation.value
from mon.core.rich import error_console
from mon.dataset.dtype.annotation.base import *
from mon.dataset.dtype.annotation.bbox import *
from mon.dataset.dtype.annotation.category import *
from mon.dataset.dtype.annotation.classlabel import *
from mon.dataset.dtype.annotation.image import *
from mon.dataset.dtype.annotation.value import *


# region Utils

def get_albumentation_target_type(annotation) -> str | None:
    """Returns Albumentations target type for an annotation.

    Args:
        annotation: Annotation object to check.

    Returns:
        Target type: ``"image"``, ``"mask"``, ``"bboxes"``, ``"keypoints"``, or
        ``"values"``; ``None`` if unknown.
    """
    if annotation in [ImageAnnotation, FrameAnnotation, DepthMapAnnotation]:
        return "image"
    elif annotation in [BBoxAnnotation, BBoxesAnnotation]:
        return "bboxes"
    elif annotation in [ClassificationAnnotation, RegressionAnnotation]:
        return "values"
    elif annotation in [SemanticSegmentationAnnotation]:
        return "mask"
    else:
        error_console.log(f"Unknown annotation type: {annotation}, {type(annotation)}.")
        return None


class DatapointAttributes(dict[str: Optional[Annotation]]):
    """Holds datapoint attributes as a ``dict``.

    Args:
        args: Positional arguments for ``dict`` initialization.
        kwargs: Keyword arguments for ``dict`` initialization.

    Attributes:
        Keys: Attribute names as ``str``.
        Values: Annotation types as ``Annotation`` or ``None``.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def to_tensor_fns(self) -> dict[str: Optional[callable]]:
        """Returns dict of functions to convert annotation to tensor.
    
        Returns:
            Dict mapping keys to ``to_tensor`` functions or ``None``.
        """
        return {k: getattr(v, "to_tensor", None) for k, v in self.items() if v}
    
    def collate_fns(self) -> dict[str: Optional[callable]]:
        """Returns dict of functions to collate annotation.
    
        Returns:
            Dict mapping keys to ``collate_fn`` functions or ``None``.
        """
        return {k: getattr(v, "collate_fn", None) for k, v in self.items() if v}
    
    def albumentation_target_types(self) -> dict[str: str]:
        """Returns dict of target types Albumentations expects.
    
        Returns:
            Dict mapping keys to target type strings.
        """
        target_types = {k: get_albumentation_target_type(v) for k, v in self.items() if v}
        target_types = {k: v for k, v in target_types.items() if v}
        return target_types
    
    def get_tensor_fn(self, key: str) -> Optional[callable]:
        """Returns function to convert annotation to tensor.
    
        Args:
            key: Key of the annotation.
    
        Returns:
            ``to_tensor`` function or ``None`` if not found.
        """
        return self.to_tensor_fns().get(key, None)
    
    def get_collate_fn(self, key: str) -> Optional[callable]:
        """Returns function to collate annotation.
    
        Args:
            key: Key of the annotation.
    
        Returns:
            ``collate_fn`` function or ``None`` if not found.
        """
        return self.collate_fns().get(key, None)
    
    def get_albumentation_target_type(self, key: str) -> Optional[str]:
        """Returns target type Albumentations expects.
    
        Args:
            key: Key of the annotation.
    
        Returns:
            Target type string or ``None`` if not found.
        """
        return self.albumentation_target_types().get(key, None)
    
# endregion
