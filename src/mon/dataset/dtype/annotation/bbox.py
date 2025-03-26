#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bounding Box Annotation.

This module implements bounding box annotations.
"""

from __future__ import annotations

__all__ = [
    "BBoxAnnotation",
    "BBoxesAnnotation",
]

import numpy as np
import torch

from mon.dataset.dtype.annotation import base


# region BBox

class BBoxAnnotation(base.Annotation):
    """Bounding box annotation in an image, including coordinates and optional instance mask.

    Args:
        class_id: Integer class ID, where ``-1`` indicates unknown.
        bbox: Bounding box coordinates as a ``[4]``-shaped array, list, or tuple.
        confidence: Confidence score in ``[0.0, 1.0]``. Default is ``1.0``.
    """
    
    def __init__(
        self,
        class_id  : int,
        bbox      : np.ndarray | list | tuple,
        confidence: float = 1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_id   = class_id
        self.bbox       = bbox
        self.confidence = confidence
    
    @property
    def bbox(self) -> np.ndarray:
        """Returns the bounding box coordinates.

        Returns:
            ``numpy.ndarray`` of shape ``[4]`` with bounding box coordinates.
        """
        return self._bbox
    
    @bbox.setter
    def bbox(self, bbox: np.ndarray | list | tuple):
        """Sets the bounding box coordinates.

        Args:
            bbox: Coordinates as a ``numpy.ndarray``, list, or tuple of shape ``[4]``.

        Raises:
            ValueError: If ``[bbox]`` is not a 1D array of size ``4``.
        """
        bbox_array = np.asarray(bbox)
        if bbox_array.ndim != 1 or bbox_array.size != 4:
            raise ValueError(f"[bbox] must be a 1D array of size 4, but got [{bbox_array}]")
        self._bbox = bbox_array
    
    @property
    def confidence(self) -> float:
        """Returns the confidence score.

        Returns:
            ``float`` representing the confidence in ``[0.0, 1.0]``.
        """
        return self._confidence
    
    @confidence.setter
    def confidence(self, confidence: float):
        """Sets the confidence score.

        Args:
            confidence: Confidence value as a ``float``.

        Raises:
            ValueError: If ``[confidence]`` is not in ``[0.0, 1.0]``.
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"[confidence] must be in [0.0, 1.0], but got [{confidence}]")
        self._confidence = confidence
    
    @property
    def data(self) -> list[float | int]:
        """Returns the annotation data.

        Returns:
            List of ``[x_min, y_min, x_max, y_max, confidence, class_id]``.
        """
        return [*self.bbox, self.confidence, self.class_id]
    
    @staticmethod
    def to_tensor(data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Converts input data to a tensor.

        Args:
            data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            ``torch.Tensor`` of the input data.
        """
        return torch.as_tensor(data)
    
    @staticmethod
    def collate_fn(batch: list[torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray | None:
        """Collates batch data for ``torch.utils.data.DataLoader``.

        Args:
            batch: List of items as ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            Collated ``torch.Tensor``, ``numpy.ndarray``, or ``None`` if batch is empty or mixed.
        """
        if not batch:
            return None
        if isinstance(batch[0], torch.Tensor):
            return torch.stack(batch, dim=0)
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch, axis=0)
        return None


class BBoxesAnnotation(list[BBoxAnnotation]):
    """List of bounding box annotations in an image."""
    
    @property
    def data(self) -> list[list[float | int]] | None:
        """Returns data of all bounding box annotations.

        Returns:
            List of data (``[x_min, y_min, x_max, y_max, confidence, class_id]``) per annotation, or ``None`` if empty.
        """
        return [item.data for item in self] if self else None
    
    @property
    def class_ids(self) -> list[int]:
        """Returns class IDs of all bounding box annotations.

        Returns:
            List of ``class_id`` values.
        """
        return [item.class_id for item in self]
    
    @property
    def bboxes(self) -> list[np.ndarray]:
        """Returns bounding boxes of all bounding box annotations.

        Returns:
            List of ``numpy.ndarray`` bounding box coordinates, each of shape ``[4]``.
        """
        return [item.bbox for item in self]
    
    @property
    def confidences(self) -> list[float]:
        """Returns confidence scores of all bounding box annotations.

        Returns:
            List of ``confidence`` values in ``[0.0, 1.0]``.
        """
        return [item.confidence for item in self]
    
# endregion
