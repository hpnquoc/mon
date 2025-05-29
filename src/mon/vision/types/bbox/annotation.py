#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements bounding box annotations."""

__all__ = [
    "BBoxAnnotation",
    "BBoxesAnnotation",
]

from typing import Any

import numpy as np
import torch

from mon import core
from mon.constants import BBoxFormat
from mon.nn import _size_2_t
from mon.vision.types import image as I


# ----- Annotation -----
class BBoxAnnotation(core.Annotation):
    """Bounding box annotation in an image with coordinates and optional mask.
    
    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``bboxes``.
    
    Args:
        class_id: Integer class ID, ``-1`` for unknown.
        bbox: Box coordinates as [4]-shaped array, list, or tuple.
        confidence: Confidence score in [0.0, 1.0]. Default is ``1.0``.
    """
    
    albumentation_target_type: str = "bboxes"
    
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
            ``numpy.ndarray`` of shape [4] with box coordinates.
        """
        return self._bbox
    
    @bbox.setter
    def bbox(self, bbox: np.ndarray | list | tuple):
        """Sets the bounding box coordinates.

        Args:
            bbox: Coordinates as ``numpy.ndarray``, list, or tuple of shape [4].

        Raises:
            ValueError: If ``bbox`` is not a 1D array of size ``4``.
        """
        bbox_array = np.asarray(bbox)
        if bbox_array.ndim != 1 or bbox_array.size != 4:
            raise ValueError(f"[bbox] must be a 1D array of size 4, got {bbox_array}.")
        self._bbox = bbox_array
    
    @property
    def confidence(self) -> float:
        """Returns the confidence score.

        Returns:
            ``float`` in [0.0, 1.0] representing confidence.
        """
        return self._confidence
    
    @confidence.setter
    def confidence(self, confidence: float):
        """Sets the confidence score.

        Args:
            confidence: Confidence value as ``float``.

        Raises:
            ValueError: If ``confidence`` is not in [0.0, 1.0].
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"[confidence] must be in [0.0, 1.0], got {confidence}.")
        self._confidence = confidence
    
    @property
    def data(self) -> list[float | int]:
        """Returns the annotation data.

        Returns:
            List of [x_min, y_min, x_max, y_max, confidence, class_id].
        """
        return [*self.bbox, self.confidence, self.class_id]
    
    @staticmethod
    def to_tensor(data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Converts input data to a tensor.

        Args:
            data: Input as ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            ``torch.Tensor`` of input data.
        """
        return torch.as_tensor(data)
    
    @staticmethod
    def collate_fn(batch: list[torch.Tensor] | list[np.ndarray]) -> torch.Tensor | np.ndarray | None:
        """Collates batch data for ``torch.utils.data.DataLoader``.

        Args:
            batch: List of items as ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            Collated ``torch.Tensor``, ``numpy.ndarray``, or ``None`` if empty/mixed.
        """
        if not batch:
            return None
        if isinstance(batch[0], torch.Tensor):
            return torch.stack(batch, dim=0)
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch, axis=0)
        return None


class BBoxesAnnotation(list[BBoxAnnotation]):
    """List of bounding box annotations in an image.
    
    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``bboxes``.
    """
    
    albumentation_target_type: str = "bboxes"
    
    @property
    def data(self) -> list[list[float | int]] | None:
        """Returns data of all bounding box annotations.

        Returns:
            List of [x_min, y_min, x_max, y_max, confidence, class_id] or ``None`` if empty.
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
            List of ``numpy.ndarray`` coordinates, each shape [4].
        """
        return [item.bbox for item in self]
    
    @property
    def confidences(self) -> list[float]:
        """Returns confidence scores of all bounding box annotations.

        Returns:
            List of ``confidence`` values in [0.0, 1.0].
        """
        return [item.confidence for item in self]


class BBoxesAnnotation2(core.Annotation):
    """List of bounding boxes from a label file.

    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``bboxes``.

    Args:
        path: Label file path.
        imgsz : Size of the image as ``_size_2_t``. Default is ``None``.
        format: Bounding boxes format from ``'BBoxFormat'``. Default is ``'BBoxFormat.YOLO'``.
    """

    albumentation_target_type: str = "bboxes"

    def __init__(
        self,
        path  : core.Path,
        imgsz : _size_2_t  = None,
        format: BBoxFormat = BBoxFormat.YOLO,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.path   = path
        self.imgsz  = I.image_size(imgsz) if imgsz else None
        self.format = BBoxFormat.from_value(format)
        self.bboxes: list[BBoxAnnotation] = []

    @property
    def path(self) -> core.Path:
        """Returns the label file path."""
        return self._path

    @path.setter
    def path(self, path: core.Path):
        """Sets the label file path.

        Args:
            path: Label file path.

        Raises:
            ValueError: If ``path`` is not a valid label file path.
        """
        path_obj = core.Path(path)
        if not path or not path_obj.is_txt_file():
            raise ValueError(f"[path] must be a valid image path, got {path}.")
        self._path = path_obj

    @property
    def name(self) -> str:
        """Returns the image file name.

        Returns:
            ``str`` of the image file name.
        """
        return self.path.name

    @property
    def stem(self) -> str:
        """Returns the stem of the image file path.

        Returns:
            ``str`` of the image file path stem.
        """
        return self.path.stem

    @property
    def data(self) -> list:
        pass

    @property
    def meta(self) -> dict:
        """Returns metadata about the bbox annotation.

        Returns:
            Dict with keys ``name``, ``stem``, ``path``, ``imgsz``, ``num_bboxes``, and ``hash``.
        """
        return {
            "name"      : self.name,
            "stem"      : self.stem,
            "path"      : self.path,
            "imgsz"     : self.imgsz,
            "num_bboxes": len(self.bboxes),
            "hash"      : self.path.stat().st_size if isinstance(self.path, core.Path) else None,
        }

    def load(
        self,
        path  : core.Path  = None,
        imgsz : _size_2_t  = None,
        format: BBoxFormat = None,
        cache : bool       = False
    ) -> np.ndarray:
        """Loads the image into memory.

        Args:
            path: Path to image file. Default is ``None``.
            imgsz: Size of the image as ``_size_2_t``. Default is ``None``.
            format: Bounding boxes format from ``'BBoxFormat'``. Default is ``None``.
            cache: If ``True``, caches image. Default is ``False``.

        Returns:
            ``numpy.ndarray`` in [H, W, C] format, values in [0, 255].
        """
        pass

    @staticmethod
    def to_tensor(data: torch.Tensor | np.ndarray, *args, **kwargs) -> torch.Tensor:
        """Converts input data to a tensor.

        Args:
            data: Input as ``torch.Tensor`` or ``numpy.ndarray``.
            normalize: If ``True``, normalizes data. Default is ``True``.

        Returns:
            ``torch.Tensor`` of converted data.
        """
        pass

    @staticmethod
    def collate_fn(batch: list[Any]) -> Any:
        """Collates batch data for ``torch.utils.data.DataLoader``.

        Args:
            batch: List of images as ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            Collated ``torch.Tensor``, ``numpy.ndarray``, or ``None`` if empty/invalid.
        """
        pass
