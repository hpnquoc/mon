#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements HBBs annotations."""

__all__ = [
    "HBBAnnotation",
    "HBBsAnnotation",
]

import numpy as np
import torch

from mon import core
from mon.constants import BBoxFormat
from mon.nn import _size_2_t
from mon.vision.types import image as I
from mon.vision.types.bbox.hbb import processing, io


# ----- Annotation -----
class HBBAnnotation(core.Annotation):
    """HBB annotation in an image with coordinates and optional mask.
    
    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``bboxes``.
    
    Args:
        class_id: Integer class ID, ``-1`` for unknown.
        bbox: Bounding box coordinates as [4]-shaped array, list, or tuple.
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
        """Returns the HBBs coordinates.

        Returns:
            ``numpy.ndarray`` of shape [4] with box coordinates.
        """
        return self._bbox
    
    @bbox.setter
    def bbox(self, bbox: np.ndarray | list | tuple):
        """Sets the HBBs coordinates.

        Args:
            bbox: Coordinates as ``numpy.ndarray``, list, or tuple of shape [4].

        Raises:
            ValueError: If ``bbox`` is not a 1D array of size ``4``.
        """
        b = np.asarray(bbox)
        if b.ndim != 1 or b.size != 4:
            raise ValueError(f"[bbox] must be a 1D array of size 4, got {b}.")
        self._bbox = b
    
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


class HBBsAnnotation(core.Annotation):
    """List of HBBs annotation for a single image.

    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``bboxes``.

    Args:
        path: Label file path.
        root: Root directory. Default is ``None``.
        fmt: Bounding box format from ``'BBoxFormat'``. Default is ``'BBoxFormat.YOLO'``.
        imgsz: Image size. Default is ``None``.
    """

    albumentation_target_type: str = "bboxes"

    def __init__(
        self,
        path : core.Path,
        root : core.Path  = None,
        fmt  : BBoxFormat = BBoxFormat.YOLO,
        imgsz: _size_2_t  = None,
        cache: bool       = True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.path    = path
        self.root    = root
        self.imgsz   = imgsz
        self.cache   = cache
        self._bboxes = None

        fmt = BBoxFormat.from_value(fmt)
        if fmt in BBoxFormat.conversion_codes():
            src_fmt, dst_fmt = fmt.value.split("_to_")
            src_fmt = BBoxFormat.from_value(value=src_fmt)
            dst_fmt = BBoxFormat.from_value(value=dst_fmt)
        else:
            src_fmt = dst_fmt = fmt
        self._fmt     = fmt
        self._src_fmt = src_fmt
        self._dst_fmt = dst_fmt

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
        if path and core.Path(path).is_file():
            self._path = core.Path(path)
        else:
            raise ValueError(f"[path] must be a valid file path, got {path}.")

    @property
    def imgsz(self) -> tuple[int, int] | None:
        """Returns the image size.

        Returns:
            Tuple of [H, W] for image dimensions.
        """
        return self._imgsz

    @imgsz.setter
    def imgsz(self, imgsz: _size_2_t):
        """Sets the image size.

        Args:
            imgsz: Image size as [H, W] tuple or list.

        Raises:
            ValueError: If ``imgsz`` is not a valid size.
        """
        self._imgsz = I.image_size(imgsz) if imgsz else None

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
    def data(self) -> np.ndarray | None:
        """Returns the bbox labels.

        Returns:
            ``numpy.ndarray`` of image data or ``None`` if not loaded.
        """
        return self._bboxes if self._bboxes is not None else self.load()

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
            "num_bboxes": len(self._bboxes)        if isinstance(self._bboxes, np.ndarray) else None,
            "hash"      : self.path.stat().st_size if isinstance(self.path,    core.Path)  else None,
        }

    def load(self, reload: bool = False) -> np.ndarray:
        """Loads the bboxes into memory.

        Args:
            reload: If ``True``, reloads the image even if already cached. Default is ``False``.

        Returns:
            ``numpy.ndarray`` in [H, W, C] format, values in [0, 255].
        """
        # Return the bboxes if it is already loaded
        if not reload and self._bboxes is not None:
            return self._bboxes
        # Load the bboxes from label file
        bboxes = io.load_hbb(
            path      = self.path,
            fmt       = self._fmt,
            height    = self.imgsz[0] if self.imgsz else None,
            width     = self.imgsz[1] if self.imgsz else None,
            to_tensor = False,
            normalize = False
        )
        # Cache the bboxes if needed
        self._bboxes = bboxes if self.cache else None
        return bboxes

    @staticmethod
    def to_tensor(
        data     : torch.Tensor | np.ndarray,
        height   : int  = None,
        width    : int  = None,
        normalize: bool = True,
        *args, **kwargs
    ) -> torch.Tensor:
        """Converts input data to a tensor.

        Args:
            data: Input as ``torch.Tensor`` or ``numpy.ndarray``.
            height: Image height. Default is ``None``.
            width: Image width. Default is ``None``.
            normalize: If ``True``, normalizes data. Default is ``True``.

        Returns:
            ``torch.Tensor`` of converted data.
        """
        return processing.hbb_to_tensor(data, height, width, normalize)

    @staticmethod
    def collate_fn(batch: list) -> torch.Tensor | np.ndarray | None:
        """Collates batch data for ``torch.utils.data.DataLoader``.

        Args:
            batch: List of images as ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            Collated ``torch.Tensor``, ``numpy.ndarray``, or ``None`` if empty/invalid.
        """
        if not batch:
            return None
        return processing.hbb_to_3d(batch)
