#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Annotation.

This module implements annotations that take the form of an image.
"""

from __future__ import annotations

__all__ = [
    "DepthMapAnnotation",
    "FrameAnnotation",
    "ImageAnnotation",
    "SemanticSegmentationAnnotation",
]

from typing import Literal

import cv2
import numpy as np
import torch

from mon import core, vision
from mon.dataset.dtype.annotation import base, classlabel
from mon.globals import DEPTH_DATA_SOURCES

ClassLabels = classlabel.ClassLabels


# region Image

class ImageAnnotation(base.Annotation):
    """Image annotation for another image.

    Args:
        path: Path to the image file as a ``core.Path`` or ``str``.
        root: Root directory as a ``core.Path`` or ``str``. Default is ``None``.
        flags: Flag to read the image, one of:
            - ``cv2.IMREAD_UNCHANGED``           = -1
            - ``cv2.IMREAD_GRAYSCALE``           = 0
            - ``cv2.IMREAD_COLOR``               = 1
            - ``cv2.IMREAD_ANYDEPTH``            = 2
            - ``cv2.IMREAD_ANYCOLOR``            = 4
            - ``cv2.IMREAD_LOAD_GDAL``           = 8
            - ``cv2.IMREAD_REDUCED_GRAYSCALE_2`` = 16
            - ``cv2.IMREAD_REDUCED_COLOR_2``     = 17
            - ``cv2.IMREAD_REDUCED_GRAYSCALE_4`` = 32
            - ``cv2.IMREAD_REDUCED_COLOR_4``     = 33
            - ``cv2.IMREAD_REDUCED_GRAYSCALE_8`` = 64
            - ``cv2.IMREAD_REDUCED_COLOR_8``     = 65
            - ``cv2.IMREAD_IGNORE_ORIENTATION``  = 128
            Default is ``cv2.IMREAD_COLOR``.
    """
    
    def __init__(
        self,
        path : core.Path | str,
        root : core.Path | str = None,
        flags: int             = cv2.IMREAD_COLOR,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root   = root
        self.path   = path
        self.flags  = flags
        self.image  = None
        self._shape = None
    
    @property
    def path(self) -> core.Path:
        """Returns the image file path.

        Returns:
            ``core.Path`` representing the image file path.
        """
        return self._path
    
    @path.setter
    def path(self, path: core.Path | str):
        """Sets the image file path.

        Args:
            path: Path to the image file as a ``core.Path`` or ``str``.

        Raises:
            ValueError: If ``[path]`` is not a valid image file path.
        """
        path_obj = core.Path(path)
        if not path or not path_obj.is_image_file():
            raise ValueError(f"[path] must be a valid image path, but got [{path}].")
        self._path  = path_obj
        self._shape = vision.read_image_shape(path=self._path)
    
    @property
    def name(self) -> str:
        """Returns the image file name.

        Returns:
            ``str`` representing the image file name.
        """
        return self.path.name
    
    @property
    def stem(self) -> str:
        """Returns the stem of the image file path.

        Returns:
            ``str`` representing the stem of the image file path.
        """
        return self.path.stem
    
    @property
    def shape(self) -> tuple[int, int, int]:
        """Returns the image shape.

        Returns:
            Tuple of ``[H, W, C]`` representing the image dimensions.
        """
        return self._shape
    
    @property
    def data(self) -> np.ndarray | None:
        """Returns the image data.

        Loads the image without caching if not already loaded, otherwise returns cached data.

        Returns:
            ``numpy.ndarray`` of image data or ``None`` if not loaded.
        """
        return self.image if self.image is not None else self.load(cache=False)
    
    @property
    def meta(self) -> dict:
        """Returns metadata about the image.

        Returns:
            Dict with keys ``name``, ``stem``, ``path``, ``shape``, and ``hash``.
        """
        return {
            "name" : self.name,
            "stem" : self.stem,
            "path" : self.path,
            "shape": self.shape,
            "hash" : self.path.stat().st_size if isinstance(self.path, core.Path) else None,
        }
    
    def load(
        self,
        path : core.Path | str = None,
        flags: int             = None,
        cache: bool            = False
    ) -> np.ndarray:
        """Loads the image into memory.

        Args:
            path: Path to the image file as a ``core.Path`` or ``str``. Default is ``None``.
            flags: Flag to read the image. Default is ``None``.
            cache: If ``True``, caches the image in memory. Default is ``False``.

        Returns:
            ``numpy.ndarray`` in ``[H, W, C]`` format with values in ``[0, 255]``.
        """
        if self.image is not None:
            return self.image
        load_path  = path or self.path
        load_flags = flags or self.flags
        image      = vision.read_image(load_path, load_flags, to_tensor=False, normalize=False)
        if self._shape != image.shape:
            self._shape = image.shape
        self.image = image if cache else None
        if path:
            self.path = load_path
        if flags:
            self.flags = load_flags
        return image
    
    @staticmethod
    def to_tensor(
        data     : torch.Tensor | np.ndarray,
        keepdim  : bool = False,
        normalize: bool = True
    ) -> torch.Tensor:
        """Converts input data to a tensor.

        Args:
            data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``.
            keepdim: If ``True``, retains input dimensions. Default is ``False``.
            normalize: If ``True``, normalizes the data. Default is ``True``.

        Returns:
            ``torch.Tensor`` of the converted data.
        """
        return vision.to_image_tensor(data, keepdim, normalize)
    
    @staticmethod
    def collate_fn(batch: list[torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray | None:
        """Collates batch data for ``torch.utils.data.DataLoader``.

        Args:
            batch: List of images as ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            Collated ``torch.Tensor``, ``numpy.ndarray``, or ``None`` if batch is empty or invalid.
        """
        if not batch:
            return None
        return vision.to_4d_image(batch)


class FrameAnnotation(base.Annotation):
    """Image annotation of a video frame.

    Args:
        index: Integer index of the frame in the video.
        frame: Ground-truth image as a ``numpy.ndarray``.
        path: Path to the video file as a ``core.Path`` or ``str``. Default is ``None``.
    """
    
    def __init__(
        self,
        index: int,
        frame: np.ndarray,
        path : core.Path | str = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.index = index
        self.frame = frame
        self.path  = path
        self.shape = vision.get_image_shape(image=frame)
    
    @property
    def path(self) -> core.Path:
        """Returns the video file path.

        Returns:
            ``core.Path`` representing the video file path or ``None`` if not set.
        """
        return self._path
    
    @path.setter
    def path(self, path: core.Path | str | None):
        """Sets the video file path.

        Args:
            path: Path to the video file as a ``core.Path`` or ``str`` or ``None``.

        Raises:
            ValueError: If ``[path]`` is not a valid video file path when provided.
        """
        if path is not None:
            path_obj = core.Path(path)
            if not path_obj.is_video_file():
                raise ValueError(f"[path] must be a valid video path, but got [{path}].")
            self._path = path_obj
        else:
            self._path = None
    
    @property
    def name(self) -> str:
        """Returns the frame name.

        Returns:
            ``str`` from ``path.name`` if available, else ``index`` as a string.
        """
        return self.path.name if self.path else str(self.index)
    
    @property
    def stem(self) -> str:
        """Returns the stem of the frame path.

        Returns:
            ``str`` from ``path.stem`` if available, else ``index`` as a string.
        """
        return self.path.stem if self.path else str(self.index)
    
    @property
    def data(self) -> np.ndarray:
        """Returns the frame data.

        Returns:
            ``numpy.ndarray`` of the frame data.
        """
        return self.frame
    
    @property
    def meta(self) -> dict:
        """Returns metadata about the frame.

        Returns:
            Dict with keys ``index``, ``name``, ``stem``, ``path``, ``shape``, and ``hash``.
        """
        return {
            "index": self.index,
            "name" : self.name,
            "stem" : self.stem,
            "path" : self.path,
            "shape": self.shape,
            "hash" : self.path.stat().st_size if self.path else None,
        }
    
    @staticmethod
    def to_tensor(
        data     : torch.Tensor | np.ndarray,
        keepdim  : bool = False,
        normalize: bool = True
    ) -> torch.Tensor:
        """Converts input data to a tensor.

        Args:
            data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``.
            keepdim: If ``True``, retains input dimensions. Default is ``False``.
            normalize: If ``True``, normalizes the data. Default is ``True``.

        Returns:
            ``torch.Tensor`` of the converted data.
        """
        return vision.to_image_tensor(data, keepdim, normalize)
    
    @staticmethod
    def collate_fn(batch: list[torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray | None:
        """Collates batch data for ``torch.utils.data.DataLoader``.

        Args:
            batch: List of images as ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            Collated ``torch.Tensor``, ``numpy.ndarray``, or ``None`` if batch is empty or invalid.
        """
        if not batch:
            return None
        return vision.to_4d_image(batch)
    
# endregion


# region Depth Map

class DepthMapAnnotation(ImageAnnotation):
    """Dense depth map annotation for an image.

    Args:
        path: Path to the depth map file as a ``core.Path`` or ``str``.
        root: Root directory as a ``core.Path`` or ``str``. Default is ``None``.
        source: Source of depth data, one of ``DEPTH_DATA_SOURCES``. Default is ``None``.
        flags: Flag to read the image, e.g., ``cv2.IMREAD_COLOR``. Default is ``cv2.IMREAD_COLOR``.

    Raises:
        ValueError: If ``[source]`` is not in ``DEPTH_DATA_SOURCES``.
    """
    
    def __init__(
        self,
        path  : core.Path | str,
        root  : core.Path | str              = None,
        source: Literal[*DEPTH_DATA_SOURCES] = None,
        flags : int                          = cv2.IMREAD_COLOR,
        *args, **kwargs
    ):
        super().__init__(path=path, root=root, flags=flags, *args, **kwargs)
        if source not in DEPTH_DATA_SOURCES:
            raise ValueError(f"[source] must be one of {DEPTH_DATA_SOURCES}, but got [{source}].")
        self.source = source
        self.flags  = (cv2.IMREAD_GRAYSCALE if source and "g" in source else cv2.IMREAD_COLOR)
        
# endregion


# region Segmentation

class SemanticSegmentationAnnotation(base.Annotation):
    """Semantic segmentation annotation (mask) for an image.

    Args:
        path: Path to the image file as a ``core.Path`` or ``str``.
        root: Root directory as a ``core.Path`` or ``str``. Default is ``None``.
        flags: Flag to read the image, one of:
            - ``cv2.IMREAD_UNCHANGED``           = -1
            - ``cv2.IMREAD_GRAYSCALE``           = 0
            - ``cv2.IMREAD_COLOR``               = 1
            - ``cv2.IMREAD_ANYDEPTH``            = 2
            - ``cv2.IMREAD_ANYCOLOR``            = 4
            - ``cv2.IMREAD_LOAD_GDAL``           = 8
            - ``cv2.IMREAD_REDUCED_GRAYSCALE_2`` = 16
            - ``cv2.IMREAD_REDUCED_COLOR_2``     = 17
            - ``cv2.IMREAD_REDUCED_GRAYSCALE_4`` = 32
            - ``cv2.IMREAD_REDUCED_COLOR_4``     = 33
            - ``cv2.IMREAD_REDUCED_GRAYSCALE_8`` = 64
            - ``cv2.IMREAD_REDUCED_COLOR_8``     = 65
            - ``cv2.IMREAD_IGNORE_ORIENTATION``  = 128
            Default is ``cv2.IMREAD_COLOR``.
    """
    
    def __init__(
        self,
        path : core.Path | str,
        root : core.Path | str = None,
        flags: int             = cv2.IMREAD_COLOR,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root   = root
        self.path   = path
        self.flags  = flags
        self.mask   = None
        self._shape = None
    
    @property
    def path(self) -> core.Path:
        """Returns the image file path.

        Returns:
            ``core.Path`` representing the image file path.
        """
        return self._path
    
    @path.setter
    def path(self, path: core.Path | str | None):
        """Sets the image file path.

        Args:
            path: Path to the image file as a ``core.Path`` or ``str`` or ``None``.

        Raises:
            ValueError: If ``[path]`` is not a valid image file path or is ``None``.
        """
        if path is None or not core.Path(path).is_image_file():
            raise ValueError(f"[path] must be a valid image path, but got [{path}].")
        self._path  = core.Path(path)
        self._shape = vision.read_image_shape(path=self._path)
    
    @property
    def name(self) -> str:
        """Returns the image file name.

        Returns:
            ``str`` representing the image file name.
        """
        return self.path.name
    
    @property
    def stem(self) -> str:
        """Returns the stem of the image file path.

        Returns:
            ``str`` representing the stem of the image file path.
        """
        return self.path.stem
    
    @property
    def shape(self) -> tuple[int, int, int]:
        """Returns the mask shape.

        Returns:
            Tuple of ``[H, W, C]`` representing the mask dimensions.
        """
        return self._shape
    
    @property
    def data(self) -> np.ndarray | None:
        """Returns the mask data.

        Loads the mask if not already loaded, otherwise returns cached data.

        Returns:
            ``numpy.ndarray`` of mask data or ``None`` if not loaded.
        """
        return self.mask if self.mask is not None else self.load()
    
    @property
    def meta(self) -> dict:
        """Returns metadata about the mask.

        Returns:
            Dict with keys ``name``, ``stem``, ``path``, ``shape``, and ``hash``.
        """
        return {
            "name" : self.name,
            "stem" : self.stem,
            "path" : self.path,
            "shape": self.shape,
            "hash" : self.path.stat().st_size if isinstance(self.path, core.Path) else None,
        }
    
    def load(
        self,
        path : core.Path | str = None,
        flags: int             = None,
        cache: bool            = False,
    ) -> np.ndarray | None:
        """Loads the mask into memory.

        Args:
            path: Path to the image file as a ``core.Path`` or ``str``. Default is ``None``.
            flags: Flag to read the image. Default is ``None``.
            cache: If ``True``, caches the mask in memory. Default is ``False``.

        Returns:
            ``numpy.ndarray`` in ``[H, W, C]`` format with values in ``[0, 255]`` or ``None``.
        """
        if self.mask is not None:
            return self.mask
        load_path  = path or self.path
        load_flags = flags or self.flags
        mask       = vision.read_image(
            path      = load_path,
            flags     = load_flags,
            to_tensor = False,
            normalize = False
        )
        self.mask = mask if cache else None
        return mask
    
    @staticmethod
    def to_tensor(
        data     : torch.Tensor | np.ndarray,
        keepdim  : bool = False,
        normalize: bool = True
    ) -> torch.Tensor:
        """Converts input data to a tensor.

        Args:
            data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``.
            keepdim: If ``True``, retains input dimensions. Default is ``False``.
            normalize: If ``True``, normalizes the data. Default is ``True``.

        Returns:
            ``torch.Tensor`` of the converted data.
        """
        return vision.to_image_tensor(data, keepdim, normalize)
    
    @staticmethod
    def collate_fn(batch: list[torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray | None:
        """Collates batch data for ``torch.utils.data.DataLoader``.

        Args:
            batch: List of masks as ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            Collated ``torch.Tensor``, ``numpy.ndarray``, or ``None`` if batch is empty or invalid.
        """
        if not batch:
            return None
        return vision.to_4d_image(batch)
    
# endregion
