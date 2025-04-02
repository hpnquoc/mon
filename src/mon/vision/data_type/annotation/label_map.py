#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements basic functionalities for image data."""

__all__ = [
    "SemanticSegmentationAnnotation",
    "convert_label_map_color_to_id",
    "convert_label_map_id_to_color",
    "convert_label_map_id_to_one_hot",
    "convert_label_map_id_to_train_id",
    "convert_label_map_one_hot_to_id",
]

import cv2
import numpy as np
import torch

from mon import core
from mon.nn import functional as F
from mon.vision.data_type import image as I


# region Conversion

def convert_label_map_id_to_train_id(label_map: np.ndarray, classlabels: core.ClassLabels) -> np.ndarray:
    """Converts label map from IDs to train IDs.

    Args:
        label_map: Label map as ``numpy.ndarray`` in [H, W] or [H, W, 1] format.
        classlabels: ``ClassLabels`` object mapping IDs to train IDs.
    
    Returns:
        Converted label map as numpy.ndarray in [H, W, 1] format.
    
    Raises:
        TypeError: If ``label_map`` is not a ``numpy.ndarray``.
    """
    if not isinstance(label_map, np.ndarray):
        raise TypeError(f"[label_map] must be a numpy.ndarray, got {type(label_map)}.")
    
    id2train_id = classlabels.id_to_train_id
    h, w        = I.get_image_size(label_map)
    label_ids   = np.zeros((h, w), dtype=np.uint8)
    label_map   = I.convert_image_to_2d(label_map)
    
    for id, train_id in id2train_id.items():
        label_ids[label_map == id] = train_id
    
    return np.expand_dims(label_ids, axis=-1)
 

def convert_label_map_id_to_color(label_map: np.ndarray, classlabels: core.ClassLabels) -> np.ndarray:
    """Converts label map from IDs to color-coded representation.

    Args:
        label_map: Label map as ``numpy.ndarray`` in [H, W] or [H, W, 1] format.
        classlabels: ``ClassLabels`` object mapping IDs to colors.
    
    Returns:
        Color-coded label map as ``numpy.ndarray`` in [H, W, 3] format.
    
    Raises:
        TypeError: If ``label_map`` is not a ``numpy.ndarray``.
    """
    if not isinstance(label_map, np.ndarray):
        raise TypeError(f"[label_map] must be a numpy.ndarray, got {type(label_map)}.")

    id2color  = classlabels.id_color
    h, w      = I.get_image_size(label_map)
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    label_map = I.convert_image_to_2d(label_map)
    for id, color in id2color.items():
        color_map[label_map == id] = color
    return color_map


def convert_label_map_color_to_id(label_map: np.ndarray, classlabels: core.ClassLabels) -> np.ndarray:
    """Converts a color-coded label map to label IDs.

    Args:
        label_map: Color-coded label map as ``numpy.ndarray`` in [H, W, C] format.
        classlabels: ``ClassLabels`` object mapping colors to IDs.
    
    Returns:
        Label map with IDs as ``numpy.ndarray`` in [H, W, 1] format.
    
    Raises:
        TypeError: If ``label_map`` is not a ``numpy.ndarray``.
    """
    if not isinstance(label_map, np.ndarray):
        raise TypeError(f"[label_map] must be a numpy.ndarray, got {type(label_map)}.")
    
    id2color  = classlabels.id_color
    h, w      = I.get_image_size(label_map)
    label_ids = np.zeros((h, w), dtype=np.uint8)
    for id, color in id2color.items():
        label_ids[np.all(label_map == color, axis=-1)] = id
    label_ids = np.expand_dims(label_ids, axis=-1)
    return label_ids


def convert_label_map_id_to_one_hot(
    label_map  : torch.Tensor | np.ndarray,
    num_classes: int              = None,
    classlabels: core.ClassLabels = None,
) ->torch.Tensor | np.ndarray:
    """Converts label map from IDs to one-hot encoded format.

    Args:
        label_map: IDs label map as ``torch.Tensor`` [B, 1, H, W] or
            ``numpy.ndarray`` [H, W, 1].
        num_classes: Number of classes in the label map, optional.
        classlabels: ``ClassLabels`` object with class info, optional.
    
    Returns:
        One-hot encoded label map as ``torch.Tensor`` [B, C, H, W] or
        ``numpy.ndarray`` [H, W, C].
    
    Raises:
        ValueError: If neither ``num_classes`` nor ``classlabels`` is provided.
        TypeError: If ``label_map`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if num_classes is None and classlabels is None:
        raise ValueError("Either [num_classes] or [classlabels] must be provided.")

    num_classes = num_classes or classlabels.num_trainable_classes
    if isinstance(label_map, torch.Tensor):
        label_map = I.convert_image_to_3d(label_map).long()
        one_hot   = F.one_hot(label_map, num_classes)
        return I.convert_image_to_channel_first(one_hot).contiguous()
    elif isinstance(label_map, np.ndarray):
        label_map = I.convert_image_to_2d(label_map)
        return np.eye(num_classes)[label_map]
    else:
        raise TypeError(f"[label_map] must be a torch.Tensor or numpy.ndarray, got {type(label_map)}.")


def convert_label_map_one_hot_to_id(label_map: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Converts a one-hot encoded label map to label IDs.

    Args:
        label_map: One-hot encoded label map as ``torch.Tensor`` [B, C, H, W] or
            ``numpy.ndarray`` [H, W, C].
    
    Returns:
        Label map with IDs as ``torch.Tensor`` [B, 1, H, W] or
        ``numpy.ndarray`` [H, W, 1].
    
    Raises:
        TypeError: If ``label_map`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if isinstance(label_map, torch.Tensor):
        label_map = torch.argmax(label_map, dim=-1, keepdim=True)
    elif isinstance(label_map, np.ndarray):
        label_map = np.argmax(label_map, axis=-1, keepdims=True)
    else:
        raise TypeError(f"[label_map] must be a torch.Tensor or numpy.ndarray, got {type(label_map)}.")
    return label_map

# endregion


# region Annotation

class SemanticSegmentationAnnotation(core.Annotation):
    """Semantic segmentation annotation (mask).

    Args:
        path: Path to image file as ``core.Path`` or ``str``.
        root: Root dir as ``core.Path`` or ``str``. Default is ``None``.
        flags: Flag to read image (e.g., ``cv2.IMREAD_COLOR``). Default is ``cv2.IMREAD_COLOR``.
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
            ``core.Path`` of the image file path.
        """
        return self._path
    
    @path.setter
    def path(self, path: core.Path | str | None):
        """Sets the image file path.

        Args:
            path: Path to image file or ``None``.

        Raises:
            ValueError: If ``path`` is not a valid image path or is ``None``.
        """
        if path is None or not core.Path(path).is_image_file():
            raise ValueError(f"[path] must be a valid image path, got {path}.")
        self._path  = core.Path(path)
        self._shape = I.read_image_shape(path=self._path)
    
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
    def shape(self) -> tuple[int, int, int]:
        """Returns the mask shape.

        Returns:
            Tuple of [H, W, C] for mask dimensions.
        """
        return self._shape
    
    @property
    def data(self) -> np.ndarray | None:
        """Returns the mask data.

        Returns:
            ``numpy.ndarray`` of mask data or ``None`` if not loaded.
        """
        return self.mask if self.mask is not None else self.load()
    
    @property
    def meta(self) -> dict:
        """Returns metadata about the mask.

        Returns:
            Dict with ``name``, ``stem``, ``path``, ``shape``, and ``hash``.
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
            path: Path to image file. Default is ``None``.
            flags: Flag to read image. Default is ``None``.
            cache: If ``True``, caches mask. Default is ``False``.

        Returns:
            ``numpy.ndarray`` in [H, W, C], values in [0, 255], or ``None``.
        """
        if self.mask is not None:
            return self.mask
        load_path  = path  or self.path
        load_flags = flags or self.flags
        mask       = I.read_image(load_path, load_flags, False, False)
        self.mask = mask if cache else None
        return mask
    
    @staticmethod
    def to_tensor(data: torch.Tensor | np.ndarray, normalize: bool = True) -> torch.Tensor:
        """Converts input data to a tensor.

        Args:
            data: Input as ``torch.Tensor`` or ``numpy.ndarray``.
            normalize: If ``True``, normalizes data. Default is ``True``.

        Returns:
            ``torch.Tensor`` of converted data.
        """
        return I.convert_image_to_tensor(data, normalize)
    
    @staticmethod
    def collate_fn(batch: list[torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray | None:
        """Collates batch data for ``torch.utils.data.DataLoader``.

        Args:
            batch: List of masks as ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            Collated ``torch.Tensor``, ``numpy.ndarray``, or ``None`` if empty/invalid.
        """
        if not batch:
            return None
        return I.convert_image_to_4d(batch)

# endregion
