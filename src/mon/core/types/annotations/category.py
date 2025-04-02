#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements category/class annotations."""

__all__ = [
    "ClassificationAnnotation",
    "convert_class_id_to_logits",
    "convert_logits_to_class_id",
]

import numpy as np
import torch

from mon.core.types.annotations import base


# ----- Processing -----
def convert_logits_to_class_id(logits: np.ndarray) -> np.ndarray:
    """Converts logits to class IDs.

    Args:
        logits: ``numpy.ndarray`` of logits, shape [N, C] (samples, classes).

    Returns:
        ``numpy.ndarray`` of class IDs, shape [N], with highest logit per sample.
    """
    return np.argmax(logits, axis=-1)


def convert_class_id_to_logits(
    class_id   : int,
    num_classes: int,
    high_value : float = 1.0,
    low_value  : float = 0.0
) -> np.ndarray:
    """Converts a class ID to logits.

    Args:
        class_id: Integer class ID to target.
        num_classes: Total number of classes.
        high_value: Logit for target class. Default is ``1.0``.
        low_value: Logit for non-target classes. Default is ``0.0``.

    Returns:
        ``numpy.ndarray`` of logits, shape [num_classes].
    """
    logits = np.full(num_classes, low_value, dtype=np.float32)
    logits[class_id] = high_value
    return logits


# ----- Annotation -----
class ClassificationAnnotation(base.Annotation):
    """Classification annotation for an image.
    
    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``values``.
        
    Args:
        class_id: Integer class ID, ``-1`` for unknown.
        num_classes: Total number of classes in task.
        confidence: Confidence score in [0.0, 1.0]. Default is ``1.0``.
    """
    
    albumentation_target_type: str = "values"
    
    def __init__(
        self,
        class_id   : int,
        num_classes: int,
        confidence : float = 1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_id    = class_id
        self.num_classes = num_classes
        self.confidence  = confidence
        self.logits      = convert_class_id_to_logits(class_id, num_classes)
    
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
            raise ValueError(f"[confidence] must be in [0.0, 1.0],  got {confidence}.")
        self._confidence = confidence
    
    @property
    def data(self) -> list[int]:
        """Returns the class ID as a list.

        Returns:
            List containing ``class_id``.
        """
        return [self.class_id]
    
    @staticmethod
    def to_tensor(data: torch.Tensor | np.ndarray, *args, **kwargs) -> torch.Tensor:
        """Converts input data to a tensor.

        Args:
            data: Input as ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            ``torch.Tensor`` of input data.
        """
        return torch.as_tensor(data)
    
    @staticmethod
    def collate_fn(batch: list[torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray | None:
        """Collates batch data for ``torch.utils.data.DataLoader``.

        Args:
            batch: List of class IDs as ``torch.Tensor`` or ``numpy.ndarray``.

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
