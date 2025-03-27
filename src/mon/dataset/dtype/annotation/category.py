#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Category Annotation.

This module implements annotations that take the form of a category or class.
"""

from __future__ import annotations

__all__ = [
    "ClassificationAnnotation",
    "class_id_to_logits",
    "logits_to_class_id",
]

import numpy as np
import torch

from mon.dataset.dtype.annotation import base


# region Utils

def logits_to_class_id(logits: np.ndarray) -> np.ndarray:
    """Converts logits to class IDs.

    Args:
        logits: ``numpy.ndarray`` of logits with shape ``[N, C]`` where ``N`` is samples and ``C`` is classes.

    Returns:
        ``numpy.ndarray`` of class IDs with shape [N], selecting the highest logit per sample.
    """
    return np.argmax(logits, axis=-1)


def class_id_to_logits(
    class_id   : int,
    num_classes: int,
    high_value : float = 1.0,
    low_value  : float = 0.0
) -> np.ndarray:
    """Converts a class ID to logits.

    Args:
        class_id: Integer class ID to set as the target.
        num_classes: Total number of classes.
        high_value: Logit value for the target class. Default is ``1.0``.
        low_value: Logit value for non-target classes. Default is ``0.0``.

    Returns:
        ``numpy.ndarray`` of logits with shape ``[num_classes]``.
    """
    logits = np.full(num_classes, low_value, dtype=np.float32)
    logits[class_id] = high_value
    return logits

# endregion


# region Classification

class ClassificationAnnotation(base.Annotation):
    """Classification annotation for an image.

    Args:
        class_id: Integer class ID, where ``-1`` indicates unknown.
        num_classes: Total number of classes in the task.
        confidence: Confidence score in [0.0, 1.0]. Default is ``1.0``.
    """
    
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
        self.logits      = class_id_to_logits(class_id, num_classes)
    
    @property
    def confidence(self) -> float:
        """Returns the confidence score.

        Returns:
            ``float`` representing the confidence in [0.0, 1.0].
        """
        return self._confidence
    
    @confidence.setter
    def confidence(self, confidence: float):
        """Sets the confidence score.

        Args:
            confidence: Confidence value as a ``float``.

        Raises:
            ValueError: If ``[confidence]`` is not in [0.0, 1.0].
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"[confidence] must be in [0.0, 1.0], but got [{confidence}].")
        self._confidence = confidence
    
    @property
    def data(self) -> list[int]:
        """Returns the class ID as a list.

        Returns:
            List containing the ``class_id``.
        """
        return [self.class_id]
    
    @staticmethod
    def to_tensor(data: torch.Tensor | np.ndarray, *args, **kwargs) -> torch.Tensor:
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
            batch: List of class IDs as ``torch.Tensor`` or ``numpy.ndarray``.

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
    
# endregion
