#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements value-based annotations (number, boolean, etc.)."""

__all__ = [
    "RegressionAnnotation",
]

import numpy as np
import torch

from mon.core.types.annotations import base


# ----- Annotation -----
class RegressionAnnotation(base.Annotation):
    """Single regression value annotation.
    
    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``values``.
        
    Args:
        value: Regression value as ``float``.
        confidence: Confidence score in [0.0, 1.0]. Default is ``1.0``.
    """
    
    albumentation_target_type: str = "values"
    
    def __init__(self, value: float, confidence: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value      = value
        self.confidence = confidence
    
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
    def data(self) -> list[float]:
        """Returns the regression value as a list.

        Returns:
            List containing regression ``value``.
        """
        return [self.value]
    
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
            batch: List of values as ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            Collated ``torch.Tensor``, ``numpy.ndarray``, or ``None`` if mixed.
        """
        if not batch:
            return None
        if isinstance(batch[0], torch.Tensor):
            return torch.stack(batch, dim=0)
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch, axis=0)
        return None
