#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base class and utility functions for all annotations.

An annotation refers to metadata or labels associated with data that provide
context, meaning, or ground truth for training, evaluation, or interpretation.

They describe specific aspects of the visual data, such as object locations, categories,
or semantic regions, and are typically created manually or semi-automatically.
"""

__all__ = [
    "Annotation",
]

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


# ----- Annotation -----
class Annotation(ABC):
    """Base class for annotation classes, representing task-specific data.
    
    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``None``.
    """
    
    albumentation_target_type: str = None
    
    @property
    @abstractmethod
    def data(self) -> list | None:
        """Returns the annotation's data.

        Returns:
            List of annotation data or ``None`` if unavailable.
        """
        pass
    
    @property
    def nparray(self) -> np.ndarray | None:
        """Returns annotation data as a NumPy array.

        Returns:
            ``numpy.ndarray`` of numeric data or original ``data`` if not convertible.
        """
        return np.asarray([x for x in self.data if isinstance(x, (int, float))], dtype=np.float32) \
            if isinstance(self.data, list) else self.data
    
    @property
    def tensor(self) -> torch.Tensor | None:
        """Returns annotation data as a PyTorch tensor.

        Returns:
            ``torch.Tensor`` of numeric data or original ``data`` if not convertible.
        """
        return torch.as_tensor([x for x in self.data if isinstance(x, (int, float))]) \
            if isinstance(self.data, list) else self.data
    
    @staticmethod
    @abstractmethod
    def to_tensor(data: torch.Tensor | np.ndarray, *args, **kwargs) -> torch.Tensor:
        """Converts input data to a tensor.

        Args:
            data: Input as ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            ``torch.Tensor`` of converted data.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def collate_fn(batch: list[Any]) -> Any:
        """Collates batch data for ``torch.utils.data.DataLoader``.

        Args:
            batch: List of annotation objects.

        Returns:
            Collated data in suitable format.
        """
        pass
