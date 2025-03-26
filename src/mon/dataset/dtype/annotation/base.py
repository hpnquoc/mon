#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Annotation.

This module implements the base class for all annotations.
"""

from __future__ import annotations

__all__ = [
    "Annotation",
]

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


# region Base

class Annotation(ABC):
    """Base class for all annotation classes, representing task-specific data collections."""
    
    @property
    @abstractmethod
    def data(self) -> list | None:
        """Returns the annotation's data.

        Returns:
            List of annotation data or ``None`` if unavailable.
        """
    
    @property
    def nparray(self) -> np.ndarray | None:
        """Returns the annotation's data as a NumPy array.

        Converts ``data`` to a ``numpy.ndarray`` if it’s a list of integers or floats; otherwise, returns ``data`` as is.

        Returns:
            ``numpy.ndarray`` of numeric data or original ``data`` if not convertible.
        """
        return np.asarray([x for x in self.data if isinstance(x, (int, float))], dtype=np.float32) if isinstance(self.data, list) else self.data
    
    @property
    def tensor(self) -> torch.Tensor | None:
        """Returns the annotation's data as a PyTorch tensor.

        Converts ``data`` to a ``torch.Tensor`` if it’s a list of integers or floats; otherwise, returns ``data`` as is.

        Returns:
            ``torch.Tensor`` of numeric data or original ``data`` if not convertible.
        """
        return torch.as_tensor([x for x in self.data if isinstance(x, (int, float))]) if isinstance(self.data, list) else self.data
    
    @staticmethod
    @abstractmethod
    def to_tensor(data: torch.Tensor | np.ndarray, *args, **kwargs) -> torch.Tensor:
        """Converts input data to a tensor.

        Args:
            data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            ``torch.Tensor`` of the converted data.
        """
    
    @staticmethod
    @abstractmethod
    def collate_fn(batch: list[Any]) -> Any:
        """Collates batch data for ``torch.utils.data.DataLoader``.

        Args:
            batch: List of annotation objects.

        Returns:
            Collated data in a suitable format.
        """

# endregion
