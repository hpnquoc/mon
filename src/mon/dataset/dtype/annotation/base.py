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
    """The base class for all annotation classes. An annotation instance
    represents a logical collection of data associated with a particular task.
    """
    
    @property
    @abstractmethod
    def data(self) -> list | None:
        """The annotation's data."""
        pass
    
    @property
    def nparray(self) -> np.ndarray | None:
        """The annotation's data as a ``numpy.ndarray``.
    
        This property converts the annotation's data to a ``numpy.ndarray`` if
        the data is a list containing integers or floats. If the data is not a
        list, it returns the data as is.
    
        Returns:
            The annotation's data as a ``numpy.ndarray`` if the data is a list,
            otherwise returns the data as is.
        """
        if isinstance(self.data, list):
            return np.array([i for i in self.data if isinstance(i, (int, float))])
        return self.data
    
    @property
    def tensor(self) -> torch.Tensor | None:
        """The annotation's data as a ``torch.Tensor``.
    
        This property converts the annotation's data to a ``torch.Tensor`` if
        the data is a list containing integers or floats. If the data is not a
        list, it returns the data as is.
    
        Returns:
            The annotation's data as a ``torch.Tensor`` if the data is a list,
            otherwise returns the data as is.
        """
        if isinstance(self.data, list):
            return torch.Tensor([i for i in self.data if isinstance(i, (int, float))])
        return self.data
    
    @staticmethod
    @abstractmethod
    def to_tensor(data: torch.Tensor | np.ndarray, *args, **kwargs) -> torch.Tensor:
        """Converts the input data to a ``torch.Tensor``.
        
        Args:
            data: The input data.
        
        Returns:
            The converted ``torch.Tensor``.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def collate_fn(batch: list[Any]) -> Any:
        """Collate function used to fused input items together when using
		``batch_size`` > 1. This is used in ``torch.utils.data.DataLoader``
		wrapper.
		
		Args:
			batch: A ``list`` of objects.
		"""
        pass
    
# endregion
