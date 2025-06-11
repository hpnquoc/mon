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
    def data(self) -> Any:
        """Returns the annotation's data."""
        pass

    # ----- DataLoader Interface -----
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
