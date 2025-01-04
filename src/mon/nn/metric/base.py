#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Metric Module.

This module implements the base classes for all metrics, and the corresponding
helper functions.
"""

from __future__ import annotations

__all__ = [
    "BootStrapper",
    "CatMetric",
    "ClasswiseWrapper",
    "MaxMetric",
    "MeanMetric",
    "Metric",
    "MetricCollection",
    "MetricTracker",
    "MinMaxMetric",
    "MinMetric",
    "MultioutputWrapper",
    "MultitaskWrapper",
    "RunningMean",
    "RunningSum",
    "SumMetric",
    "scale_gt_mean",
]

from abc import ABC
from typing import Literal

import cv2
import numpy as np
import torch
import torchmetrics


# region Base

class Metric(torchmetrics.Metric, ABC):
    """The base class for all loss functions.

    Args:
        mode: One of: ``'FR'`` or ``'NR'``. Default: ``'FR'``.
        lower_is_better: Default: ``False``.
    """
    
    def __init__(
        self,
        mode           : Literal["FR", "NR"] = "FR",
        lower_is_better: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mode            = mode
        self.lower_is_better = lower_is_better

# endregion


# region Aggregation

MetricCollection = torchmetrics.MetricCollection

CatMetric   = torchmetrics.CatMetric
MaxMetric   = torchmetrics.MaxMetric
MeanMetric  = torchmetrics.MeanMetric
MinMetric   = torchmetrics.MinMetric
RunningMean = torchmetrics.RunningMean
RunningSum  = torchmetrics.RunningSum
SumMetric   = torchmetrics.SumMetric

# endregion


# region Wrapper

BootStrapper       = torchmetrics.BootStrapper
ClasswiseWrapper   = torchmetrics.ClasswiseWrapper
MetricTracker      = torchmetrics.MetricTracker
MinMaxMetric       = torchmetrics.MinMaxMetric
MultioutputWrapper = torchmetrics.MultioutputWrapper
MultitaskWrapper   = torchmetrics.MultitaskWrapper

# endregion


# region Scale

def scale_gt_mean(
    image : torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray
) -> torch.Tensor | np.ndarray:
    """Scale the image to match the mean of the target image.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        target: The target image of the same type as `image`.
    
    References:
        https://github.com/Fediory/HVI-CIDNet/blob/master/measure.py
    """
    from mon.core.image import color_space
    
    if isinstance(image, torch.Tensor) and isinstance(target, torch.Tensor):
        mean_image  = color.rgb_to_grayscale(image).mean()
        mean_target = color.rgb_to_grayscale(target).mean()
        image       = torch.clip(image * (mean_target / mean_image), 0, 1)
    elif isinstance(image, np.ndarray) and isinstance(target, np.ndarray):
        mean_restored = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean()
        mean_target   = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean()
        image         = np.clip(image * (mean_target/mean_restored), 0, 255)
    else:
        raise TypeError(f"Both `image` and `target` must be of the same type, "
                        f"but got {type(image)} and {type(target)}.")
    return image
    
# endregion
