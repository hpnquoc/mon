#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base class and utility functions for depth estimation models."""

from __future__ import annotations

__all__ = [
    "DepthEstimationModel",
    "convert_depth_to_color",
]

from abc import ABC

import cv2
import numpy as np

from mon import core, nn
from mon.vision import data_type, model


# region Conversion

def convert_depth_to_color(
    depth    : np.ndarray,
    color_map: int = cv2.COLORMAP_JET,
    use_rgb  : bool = False
) -> np.ndarray:
    """Converts a depth map to a color-coded image.

    Args:
        depth: Depth map as ``numpy.ndarray`` in [H, W, 1] format.
        color_map: Color map for the depth map. Default is ``cv2.COLORMAP_JET``.
        use_rgb: Convert to RGB format if ``True``. Default is ``False``.
    
    Returns:
        Color-coded depth map as ``numpy.ndarray`` in [H, W, 3] format.
    
    Raises:
        TypeError: If ``depth`` is not a ``numpy.ndarray``.
    """
    if not isinstance(depth, np.ndarray):
        raise TypeError(f"[depth] must be a numpy.ndarray, got {type(depth)}.")
    depth = np.uint8(255 * depth) if data_type.is_image_normalized(depth) else depth
    depth = cv2.applyColorMap(depth, color_map)
    return cv2.cvtColor(depth, cv2.COLOR_BGR2RGB) if use_rgb else depth
    
# endregion


# region Model

class DepthEstimationModel(model.VisionModel, ABC):
    """Base class for depth estimation models."""
    
    tasks: list[core.Task] = [core.Task.DEPTH]
    
    # region Forward
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        """Computes forward pass and loss.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"loss"`` and ``"depth"`` keys.
        """
        # Forward
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        
        # Loss
        pred    = outputs["depth"]
        target  = datapoint["depth"]
        loss    = self.loss(pred, target) if self.loss else None
        
        return outputs | {
			"loss": loss,
		}
    
    def compute_metrics(self, datapoint: dict, outputs: dict, metrics: list[nn.Metric] = None) -> dict:
        """Computes metrics for given predictions.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
            outputs: ``dict`` with model predictions.
            metrics: ``list`` of ``M.Metric`` or ``None``. Default is ``None``.
    
        Returns:
            ``dict`` of computed metric values.
        """
        pred    = outputs["depth"]
        target  = datapoint["depth"]
        results = {}
        if metrics:
            for i, metric in enumerate(metrics):
                metric_name = getattr(metric, "name", f"metric_{i}")
                results[metric_name] = metric(pred, target)
        return results
        
    # endregion
    
# endregion
