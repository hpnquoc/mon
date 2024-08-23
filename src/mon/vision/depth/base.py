#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Depth Estimation Model.

This module implements the base class for depth estimation models.
"""

from __future__ import annotations

__all__ = [
    "DepthEstimationModel",
]

from abc import ABC

from mon import core, nn
from mon.globals import Scheme, Task, ZOO_DIR
from mon.vision.model import VisionModel

console = core.console


# region Model

class DepthEstimationModel(VisionModel, ABC):
    """The base class for all depth estimation models."""
    
    tasks: list[Task] = [Task.DEPTH]
    
    # region Forward Pass
    
    def assert_datapoint(self, datapoint: dict) -> bool:
        if "image" not in datapoint:
            raise ValueError("The key ``'image'`` must be defined in the "
                             "`datapoint`.")
        
        has_target = any(item in self.schemes for item in [Scheme.SUPERVISED])
        if has_target:
            if "depth" not in datapoint:
                raise ValueError("The key ``'depth'`` must be defined in the "
                                 "`datapoint`.")
            
    def assert_outputs(self, outputs: dict) -> bool:
        if "depth" not in outputs:
            raise ValueError("The key ``'depth'`` must be defined in the "
                             "`outputs`.")
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        pred    = outputs.get("depth")
        target  = datapoint.get("depth")
        outputs["loss"] = self.loss(pred, target) if self.loss else None
        # Return
        return outputs
    
    def compute_metrics(
        self,
        datapoint: dict,
        outputs  : dict,
        metrics  : list[nn.Metric] = None
    ) -> dict:
        # Check
        self.assert_datapoint(datapoint)
        self.assert_outputs(outputs)
        # Metrics
        pred    = outputs.get("depth")
        target  = datapoint.get("depth")
        results = {}
        if metrics is not None:
            for i, metric in enumerate(metrics):
                metric_name = getattr(metric, "name", f"metric_{i}")
                results[metric_name] = metric(pred, target)
        # Return
        return results
    
    # endregion
    
# endregion
