#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Classification Model.

This module implements the base model class for classification models.
"""

from __future__ import annotations

__all__ = [
    "ImageClassificationModel",
]

from abc import ABC

from mon import core, nn
from mon.globals import Scheme, Task
from mon.vision.model import VisionModel

console = core.console


# region Model

class ImageClassificationModel(VisionModel, ABC):
    """The base class for all image classification models."""
    
    tasks: list[Task] = [Task.CLASSIFY]
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        # Loss
        pred    = outputs["logits"]
        target  = datapoint["class_id"]
        outputs["loss"] = self.loss(pred, target) if self.loss else None
        # Return
        return outputs
    
    def compute_metrics(
        self,
        datapoint: dict,
        outputs  : dict,
        metrics  : list[nn.Metric] = None
    ) -> dict:
        # Metrics
        pred    = outputs["logits"]
        target  = datapoint["class_id"]
        results = {}
        if metrics is not None:
            for i, metric in enumerate(metrics):
                metric_name = getattr(metric, "name", f"metric_{i}")
                results[metric_name] = metric(pred, target)
        # Return
        return results
    
# endregion
