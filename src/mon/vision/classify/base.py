#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base class for classification models."""

from __future__ import annotations

__all__ = [
    "ImageClassificationModel",
]

from abc import ABC

from mon import core, nn
from mon.globals import Task
from mon.vision.model import VisionModel

console = core.console


# region Model

class ImageClassificationModel(VisionModel, ABC):
    """Base class for image classification models."""

    tasks: list[Task] = [Task.CLASSIFY]

    def parse_num_classes(self, num_classes: int) -> int:
        """Updates num_classes from pretrained weights if needed.

        Args:
            num_classes: Initial number of classes.
        
        Returns:
            Updated number of classes.
        """
        if isinstance(self.weights, dict):
            num_classes_ = self.weights.get("num_classes", None)
            if num_classes_ and num_classes_ != num_classes:
                num_classes = num_classes_
                console.log(f"Overriding num_classes from {num_classes} to {num_classes_}")
        return num_classes

    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        """Computes forward pass and loss.

        Args:
            datapoint: Dict with datapoint attributes.
        
        Returns:
            Dict with predictions and loss.
        """
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        pred    = outputs["logits"]
        target  = datapoint["class_id"]
        outputs["loss"] = self.loss(pred, target) if self.loss else None
        return outputs

    def compute_metrics(
        self,
        datapoint: dict,
        outputs  : dict,
        metrics  : list[nn.Metric] = None
    ) -> dict:
        """Computes metrics for predictions.

        Args:
            datapoint: Dict with datapoint attributes.
            outputs: Dict with model predictions.
            metrics: List of metric functions or ``None``. Default is ``None``.
        
        Returns:
            Dict of computed metric values.
        """
        pred    = outputs["logits"]
        target  = datapoint["class_id"]
        results = {}
        if metrics:
            for i, metric in enumerate(metrics):
                metric_name = getattr(metric, "name", f"metric_{i}")
                results[metric_name] = metric(pred, target)
        return results
    
# endregion
