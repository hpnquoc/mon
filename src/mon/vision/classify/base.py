#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base class for classification models."""

__all__ = [
    "ImageClassificationModel",
]

from abc import ABC

from mon import core, nn
from mon.constants import Task
from mon.vision import model


# ----- Classification Model -----
class ImageClassificationModel(model.VisionModel, ABC):
    """Base class for image classification models."""

    tasks: list[Task] = [Task.CLASSIFY]
    
    # ----- Initialize -----
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
                core.console.log(f"Overriding num_classes from {num_classes} to {num_classes_}")
        return num_classes
    
    # ----- Forward Pass -----
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        """Computes forward pass and loss.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"loss"`` and ``"logits"`` keys.
        """
        # Forward
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        
        # Loss
        pred    = outputs["logits"]
        target  = datapoint["class_id"]
        loss    = self.loss(pred, target) if self.loss else None
        
        return outputs | {
            "loss": loss,
        }

    def compute_metrics(
        self,
        datapoint: dict,
        outputs  : dict,
        metrics  : list[nn.Metric] = None
    ) -> dict:
        """Computes metrics for given predictions.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
            outputs: ``dict`` with model predictions.
            metrics: ``list`` of ``M.Metric`` or ``None``. Default is ``None``.
    
        Returns:
            ``dict`` of computed metric values.
        """
        pred    = outputs["logits"]
        target  = datapoint["class_id"]
        results = {}
        if metrics:
            for i, metric in enumerate(metrics):
                metric_name = getattr(metric, "name", f"metric_{i}")
                results[metric_name] = metric(pred, target)
        return results
