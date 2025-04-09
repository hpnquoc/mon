#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base class and utility functions for segmentation models."""

__all__ = [
    "SegmentationModel",
]

from abc import ABC

import cv2

from mon import nn
from mon.constants import SAVE_IMAGE_EXT, Task
from mon.vision import model, types


# ----- Segmentation Model -----
class SegmentationModel(model.VisionModel, ABC):
    """The base class for all segmentation models."""
    
    tasks: list[Task] = [Task.SEGMENT]
    
    # ----- Forward Pass -----
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        """Computes forward pass and loss.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"loss"`` and ``"semantic"`` keys.
        """
        # Forward
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        
        # Loss
        pred    = outputs["semantic"]
        target  = datapoint["semantic"]
        loss    = self.loss(pred, target)
        
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
        pred    = outputs["semantic"]
        target  = datapoint["semantic"]
        results = {}
        if metrics:
            for i, metric in enumerate(metrics):
                metric_name = getattr(metric, "name", f"metric_{i}")
                results[metric_name] = metric(pred, target)
        return results
    
    # ----- Log -----
    def log_images(self, epoch: int, step: int, data: dict, extension: str = SAVE_IMAGE_EXT):
        """Logs debug images to ``debug_dir``.
    
        Args:
            epoch: Current epoch number.
            step: Current step number.
            data: Dict with images to log.
            extension: Image file extension. Default is ``SAVE_IMAGE_EXT``.
        """
        epoch    = int(epoch)
        step     = int(step)
        save_dir = self.debug_dir / f"epoch_{epoch:04d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        image         =    data.get("image",    None)
        tar_semantic  =    data.get("semantic", None)
        outputs       =    data.get("outputs",  {})
        pred_semantic = outputs.pop("semantic", None)
        
        image         = list(types.convert_image_to_array(image, denormalize=True))
        tar_semantic  = list(types.convert_image_to_array(tar_semantic, denormalize=True)) if tar_semantic else None
        pred_semantic = list(types.convert_image_to_array(pred_semantic, denormalize=True))
        extra_images  = {k: v for k, v in outputs.items() if types.is_image(v)}
        extra         = {
            k: list(types.convert_image_to_array(v, denormalize=True))
            for k, v in extra_images.items()
        } if extra_images else {}
        
        if len(image) != len(pred_semantic):
            raise ValueError(f"[image] and [pred_semantic] counts must match, "
                             f"got {len(image)} != {len(pred_semantic)}.")
        if tar_semantic:
            if len(image) != len(tar_semantic):
                raise ValueError(f"[image] and [tar_semantic] counts must match, "
                                 f"got {len(image)} != {len(tar_semantic)}.")
        
        for i in range(len(image)):
            if tar_semantic:
                combined = cv2.hconcat([image[i], pred_semantic[i], tar_semantic[i]])
            else:
                combined = cv2.hconcat([image[i], pred_semantic[i]])
            combined    = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            output_path = save_dir / f"{i}{extension}"
            cv2.imwrite(str(output_path), combined)
            
            for k, v in extra.items():
                v_i = v[i]
                extra_path = save_dir / f"{i}_{k}{extension}"
                cv2.imwrite(str(extra_path), v_i)
