#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Image Enhancement Model.

This module implements the base class for enhancement models.
"""

from __future__ import annotations

__all__ = [
    "ImageEnhancementModel",
]

from abc import ABC

import cv2

from mon import core, nn
from mon.vision import dtype
from mon.vision.model import VisionModel

console = core.console


# region Model

class ImageEnhancementModel(VisionModel, ABC):
    """The base class for all image enhancement models."""
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        # Loss
        pred   = outputs["enhanced"]
        target = datapoint["ref_image"]
        outputs["loss"] = self.loss(pred, target)
        # Return
        return outputs
    
    def compute_metrics(
        self,
        datapoint: dict,
        outputs  : dict,
        metrics  : list[nn.Metric] = None
    ) -> dict:
        # Metrics
        pred    = outputs["enhanced"]
        target  = datapoint["ref_image"]
        results = {}
        if metrics is not None:
            for i, metric in enumerate(metrics):
                metric_name = getattr(metric, "name", f"metric_{i}")
                results[metric_name] = metric(pred, target)
        # Return
        return results
        
    def log_images(
        self,
        epoch    : int,
        step     : int,
        data     : dict,
        extension: str = ".jpg"
    ):
        epoch    = int(epoch)
        step     = int(step)
        save_dir = self.debug_dir / f"epoch_{epoch:04d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        image     =    data.get("image",    None)
        ref_image =    data.get("ref_image", None)
        outputs   =    data.get("outputs",  {})
        enhanced  = outputs.pop("enhanced", None)
        
        image        = list(dtype.convert_image_to_array(image, denormalize=True))
        ref_image    = list(dtype.convert_image_to_array(ref_image, denormalize=True)) if ref_image is not None else None
        enhanced     = list(dtype.convert_image_to_array(enhanced, denormalize=True))
        extra_images = {k: v for k, v in outputs.items() if dtype.is_image(v)}
        extra        = {
            k: list(dtype.convert_image_to_array(v, denormalize=True))
            for k, v in extra_images.items()
        } if extra_images else {}
        
        if len(image) != len(enhanced):
            raise ValueError(f"The number of `image` and `enhanced` must be "
                             f"the same, but got {len(image)} != {len(enhanced)}.")
        if ref_image is not None:
            if len(image) != len(ref_image):
                raise ValueError(f"The number of `image` and `ref_image` must "
                                 f"be the same, but got {len(image)} != {len(ref_image)}.")
            
        for i in range(len(image)):
            if ref_image:
                combined = cv2.hconcat([image[i], enhanced[i], ref_image[i]])
            else:
                combined = cv2.hconcat([image[i], enhanced[i]])
            combined    = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            output_path = save_dir / f"{i}{extension}"
            cv2.imwrite(str(output_path), combined)
            
            for k, v in extra.items():
                v_i = v[i]
                extra_path = save_dir / f"{i}_{k}{extension}"
                cv2.imwrite(str(extra_path), v_i)
            
# endregion
