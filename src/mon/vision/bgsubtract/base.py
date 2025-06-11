#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base class and utility functions for background substraction models.
"""

__all__ = [
    "BackgroundSubtractionModel",
]

import abc

import cv2

from mon import nn
from mon.constants import SAVE_IMAGE_EXT
from mon.vision import model, types


# ----- Background Subtraction Model -----
class BackgroundSubtractionModel(model.VisionModel, abc.ABC):
    """The base class for all background substraction models."""
    
    # ----- Forward Pass -----
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        """Computes forward pass and loss.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"loss"`` and ``"enhanced"`` keys.
        """
        # Forward
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        
        # Loss
        pred   = outputs["background"]
        target = datapoint["ref_background"]
        loss   = self.loss(pred, target)
        
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
        pred    = outputs["background"]
        target  = datapoint["ref_background"]
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

        image   = data.get("image",         None)
        ref_bg  = data.get("ref_bg",        None)
        ref_fg  = data.get("ref_fg",        None)
        outputs = data.get("outputs",       {})
        bg      = outputs.pop("background", None)
        fg      = outputs.pop("foreground", None)
        
        image   = list(types.image_to_array(image,  denormalize=True))
        ref_bg  = list(types.image_to_array(ref_bg, denormalize=True)) if ref_bg else None
        ref_fg  = list(types.image_to_array(ref_fg, denormalize=True)) if ref_fg else None
        bg      = list(types.image_to_array(bg,     denormalize=True))
        fg      = list(types.image_to_array(fg,     denormalize=True)) if fg else None
        extra_images = {k: v for k, v in outputs.items() if types.is_image(v)}
        extra        = {
            k: list(types.image_to_array(v, denormalize=True))
            for k, v in extra_images.items()
        } if extra_images else {}
        
        if len(image) != len(bg):
            raise ValueError(f"[image] and [bg] counts must match, "
                             f"got {len(image)} != {len(bg)}.")
        if ref_bg:
            if len(image) != len(ref_bg):
                raise ValueError(f"[image] and [ref_bg] counts must match, "
                                 f"got {len(image)}] != [{len(ref_bg)}.")
            
        for i in range(len(image)):
            if ref_bg:
                combined = cv2.hconcat([image[i], bg[i], ref_bg[i]])
            else:
                combined = cv2.hconcat([image[i], bg[i]])
            combined    = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            output_path = save_dir / f"{i}{extension}"
            cv2.imwrite(str(output_path), combined)
            
            for k, v in extra.items():
                v_i = v[i]
                extra_path = save_dir / f"{i}_{k}{extension}"
                cv2.imwrite(str(extra_path), v_i)
