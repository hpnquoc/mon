#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Vision Model.

This module implements the base class for all vision models.
"""

from __future__ import annotations

__all__ = [
    "VisionModel",
]

from abc import ABC
from copy import deepcopy
from typing import Sequence

import torch
from fvcore.nn import parameter_count

from mon import core, nn

console = core.console


# region Model

class VisionModel(nn.Model, ABC):
    """Base class for vision models with image/video input."""
    
    # region Initialize Model
    
    def compute_efficiency_score(self, image_size: int | Sequence[int] = 512) -> tuple[float, float]:
        """Computes model efficiency score (FLOPs, params).

        Args:
            image_size: Input size as int or [H, W]. Default is ``512``.
        
        Returns:
            Tuple of (FLOPs, parameter count) as floats.
        """
        from mon.vision.dtype import image as I
        h, w      = I.get_image_size(image_size)
        datapoint = {"image": torch.rand(1, 3, h, w).to(self.device)}
        flops, params = core.custom_profile(deepcopy(self), inputs=datapoint, verbose=False)
        params        = self.params if hasattr(self, "params") and params == 0 else params
        params        = parameter_count(self) if hasattr(self, "params")  else params
        params        = sum(params.values())  if isinstance(params, dict) else params
        return flops, params
        
    # endregion
    
    # region Predicting
    
    def infer(
        self,
        datapoint : dict,
        image_size: int | Sequence[int] = 512,
        resize    : bool = False,
        *args, **kwargs
    ) -> dict:
        """Infers model output with optional processing.

        Args:
            datapoint: Dict with datapoint attributes.
            image_size: Input size as int or [H, W]. Default is ``512``.
            resize: Resize input to image_size if ``True``. Default is ``False``.
        
        Returns:
            Dict of predictions with inference time.
       
        Notes:
            Override for custom pre/post-processing; defaults to forward.
        """
        from mon.vision.dtype import image as I
        from mon.vision import geometry
        
        # Input
        image  = datapoint["image"]
        h0, w0 = I.get_image_size(image)
        for k, v in datapoint.items():
            if I.is_image(v):
                datapoint[k] = geometry.resize(v, image_size if resize else 32 * ((max(h0, w0) + 31) // 32))
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        
        # Infer
        timer = core.Timer()
        timer.tick()
        outputs = self.forward(datapoint, *args, **kwargs)
        timer.tock()
    
        # Post-processing
        for k, v in outputs.items():
            if I.is_image(v):
                h1, w1 = I.get_image_size(v)
                if h1 != h0 or w1 != w0:
                    outputs[k] = geometry.resize(v, (h0, w0))
        
        # Return
        return outputs | {
            "time": timer.avg_time
        }
    
    # endregion
    
# endregion
