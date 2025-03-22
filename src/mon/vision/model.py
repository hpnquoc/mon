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
    """The base class for all vision models, i.e., image or video as the
    primary input.
    """
    
    # region Initialize Model
    
    def compute_efficiency_score(self, image_size: int | Sequence[int] = 512) -> tuple[float, float]:
        """Compute the efficiency score of the model, including FLOPs and number
        of parameters.
        """
        # Define input tensor
        from mon.vision.dtype import image as I
        h, w      = I.get_image_size(image_size)
        datapoint = {"image": torch.rand(1, 3, h, w).to(self.device)}
        # Get FLOPs and Params
        flops, params = core.custom_profile(deepcopy(self), inputs=datapoint, verbose=False)
        params        = self.params                if hasattr(self, "params") and params == 0 else params
        params        = parameter_count(self)      if hasattr(self, "params")  else params
        params        = sum(list(params.values())) if isinstance(params, dict) else params
        # Print
        if self.verbose:
            console.log(f"FLOPs : {flops:.4f}")
            console.log(f"Params: {params:.4f}")
        # Return
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
        """Infer the model on a single datapoint. This method is different from
        `forward()` in term that you may want to perform additional
        pre-processing or post-processing steps.
        
        Notes:
            If you want to perform specific pre-processing or post-processing
            steps, you should override this method.
        
        Args:
            datapoint: A `dict` containing the attributes of a datapoint.
            image_size: The input size. Default: ``512``.
            resize: Resize the input image to the model's input size.
                Default: ``False``.
        """
        from mon.vision.dtype import image as I
        from mon.vision import geometry
        
        # Pre-processing
        image  = datapoint["image"]
        h0, w0 = I.get_image_size(image)
        for k, v in datapoint.items():
            if I.is_image(v):
                if resize:
                    datapoint[k] = geometry.resize(v, image_size)
                else:
                    datapoint[k] = geometry.resize(v, divisible_by=32)
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
                
        # Forward
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
        outputs["time"] = timer.avg_time
        return outputs
    
    # endregion
    
# endregion
