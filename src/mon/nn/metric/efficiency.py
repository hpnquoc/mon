#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Efficiency Metric.

This module implements efficiency score metrics.
"""

from __future__ import annotations

__all__ = [
	"compute_efficiency_score",
]

from copy import deepcopy
from typing import Sequence

import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from torch import nn

from mon import core

console = core.console


# region Efficiency Metric

def compute_efficiency_score(
	model     : nn.Module,
	image_size: int | Sequence[int] = 512,
	channels  : int  = 3,
) -> tuple[float, float]:
	from mon.vision.dtype import image as I
	# Define input tensor
	h, w  = I.get_image_size(image_size)
	input = torch.rand(1, channels, h, w)
	input = input.to(core.get_model_device(model))
	# Get FLOPs and Params
	flops, params = core.profile(deepcopy(model), inputs=(input, ), verbose=False)
	flops         = FlopCountAnalysis(model, input).total() if flops == 0 else flops
	params        = model.params               if hasattr(model, "params") and params == 0 else params
	params        = parameter_count(model)     if hasattr(model, "params") else params
	params        = sum(list(params.values())) if isinstance(params, dict) else params
	# Return
	return flops, params

# endregion
