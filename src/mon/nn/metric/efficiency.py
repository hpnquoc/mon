#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Efficiency Metric Module.

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
	runs      : int  = 1000,
	use_cuda  : bool = True,
	verbose   : bool = False,
):
	# Define input tensor
	h, w  = core.get_image_size(image_size)
	input = torch.rand(1, channels, h, w)
	
	# Deploy to cuda
	if use_cuda:
		input = input.cuda()
		model = model.cuda()
		# device = torch.device("cuda:0")
		# input  = input.to(device)
		# model  = model.to(device)
	
	# Get FLOPs and Params
	flops, params = core.profile(deepcopy(model), inputs=(input, ), verbose=verbose)
	flops         = FlopCountAnalysis(model, input).total() if flops == 0 else flops
	params        = model.params               if hasattr(model, "params") and params == 0 else params
	params        = parameter_count(model)     if hasattr(model, "params") else params
	params        = sum(list(params.values())) if isinstance(params, dict) else params
	g_flops       = flops * 1e-9
	m_params      = int(params) * 1e-6
	
	# Get time
	timer = core.Timer()
	for i in range(runs):
		timer.tick()
		_ = model(input)
		timer.tock()
	avg_time = timer.avg_time
	
	# Print
	if verbose:
		console.log(f"FLOPs (G) : {flops:.4f}")
		console.log(f"Params (M): {params:.4f}")
		console.log(f"Time (s)  : {avg_time:.17f}")
	
	return flops, params, avg_time

# endregion
