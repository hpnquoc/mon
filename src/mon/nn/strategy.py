#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Strategy.

This module implements strategies used during training machine learning models.
A strategy is a composition of one Accelerator, one Precision Plugin, a
CheckpointIO plugin, and other optional plugins such as the ClusterEnvironment.

References:
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html
"""

from __future__ import annotations

__all__ = [
    # Accelerator
    "Accelerator",
    "CPUAccelerator",
    "CUDAAccelerator",
    "MPSAccelerator",
    "XLAAccelerator",
    # Strategy
    "DDPStrategy",
    "DeepSpeedStrategy",
    "FSDPStrategy",
    "ParallelStrategy",
    "SingleDeviceStrategy",
    "Strategy",
    "XLAStrategy",
]

import os
import platform
from typing import Callable

import torch
import torch.cuda
from lightning.pytorch import accelerators, strategies
from torch import distributed

from mon import core
from mon.globals import ACCELERATORS, STRATEGIES

console = core.console


# region Accelerator

Accelerator     = accelerators.Accelerator
CPUAccelerator  = accelerators.CPUAccelerator
CUDAAccelerator = accelerators.CUDAAccelerator
MPSAccelerator  = accelerators.MPSAccelerator
XLAAccelerator  = accelerators.XLAAccelerator

ACCELERATORS.register(name="cpu" , module=CPUAccelerator)
ACCELERATORS.register(name="cuda", module=CUDAAccelerator)
ACCELERATORS.register(name="gpu" , module=CUDAAccelerator)
ACCELERATORS.register(name="mps" , module=MPSAccelerator)
ACCELERATORS.register(name="xla" , module=XLAAccelerator)

# endregion


# region Strategy

Strategy             = strategies.Strategy
DDPStrategy          = strategies.DDPStrategy
DeepSpeedStrategy    = strategies.DeepSpeedStrategy
FSDPStrategy         = strategies.FSDPStrategy
ParallelStrategy     = strategies.ParallelStrategy
SingleDeviceStrategy = strategies.SingleDeviceStrategy
XLAStrategy          = strategies.XLAStrategy

STRATEGIES.register(name = "ddp"          , module = DDPStrategy)
STRATEGIES.register(name = "deepspeed"    , module = DeepSpeedStrategy)
STRATEGIES.register(name = "fsdp"         , module = FSDPStrategy)
STRATEGIES.register(name = "parallel"     , module = ParallelStrategy)
STRATEGIES.register(name = "single_device", module = SingleDeviceStrategy)
STRATEGIES.register(name = "xla"          , module = XLAStrategy)

# endregion


# region Helper Function

def get_distributed_info() -> list[int]:
    """Returns rank and world size if distributed, else [0, 1].

    Returns:
        List of ``[rank, world_size]`` for the current process.
    """
    if distributed.is_available() and distributed.is_initialized():
        return [distributed.get_rank(), distributed.get_world_size()]
    return [0, 1]


def set_distributed_backend(strategy: str | Callable, cudnn: bool = True):
    """Sets distributed backend based on OS and strategy.

    Args:
        strategy: Distributed strategy (``"ddp"``, ``"ddp2"``) or callable.
        cudnn: Enable cuDNN if ``True``. Default is ``True``.
    """
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = cudnn
        console.log(f"cuDNN available: [bright_green]True[/bright_green], used: [bright_green]{cudnn}[/bright_green]")
    else:
        console.log(f"cuDNN available: [red]False[/red]")

    if strategy in ["ddp"] or isinstance(strategy, DDPStrategy):
        backend = "gloo" if platform.system() == "Windows" else "nccl"
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = backend
        console.log(f"Running on a {platform.system()} machine, set torch distributed backend to {backend}.")
            
# endregion
