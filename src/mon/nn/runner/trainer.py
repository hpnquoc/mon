#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trainer.

This module implements the training procedure for neural networks.
"""

from __future__ import annotations

__all__ = [
    "Trainer",
    "seed_everything",
]

import lightning
from lightning.pytorch.trainer import *

from mon import core
from mon.nn import strategy

console = core.console


# region Trainer

class Trainer(lightning.Trainer):
    """Extends lightning.Trainer with custom methods and properties.

    Args:
        log_image_every_n_epochs: Log debug images every n epochs. Default is ``0``.
    """
    
    def __init__(self, log_image_every_n_epochs: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_image_every_n_epochs = log_image_every_n_epochs
        
    @lightning.Trainer.current_epoch.setter
    def current_epoch(self, current_epoch: int):
        """Sets the current epoch."""
        self.fit_loop.current_epoch = current_epoch

    @lightning.Trainer.global_step.setter
    def global_step(self, global_step: int):
        """Sets the global step."""
        self.fit_loop.global_step = global_step
    
    def _log_device_info(self):
        """Logs device availability and usage info."""
        gpu_available, gpu_type = (
            (True, " (cuda)") if strategy.CUDAAccelerator.is_available() else
            (True, " (mps)") if strategy.MPSAccelerator.is_available() else
            (False, "")
        )
        gpu_used = isinstance(self.accelerator, (strategy.CUDAAccelerator, strategy.MPSAccelerator))
        console.log(f"GPU available: {gpu_available}{gpu_type}, used: {gpu_used}.")
    
        num_tpu_cores = self.num_devices if isinstance(self.accelerator, strategy.TPUAccelerator) else 0
        console.log(f"TPU available: {strategy.TPUAccelerator.is_available()}, using: {num_tpu_cores} TPU cores.")
    
        if strategy.CUDAAccelerator.is_available() and not isinstance(self.accelerator, strategy.CUDAAccelerator):
            console.log(
                f"GPU available but not used. Set `accelerator` and `devices` using "
                f"Trainer(accelerator='gpu', devices={strategy.CUDAAccelerator.auto_device_count()})."
            )
        if strategy.TPUAccelerator.is_available() and not isinstance(self.accelerator, strategy.TPUAccelerator):
            console.log(
                f"TPU available but not used. Set `accelerator` and `devices` using "
                f"Trainer(accelerator='tpu', devices={strategy.TPUAccelerator.auto_device_count()})."
            )
        if strategy.MPSAccelerator.is_available() and not isinstance(self.accelerator, strategy.MPSAccelerator):
            console.log(
                f"MPS available but not used. Set `accelerator` and `devices` using "
                f"Trainer(accelerator='mps', devices={strategy.MPSAccelerator.auto_device_count()})."
            )

# endregion
