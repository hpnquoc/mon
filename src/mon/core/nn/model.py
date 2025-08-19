#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base class for neural network models."""

__all__ = [
    "ModelMixin",
]

from typing import Any, Union

import box
import torch

from mon.core.enum import MLType, Task
from mon.core.pathlib import download_url_to_file, Path


# ----- Model -----
class ModelMixin:
    """Mixin for model attributes and methods."""

    arch     : str          = ""         # The model's architecture.
    name     : str          = ""         # The model's name.
    tasks    : list[Task]   = []         # A list of tasks that the model can perform.
    mltypes  : list[MLType] = []         # A list of learning types that the model can perform.
    model_dir: Path         = None       # The model's directory
    zoo      : dict         = box.Box()  # A dictionary containing all pretrained weights of the model.
    
    def parse_weights(
        self,
        weights    : Any,
        num_classes: int  = None,
        overwrite  : bool = False
    ) -> Union[dict, str, int]:
        """Parses and loads pretrained weights for the model.
    
        Args:
            weights: Weights as a ``dict``, ``str``, or ``Path`` to parse;
                Pass ``None`` to skip.
            num_classes: Number of classes for the model. Default: ``None``.
            overwrite: Overwrites existing weights if ``True``. Default: ``False``.
    
        Raises:
            ValueError: If the given weights path is invalid.
        """
        path = None
        
        # Pretrained weights from zoo
        if isinstance(weights, str) and weights in self.zoo:
            url         = self.zoo[weights].get("url",         None)
            path        = self.zoo[weights].get("path",        path)
            num_classes = self.zoo[weights].get("num_classes", num_classes)
            if url and path and not Path(path).is_weights_file(exist=True):
                download_url_to_file(url, path, overwrite)
        elif isinstance(weights, Path | str):
            path = weights
        
        # Path to weights file
        if path and Path(path).is_weights_file(exist=True):
            weights = torch.load(str(path))
        
        # State dict
        if isinstance(weights, dict):
            num_classes = weights.get("num_classes", num_classes)
        else:
            weights = None
        
        return weights, path, num_classes
