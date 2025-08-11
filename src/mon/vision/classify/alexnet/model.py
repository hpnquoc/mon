#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AlexNet models for image classification."""

__all__ = [
    "AlexNet",
]

import box
from torchvision import models as tvm

import mon.nn as nn
from mon.constants import MLType, MODELS, Task, ZOO_DIR
from mon.core import pathlib

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="alexnet", arch="alexnet")
class AlexNet(tvm.AlexNet, nn.ModelMixin):
    """AlexNet model for image classification.
    
    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.5``.
    """
    
    arch     : str          = "alexnet"
    name     : str          = "alexnet",
    tasks    : list[Task]   = [Task.CLASSIFY]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
            "path"       : ZOO_DIR / "vision/classify/alexnet/alexnet/imagenet1k_v1/alexnet_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(
        self,
        weights    : str   = "imagenet1k_v1",
        num_classes: int   = 1000,
        dropout    : float = 0.5,
        *args, **kwargs
    ):
        weights, path, num_classes = self.parse_weights(weights, num_classes)
        super().__init__(num_classes=num_classes, dropout=dropout, *args, **kwargs)
        if weights:
            self.load_state_dict(weights)
