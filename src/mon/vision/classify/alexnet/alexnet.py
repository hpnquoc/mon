#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AlexNet.

This module implements AlexNet models.
"""

from __future__ import annotations

__all__ = [
    "AlexNet",
]

from torchvision.models import alexnet

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Model

@MODELS.register(name="alexnet", arch="alexnet")
class AlexNet(nn.ExtraModel, base.ImageClassificationModel):
    
    arch     : str          = "alexnet"
    name     : str          = "alexnet",
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
            "path"       : ZOO_DIR / "vision/classify/alexnet/alexnet/imagenet1k_v1/alexnet_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = alexnet(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        x = datapoint["image"]
        y = self.model(x)
        return {"logits": y}

# endregion
