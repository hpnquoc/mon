#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GoogLeNet (Inception v1).

This module implements GoogleNet models.
"""

from __future__ import annotations

__all__ = [
    "GoogleNet",
]

from torchvision.models import googlenet

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Model

@MODELS.register(name="googlenet", arch="googlenet")
class GoogleNet(nn.ExtraModel, base.ImageClassificationModel):
    """GoogLeNet (Inception v1) models from the paper: "Going Deeper with
    Convolutions".
    
    References:
        https://arxiv.org/abs/1409.4842
    """
    
    arch     : str          = "googlenet"
    name     : str          = "googlenet"
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/googlenet-1378be20.pth",
            "path"       : ZOO_DIR / "vision/classify/googlenet/googlenet/imagenet1k_v1/googlenet_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        num_classes: int   = 1000,
        aux_logits : bool  = True,
        dropout    : float = 0.2,
        dropout_aux: float = 0.7,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = googlenet(
            num_classes = num_classes,
            aux_logits  = aux_logits,
            dropout     = dropout,
            dropout_aux = dropout_aux,
        )
        
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
