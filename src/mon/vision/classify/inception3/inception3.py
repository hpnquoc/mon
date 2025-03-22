#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Inception v3.

This module implements Inception models.
"""

from __future__ import annotations

__all__ = [
    "Inception3",
]

from torchvision.models import inception_v3

from mon import core, nn
from mon.globals import MODELS, LType, ZOO_DIR
from mon.vision.classify import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Model

@MODELS.register(name="inception_v3", arch="inception")
class Inception3(nn.ExtraModel, base.ImageClassificationModel):
    """Inception v3 model architecture from
    `Rethinking the Inception Architecture for Computer Vision <http://arxiv.org/abs/1512.00567>`_.

    Notes:
        **Important**: In contrast to the other models, the ``inception_v3``
        expects tensors with a size of `N x 3 x 299 x 299`, so ensure
        your images are sized accordingly.
    
    """
    
    arch     : str          = "inception"
    name     : str          = "inception_v3"
    ltypes   : list[LType]  = [LType.SUPERVISED]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth",
            "path"       : ZOO_DIR / "vision/classify/inception/inception_v3/imagenet1k_v1/inception_v3_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        num_classes: int   = 1000,
        aux_logits : bool  = True,
        dropout    : float = 0.5,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = inception_v3(
            num_classes = num_classes,
            aux_logits  = aux_logits,
            dropout     = dropout,
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
