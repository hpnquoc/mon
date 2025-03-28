#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements AlexNet models."""

from __future__ import annotations

__all__ = [
    "AlexNet",
]

from torchvision.models import alexnet

from mon import core, nn
from mon.globals import LType, MODELS, ZOO_DIR
from mon.vision.classify import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Model

@MODELS.register(name="alexnet", arch="alexnet")
class AlexNet(nn.ExtraModel, base.ImageClassificationModel):
    """AlexNet model for image classification.
    
    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.5``.
    """
    
    arch     : str         = "alexnet"
    name     : str         = "alexnet",
    ltypes   : list[LType] = [LType.SUPERVISED]
    model_dir: core.Path   = current_dir
    zoo      : dict        = {
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
        """Initializes weights for the model.
    
        Args:
            m: ``nn.Module`` to initialize weights for.
        """
        pass

    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        """Performs forward pass on the model.
    
        Args:
            datapoint: ``dict`` with image data.
            args: Additional positional arguments.
            kwargs: Additional keyword arguments.
    
        Returns:
            ``dict`` with logits from the model.
        """
        x = datapoint["image"]
        y = self.model(x)
        return {"logits": y}

# endregion
