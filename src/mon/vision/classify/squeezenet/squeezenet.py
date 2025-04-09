#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements SqueezeNet models."""

__all__ = [
    "SqueezeNet1_0",
    "SqueezeNet1_1",
]

from abc import ABC

from torchvision.models import squeezenet1_0, squeezenet1_1

from mon import core, nn
from mon.constants import MLType, MODELS, ZOO_DIR
from mon.vision.classify import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Model -----
class SqueezeNet(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """SqueezeNet model for image classification.

    References:
        - https://arxiv.org/abs/1602.07360
    """
    
    arch     : str          = "squeezenet"
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {}

    # ----- Initialize -----
    def init_weights(self, m: nn.Module):
        """Initializes weights for the model.
    
        Args:
            m: ``nn.Module`` to initialize weights for.
        """
        pass

    # ----- Forward Pass -----
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        """Performs forward pass on the model.
    
        Args:
            datapoint: ``dict`` with image data.
    
        Returns:
            ``dict`` of predictions with ``"logits"`` keys.
        """
        x = datapoint["image"]
        y = self.model(x)
        return {"logits": y}


@MODELS.register(name="squeezenet1_0", arch="squeezenet")
class SqueezeNet1_0(SqueezeNet):
    """SqueezeNet-1.0 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.5``.
    """
    
    name: str  = "squeezenet1_0"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth",
            "path"       : ZOO_DIR / "vision/classify/squeezenet/squeezenet1_0/imagenet1k_v1/squeezenet1_0_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = squeezenet1_0(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
       

@MODELS.register(name="squeezenet1_1", arch="squeezenet")
class SqueezeNet1_1(SqueezeNet):
    """SqueezeNet-1.1 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.5``.
    """
    
    name: str  = "squeezenet1_1"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
            "path"       : ZOO_DIR / "vision/classify/squeezenet/squeezenet1_1/imagenet1k_v1/squeezenet1_1_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = squeezenet1_1(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
