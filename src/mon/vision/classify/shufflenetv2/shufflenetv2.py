#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ShuffleNetV2 models."""

__all__ = [
    "ShuffleNetV2_X1_0",
    "ShuffleNetV2_X1_5",
    "ShuffleNetV2_X2_0",
    "ShuffleNetV2_x0_5",
]

from abc import ABC

from torchvision.models import (
    shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
)

from mon import core, nn
from mon.constants import MLType, MODELS, ZOO_DIR
from mon.vision.classify import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Model -----
class ShuffleNetV2(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """ShuffleNetV2 model for image classification.

    References:
        - https://arxiv.org/abs/1807.11164
    """
    
    arch     : str          = "shufflenet"
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


@MODELS.register(name="shufflenet_v2_x0_5", arch="shufflenet")
class ShuffleNetV2_x0_5(ShuffleNetV2):
    """ShuffleNetV2-x0.5 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "shufflenet_v2_x0_5"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
            "path"       : ZOO_DIR / "vision/classify/shufflenet/shufflenet_v2_x0_5/imagenet1k_v1/shufflenet_v2_x0_5_x0_5_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = shufflenet_v2_x0_5(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="shufflenet_v2_x1_0", arch="shufflenet")
class ShuffleNetV2_X1_0(ShuffleNetV2):
    """ShuffleNetV2-x1.0 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "shufflenet_v2_x1_0"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
            "path"       : ZOO_DIR / "vision/classify/shufflenet/shufflenetv2_x1_0/imagenet1k_v1/shufflenetv2_x1_0_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = shufflenet_v2_x1_0(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="shufflenet_v2_x1_5", arch="shufflenet")
class ShuffleNetV2_X1_5(ShuffleNetV2):
    """ShuffleNetV2-x1.5 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "shufflenet_v2_x1_5"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth",
            "path"       : ZOO_DIR / "vision/classify/shufflenet/shufflenetv2_x1_5/imagenet1k_v1/shufflenetv2_x1_5_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = shufflenet_v2_x1_5(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="shufflenet_v2_x2_0", arch="shufflenet")
class ShuffleNetV2_X2_0(ShuffleNetV2):
    """ShuffleNetV2-x2.0 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "shufflenet_v2_x2_0"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth",
            "path"       : ZOO_DIR / "vision/classify/shufflenet/shufflenetv2_x2_0/imagenet1k_v1/shufflenetv2_x2_0_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = shufflenet_v2_x2_0(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
