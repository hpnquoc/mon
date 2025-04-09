#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DenseNet models."""

__all__ = [
    "DenseNet121",
    "DenseNet161",
    "DenseNet169",
    "DenseNet201",
]

from abc import ABC

from torchvision.models import (
    densenet121, densenet161, densenet169, densenet201,
)

from mon import core, nn
from mon.constants import MLType, MODELS, ZOO_DIR
from mon.vision.classify import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Model -----
class DenseNet(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """DenseNet model for image classification.

    References:
        - https://arxiv.org/pdf/1608.06993.pdf
    """
    
    arch     : str          = "densenet"
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
    

@MODELS.register(name="densenet121", arch="densenet")
class DenseNet121(DenseNet):
    """DenseNet-121 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "densenet121"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/densenet121-a639ec97.pth",
            "path"       : ZOO_DIR / "vision/classify/densenet/densenet121/imagenet1k_v1/densenet121_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = densenet121(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        

@MODELS.register(name="densenet161", arch="densenet")
class DenseNet161(DenseNet):
    """DenseNet-161 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "densenet161"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/densenet161-8d451a50.pth",
            "path"       : ZOO_DIR / "vision/classify/densenet/densenet161/imagenet1k_v1/densenet161_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = densenet161(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="densenet169", arch="densenet")
class DenseNet169(DenseNet):
    """DenseNet-169 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "densenet169"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
            "path"       : ZOO_DIR / "vision/classify/densenet/densenet169/imagenet1k_v1/densenet169_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = densenet169(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="densenet201", arch="densenet")
class DenseNet201(DenseNet):
    """DenseNet-201 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "densenet201"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/densenet201-c1103571.pth",
            "path"       : ZOO_DIR / "vision/classify/densenet/densenet201/imagenet1k_v1/densenet201_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = densenet201(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
