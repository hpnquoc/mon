#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements MNASNet models."""

__all__ = [
    "MNASNet0_5",
    "MNASNet0_75",
    "MNASNet1_0",
    "MNASNet1_3",
]

from abc import ABC

from torchvision.models import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3

from mon import core, nn
from mon.constants import MLType, MODELS, ZOO_DIR
from mon.vision.classify import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Model -----
class MNASNet(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """MNASNet model for image classification.

    References:
        - https://arxiv.org/abs/1807.11626
    """
    
    arch     : str          = "mnasnet"
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
    

@MODELS.register(name="mnasnet0_5", arch="mnasnet")
class MNASNet0_5(MNASNet):
    """MNASNet-0.5 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.2``.
    """
    
    name: str  = "mnasnet0_5"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
            "path"       : ZOO_DIR / "vision/classify/mnasnet/mnasnet0_5/imagenet1k_v1/mnasnet0_5_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = mnasnet0_5(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        

@MODELS.register(name="mnasnet0_75", arch="mnasnet")
class MNASNet0_75(MNASNet):
    """MNASNet-0.75 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.2``.
    """
    
    name: str  = "mnasnet0_75"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mnasnet0_75-7090bc5f.pth",
            "path"       : ZOO_DIR / "vision/classify/mnasnet/mnasnet0_75/imagenet1k_v1/mnasnet0_75_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = mnasnet0_75(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
        
@MODELS.register(name="mnasnet1_0", arch="mnasnet")
class MNASNet1_0(MNASNet):
    """MNASNet-1.0 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.2``.
    """
    
    name: str  = "mnasnet1_0"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
            "path"       : ZOO_DIR / "vision/classify/mnasnet/mnasnet1_0/imagenet1k_v1/mnasnet1_0_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = mnasnet1_0(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
        
@MODELS.register(name="mnasnet1_3", arch="mnasnet")
class MNASNet1_3(MNASNet):
    """MNASNet-1.3 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.2``.
    """
    
    name: str  = "mnasnet1_3"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mnasnet1_3-a4c69d6f.pth",
            "path"       : ZOO_DIR / "vision/classify/mnasnet/mnasnet1_3/imagenet1k_v1/mnasnet1_3_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = mnasnet1_3(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
