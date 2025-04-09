#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ResNet models."""

__all__ = [
    "ResNeXt101_32X8D",
    "ResNeXt101_64X4D",
    "ResNeXt50_32X4D",
    "ResNet101",
    "ResNet152",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "WideResNet101_2",
    "WideResNet50_2",
]

from abc import ABC

from torchvision.models import (
    resnet101, resnet152, resnet18, resnet34,
    resnet50, resnext101_32x8d, resnext101_64x4d, resnext50_32x4d,
    wide_resnet101_2, wide_resnet50_2,
)

from mon import core, nn
from mon.constants import MLType, MODELS, ZOO_DIR
from mon.vision.classify import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- ResNet -----
class ResNet(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """ResNet model for image classification.

    References:
        - https://arxiv.org/abs/1512.03385
    """
    
    arch     : str          = "resnet"
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
    

@MODELS.register(name="resnet18", arch="resnet")
class ResNet18(ResNet):
    """ResNet-18 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "resnet18"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/resnet18/imagenet1k_v1/resnet18_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = resnet18(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="resnet34", arch="resnet")
class ResNet34(ResNet):
    """ResNet-34 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "resnet34"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/resnet34-b627a593.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/resnet34/imagenet1k_v1/resnet34_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = resnet34(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="resnet50", arch="resnet")
class ResNet50(ResNet):
    """ResNet-50 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "resnet50"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/resnet50/imagenet1k_v1/resnet50_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/resnet50-0676ba61.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/resnet50/imagenet1k_v2/resnet50_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = resnet50(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="resnet101", arch="resnet")
class ResNet101(ResNet):
    """ResNet-101 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "resnet101"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/resnet101-63fe2227.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/resnet101/imagenet1k_v1/resnet101_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/resnet101/imagenet1k_v2/resnet101_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = resnet101(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="resnet152", arch="resnet")
class ResNet152(ResNet):
    """ResNet-152 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "resnet152"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/resnet152-394f9c45.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/resnet152/imagenet1k_v1/resnet152_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/resnet152-f82ba261.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/resnet152/imagenet1k_v2/resnet152_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = resnet152(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        

# ----- ResNeXt -----
@MODELS.register(name="resnext50_32x4d", arch="resnet")
class ResNeXt50_32X4D(ResNet):
    """ResNeXt-50-32x4d model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "resnext50_32x4d"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/resnext50_32x4d/imagenet1k_v1/resnext50_32x4d_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/resnext50_32x4d/imagenet1k_v2/resnext50_32x4d_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = resnext50_32x4d(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="resnext101_32x8d", arch="resnet")
class ResNeXt101_32X8D(ResNet):
    """ResNeXt-101-32x8d model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "resnext101_32x8d"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/resnext101_32x8d/imagenet1k_v1/resnext101_32x8d_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/resnext101_32x8d/imagenet1k_v2/resnext101_32x8d_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = resnext101_32x8d(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="resnext101_64x4d", arch="resnet")
class ResNeXt101_64X4D(ResNet):
    """ResNeXt-101-64x4d model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "resnext101_64x4d"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/resnext101_64x4d/imagenet1k_v1/resnext101_64x4d_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = resnext101_64x4d(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        

# ----- WideResNet -----
@MODELS.register(name="wide_resnet50_2", arch="resnet")
class WideResNet50_2(ResNet):
    """WideResNet-50-2 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "wide_resnet50_2"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/wide_resnet50/imagenet1k_v1/wide_resnet50_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/wide_resnet50/imagenet1k_v2/wide_resnet50_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = wide_resnet50_2(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="wide_resnet101_2", arch="resnet")
class WideResNet101_2(ResNet):
    """WideResNet-101-2 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "wide_resnet101_2"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/wide_resnet101/imagenet1k_v1/wide_resnet101_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
            "path"       : ZOO_DIR / "vision/classify/resnet/wide_resnet101/imagenet1k_v2/wide_resnet101_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = wide_resnet101_2(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
