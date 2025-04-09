#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements EfficientNet models."""

__all__ = [
    "EfficientNet_B0",
    "EfficientNet_B1",
    "EfficientNet_B2",
    "EfficientNet_B3",
    "EfficientNet_B4",
    "EfficientNet_B5",
    "EfficientNet_B6",
    "EfficientNet_B7",
    "EfficientNet_V2_L",
    "EfficientNet_V2_M",
    "EfficientNet_V2_S",
]

from abc import ABC

from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
    efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s,
)

from mon import core, nn
from mon.constants import MLType, MODELS, ZOO_DIR
from mon.vision.classify import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Model -----
class EfficientNet(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """EfficientNet model for image classification.

    References:
        - https://arxiv.org/abs/1905.11946
    """
    
    arch     : str          = "efficientnet"
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
    

@MODELS.register(name="efficientnet_b0", arch="efficientnet")
class EfficientNet_B0(EfficientNet):
    """EfficientNet-B0 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "efficientnet_b0"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b0/imagenet1k_v1/efficientnet_b0_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = efficientnet_b0(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_b1", arch="efficientnet")
class EfficientNet_B1(EfficientNet):
    """EfficientNet-B1 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "efficientnet_b1"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b1_rwightman-bac287d4.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b1/imagenet1k_v1/efficientnet_b1_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b1/imagenet1k_v2/efficientnet_b1_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = efficientnet_b1(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_b2", arch="efficientnet")
class EfficientNet_B2(EfficientNet):
    """EfficientNet-B2 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "efficientnet_b2"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b2_rwightman-c35c1473.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b2/imagenet1k_v1/efficientnet_b2_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = efficientnet_b2(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_b3", arch="efficientnet")
class EfficientNet_B3(EfficientNet):
    """EfficientNet-B3 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "efficientnet_b3"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b3_rwightman-b3899882.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b3/imagenet1k_v1/efficientnet_b3_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = efficientnet_b3(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_b4", arch="efficientnet")
class EfficientNet_B4(EfficientNet):
    """EfficientNet-B4 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "efficientnet_b4"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b4_rwightman-23ab8bcd.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b4/imagenet1k_v1/efficientnet_b4_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = efficientnet_b4(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_b5", arch="efficientnet")
class EfficientNet_B5(EfficientNet):
    """EfficientNet-B5 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "efficientnet_b5"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b5_lukemelas-1a07897c.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b5/imagenet1k_v1/efficientnet_b5_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = efficientnet_b5(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_b6", arch="efficientnet")
class EfficientNet_B6(EfficientNet):
    """EfficientNet-B6 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "efficientnet_b6"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b6_lukemelas-24a108a5.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b6/imagenet1k_v1/efficientnet_b6_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = efficientnet_b6(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
   
   
@MODELS.register(name="efficientnet_b7", arch="efficientnet")
class EfficientNet_B7(EfficientNet):
    """EfficientNet-B7 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "efficientnet_b7"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b7_lukemelas-c5b4e57e.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b7/imagenet1k_v1/efficientnet_b7_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = efficientnet_b7(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_v2_s", arch="efficientnet")
class EfficientNet_V2_S(EfficientNet):
    """EfficientNet-V2 Small model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "efficientnet_v2_s"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_v2_s/imagenet1k_v1/efficientnet_v2_s_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = efficientnet_v2_s(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_v2_m", arch="efficientnet")
class EfficientNet_V2_M(EfficientNet):
    """EfficientNet-V2 Medium model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "efficientnet_v2_m"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_v2_m/imagenet1k_v1/efficientnet_v2_m_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = efficientnet_v2_m(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_v2_l", arch="efficientnet")
class EfficientNet_V2_L(EfficientNet):
    """EfficientNet-V2 Large model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "efficientnet_v2_l"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_v2_l-59c71312.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_v2_l/imagenet1k_v1/efficientnet_v2_l_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = efficientnet_v2_l(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
