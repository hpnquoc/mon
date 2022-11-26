#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from one.nn import *
from one.vision.classification.resnet import ResNet

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "densenet121": {
        "name"    : "densenet121",
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64, 7, 2, 3, 1, 1, False]],     # 0
            [-1,     1,      BatchNorm2d,       []],                             # 1
            [-1,     1,      ReLU,              [True]],                         # 2
            [-1,     1,      MaxPool2d,         [3, 2, 1]],                      # 3
            [-1,     1,      DenseBlock,        [64,   32, 6,  4, 0.0, False]],  # 4   c2 = 64 + 32 * 6   = 256
            [-1,     1,      DenseTransition,   [256,  128]],                    # 5   c2 = 256 // 2      = 128
            [-1,     1,      DenseBlock,        [128,  32, 12, 4, 0.0, False]],  # 6   c2 = 128 + 32 * 12 = 512
            [-1,     1,      DenseTransition,   [512,  256]],                    # 7   c2 = 512 // 2      = 256
            [-1,     1,      DenseBlock,        [256,  32, 24, 4, 0.0, False]],  # 8   c2 = 256 + 32 * 24 = 1024
            [-1,     1,      DenseTransition,   [1024, 512]],                    # 9   c2 = 1024 // 2     = 512
            [-1,     1,      DenseBlock,        [512,  32, 16, 4, 0.0, False]],  # 10  c2 = 512 + 32 * 16 = 1024
            [-1,     1,      BatchNorm2d,       []],                             # 11
            [-1,     1,      AdaptiveAvgPool2d, [1]],                            # 12
        ],
        "head"    : [
            [-1,     1,      LinearClassifier,  [1024]],                         # 13
        ]
    },
    "densenet161": {
        "name"    : "densenet161",
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [96, 7, 2, 3, 1, 1, False]],     # 0
            [-1,     1,      BatchNorm2d,       []],                             # 1
            [-1,     1,      ReLU,              [True]],                         # 2
            [-1,     1,      MaxPool2d,         [3, 2, 1]],                      # 3
            [-1,     1,      DenseBlock,        [96,   48, 6,  4, 0.0, False]],  # 4   c2 = 96   + 48 * 6  = 384
            [-1,     1,      DenseTransition,   [384,  192]],                    # 5   c2 = 384 // 2       = 192
            [-1,     1,      DenseBlock,        [192,  48, 12, 4, 0.0, False]],  # 6   c2 = 192  + 48 * 12 = 768
            [-1,     1,      DenseTransition,   [768,  384]],                    # 7   c2 = 768 // 2       = 384
            [-1,     1,      DenseBlock,        [384,  48, 36, 4, 0.0, False]],  # 8   c2 = 384  + 48 * 36 = 2112
            [-1,     1,      DenseTransition,   [2112, 1056]],                   # 9   c2 = 2112 // 2      = 1056
            [-1,     1,      DenseBlock,        [1056, 48, 24, 4, 0.0, False]],  # 10  c2 = 1056 + 48 * 24 = 2208
            [-1,     1,      BatchNorm2d,       []],                             # 11
            [-1,     1,      AdaptiveAvgPool2d, [1]],                            # 12
        ],
        "head"    : [
            [-1,     1,      LinearClassifier,  [2208]],                         # 13
        ]
    },
    "densenet169": {
        "name"    : "densenet169",
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64, 7, 2, 3, 1, 1, False]],     # 0
            [-1,     1,      BatchNorm2d,       []],                             # 1
            [-1,     1,      ReLU,              [True]],                         # 2
            [-1,     1,      MaxPool2d,         [3, 2, 1]],                      # 3
            [-1,     1,      DenseBlock,        [64,   32, 6,  4, 0.0, False]],  # 4   c2 = 64 + 32 * 6   = 256
            [-1,     1,      DenseTransition,   [256,  128]],                    # 5   c2 = 256 // 2      = 128
            [-1,     1,      DenseBlock,        [128,  32, 12, 4, 0.0, False]],  # 6   c2 = 128 + 32 * 12 = 512
            [-1,     1,      DenseTransition,   [512,  256]],                    # 7   c2 = 512 // 2      = 256
            [-1,     1,      DenseBlock,        [256,  32, 32, 4, 0.0, False]],  # 8   c2 = 256 + 32 * 32 = 1280
            [-1,     1,      DenseTransition,   [1280, 640]],                    # 9   c2 = 1280 // 2     = 640
            [-1,     1,      DenseBlock,        [640,  32, 32, 4, 0.0, False]],  # 10  c2 = 640 + 32 * 32 = 1664
            [-1,     1,      BatchNorm2d,       []],                             # 11
            [-1,     1,      AdaptiveAvgPool2d, [1]],                            # 12
        ],
        "head"    : [
            [-1,     1,      LinearClassifier,  [1664]],                         # 13
        ]
    },
    "densenet201": {
        "name"    : "densenet201",
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64, 7, 2, 3, 1, 1, False]],     # 0
            [-1,     1,      BatchNorm2d,       []],                             # 1
            [-1,     1,      ReLU,              [True]],                         # 2
            [-1,     1,      MaxPool2d,         [3, 2, 1]],                      # 3
            [-1,     1,      DenseBlock,        [64,   32, 6,  4, 0.0, False]],  # 4   c2 = 64 + 32 * 6   = 256
            [-1,     1,      DenseTransition,   [256,  128]],                    # 5   c2 = 256 // 2      = 128
            [-1,     1,      DenseBlock,        [128,  32, 12, 4, 0.0, False]],  # 6   c2 = 128 + 32 * 12 = 512
            [-1,     1,      DenseTransition,   [512,  256]],                    # 7   c2 = 512 // 2      = 256
            [-1,     1,      DenseBlock,        [256,  32, 48, 4, 0.0, False]],  # 8   c2 = 256 + 32 * 48 = 1792
            [-1,     1,      DenseTransition,   [1792, 896]],                    # 9   c2 = 1792 // 2     = 896
            [-1,     1,      DenseBlock,        [896,  32, 32, 4, 0.0, False]],  # 10  c2 = 896 + 32 * 32 = 1920
            [-1,     1,      BatchNorm2d,       []],                             # 11
            [-1,     1,      AdaptiveAvgPool2d, [1]],                            # 12
        ],
        "head"    : [
            [-1,     1,      LinearClassifier,  [1920]],                         # 13
        ]
    },
}


@MODELS.register(name="densenet")
class DenseNet(ImageClassificationModel):
    """
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        project (str | None): Project name. Defaults to None.
        name (str | None): Model's name. In case None is given, it will be
            `self.__class__.__name__`. Defaults to None.
        fullname (str | None): Model's fullname in the following format:
            {name}-{data_name}-{postfix}. In case None is given, it will be
            `self.name`. Defaults to None.
        channels (int): Input channel. Defaults to 3.
        num_classes (int | None): Number of classes for classification or
            detection tasks. Defaults to None.
        classlabels (ClassLabels | None): ClassLabels object that contains all
            labels in the dataset. Defaults to None.
        pretrained (Pretrained): Initialize weights from pretrained.
            - If True, use the original pretrained described by the author
              (usually, ImageNet or COCO). By default, it is the first element
              in the `model_zoo` dictionary.
            - If str and is a file/path, then load weights from saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
        phase (ModelPhase_): Model's running phase. Defaults to training.
        loss (Losses_ | None): Loss function for training model.
            Defaults to None.
        metrics (Metrics_ | None): Metric(s) for validating and testing model.
            Defaults to None.
        optimizers (Optimizers_ | None): Optimizer(s) for training model.
            Defaults to None.
        debug (dict | Munch | None): Debug configs. Defaults to None.
        verbose (bool): Verbosity.
    """
    
    model_zoo = {
        "densenet121-imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/densenet121-a639ec97.pth",
            filename    = "densenet121-imagenet.pth",
            num_classes = 1000,
        ),
        "densenet161-imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/densenet161-8d451a50.pth",
            filename    = "densenet161-imagenet.pth",
            num_classes = 1000,
        ),
        "densenet169-imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
            filename    = "densenet169-imagenet.pth",
            num_classes = 1000,
        ),
        "densenet201-imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/densenet201-c1103571.pth",
            filename    = "densenet201-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "densenet121.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "densenet",
        fullname   : str          | None = "densenet121",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg, variant = parse_cfg_variant(cfg=cfg, cfgs=cfgs, cfg_dir=CFG_DIR)
        pretrained   = parse_pretrained(pretrained=pretrained, variant=variant)
        super().__init__(
            cfg         = cfg,
            root        = root,
            project     = project,
            name        = name,
            fullname    = fullname,
            variant     = variant,
            channels    = channels,
            num_classes = num_classes,
            pretrained  = DenseNet.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def init_weights(self, m: Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                nn.init.kaiming_normal_(m.conv.weight)
            else:
                nn.init.kaiming_normal_(m.weight)
        elif classname.find("BatchNorm") != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find("Linear") != -1:
            nn.init.constant_(m.bias, 0)
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained) and self.pretrained["name"] == "imagenet":
            state_dict = load_state_dict_from_path(
                model_dir=self.pretrained_dir, **self.pretrained
            )
            """
            for k in sorted(self.model.state_dict().keys()):
                print(f"model_state_dict[\"{k}\"] = ")
            for k in sorted(state_dict.keys()):
                print(f"state_dict[\"{k}\"]")
            """
            model_state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                if "features.conv0" in k:
                    k = k.replace("features.conv0", "0")
                elif "features.norm0" in k:
                    k = k.replace("features.norm0", "1")
                elif "features.denseblock1" in k:
                    k = k.replace("features.denseblock1", "4")
                    k = k.replace("norm.", "norm")
                    k = k.replace("conv.", "conv")
                elif "features.transition1" in k:
                    k = k.replace("features.transition1", "5")
                elif "features.denseblock2" in k:
                    k = k.replace("features.denseblock2", "6")
                    k = k.replace("norm.", "norm")
                    k = k.replace("conv.", "conv")
                elif "features.transition2" in k:
                    k = k.replace("features.transition2", "7")
                elif "features.denseblock3" in k:
                    k = k.replace("features.denseblock3", "8")
                    k = k.replace("norm.", "norm")
                    k = k.replace("conv.", "conv")
                elif "features.transition3" in k:
                    k = k.replace("features.transition3", "9")
                elif "features.denseblock4" in k:
                    k = k.replace("features.denseblock4", "10")
                    k = k.replace("norm.", "norm")
                    k = k.replace("conv.", "conv")
                elif "features.norm5" in k:
                    k = k.replace("features.norm5", "11")
                elif "classifier" in k:
                    continue
                model_state_dict[k] = v
            
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["13.linear.bias"]   = state_dict["classifier.bias"]
                model_state_dict["13.linear.weight"] = state_dict["classifier.weight"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()
