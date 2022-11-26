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
    "wide-resnet50" : {
        "name"              : "wide-resnet50",
        "zero_init_residual": False,
        "channels"          : 3,
        "backbone"          : [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64, 7, 2, 3, 1, 1, False]],                                          # 0
            [-1,     1,      BatchNorm2d,       []],                                                                  # 1
            [-1,     1,      ReLU,              [True]],                                                              # 2
            [-1,     1,      MaxPool2d,         [3, 2, 1]],                                                           # 3
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 3, 64,   64,  1, 1, 1, 128, False, BatchNorm2d]],  # 4
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 4, 256,  128, 2, 1, 1, 128, False, BatchNorm2d]],  # 5
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 6, 512,  256, 2, 1, 1, 128, False, BatchNorm2d]],  # 6
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 3, 1024, 512, 2, 1, 1, 128, False, BatchNorm2d]],  # 7
            [-1,     1,      AdaptiveAvgPool2d, [1]],                                                                 # 8
        ],
        "head"              : [
            [-1,     1,      LinearClassifier,  [2048]],                                                              # 9
        ]
    },
    "wide-resnet101": {
        "name"              : "wide-resnet101",
        "zero_init_residual": False,
        "channels"          : 3,
        "backbone"          : [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64, 7, 2, 3, 1, 1, False]],                                           # 0
            [-1,     1,      BatchNorm2d,       []],                                                                   # 1
            [-1,     1,      ReLU,              [True]],                                                               # 2
            [-1,     1,      MaxPool2d,         [3, 2, 1]],                                                            # 3
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 3,  64,   64,  1, 1, 1, 128, False, BatchNorm2d]],  # 4
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 4,  256,  128, 2, 1, 1, 128, False, BatchNorm2d]],  # 5
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 23, 512,  256, 2, 1, 1, 128, False, BatchNorm2d]],  # 6
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 3,  1024, 512, 2, 1, 1, 128, False, BatchNorm2d]],  # 7
            [-1,     1,      AdaptiveAvgPool2d, [1]],                                                                  # 8
        ],
        "head"              : [
            [-1,     1,      LinearClassifier,  [2048]],                                                               # 9
        ]
    },
}


@MODELS.register(name="wide-resnet")
class WideResNet(ImageClassificationModel):
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
        "wide-resnet50-imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
            filename    = "wide-resnet50-imagenet.pth",
            num_classes = 1000,
        ),
        "wide-resnet101-imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
            filename    = "wide-resnet101-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "wide-resnet50.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "wide-resnet",
        fullname   : str          | None = "wide-resnet",
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
            classlabels = classlabels,
            pretrained  = WideResNet.init_pretrained(pretrained),
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
                nn.init.kaiming_normal_(m.conv.weight, mode="fan_out", nonlinearity="relu")
            else:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif classname.find("BatchNorm") != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find("GroupNorm") != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        zero_init_residual = self.cfg["zero_init_residual"]
        if zero_init_residual:
            if isinstance(m, ResNetBottleneck) and m.bn3.weight is not None:
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, ResNetBottleneck) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)
    
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
                if "layer1" in k:
                    k = k.replace("layer1", "4.convs")
                elif "layer2" in k:
                    k = k.replace("layer2", "5.convs")
                elif "layer3" in k:
                    k = k.replace("layer3", "6.convs")
                elif "layer4" in k:
                    k = k.replace("layer4", "7.convs")
                else:
                    continue
                model_state_dict[k] = v
            model_state_dict["0.weight"]       = state_dict["conv1.weight"]
            model_state_dict["1.weight"]       = state_dict["bn1.weight"]
            model_state_dict["1.bias"]         = state_dict["bn1.bias"]
            model_state_dict["1.running_mean"] = state_dict["bn1.running_mean"]
            model_state_dict["1.running_var"]  = state_dict["bn1.running_var"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["9.linear.weight"] = state_dict["fc.weight"]
                model_state_dict["9.linear.bias"]   = state_dict["fc.bias"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()
