#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from torchvision.ops import Conv2dNormActivation

from one.nn import *

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "convnext-base" : {
        "name"    : "convnext-base",
        "channels": 3,
        "backbone": [
            # [from, number, module,               args(out_channels, ...)]
            [-1,     1,      Conv2dNormActivation, [128,  4, 4, 0, 1, LayerNorm2d, None, 1, True, True]],  # 0 total_stage_blocks = 36
            [-1,     1,      ConvNeXtBlock,        [128,  1e-6, 0.5, 3,  0,  36]],                         # 1
            [-1,     1,      LayerNorm2d,          [128]],                                                 # 2
            [-1,     1,      Conv2d,               [256, 2, 2]],                                           # 3
            [-1,     1,      ConvNeXtBlock,        [256,  1e-6, 0.5, 3,  3,  36]],                         # 4
            [-1,     1,      LayerNorm2d,          [256]],                                                 # 5
            [-1,     1,      Conv2d,               [512, 2, 2]],                                           # 6
            [-1,     1,      ConvNeXtBlock,        [512,  1e-6, 0.5, 27, 6,  36]],                         # 7
            [-1,     1,      LayerNorm2d,          [512]],                                                 # 8
            [-1,     1,      Conv2d,               [1024, 2, 2]],                                          # 9
            [-1,     1,      ConvNeXtBlock,        [1024, 1e-6, 0.5, 3,  33, 36]],                         # 10
            [-1,     1,      AdaptiveAvgPool2d,    [1]],                                                   # 11
        ],
        "head"    : [
            [-1,     1,      ConvNeXtClassifier,   [1024, LayerNorm2d]],                                   # 12
        ]
    },
    "convnext-tiny" : {
        "name"    : "convnext-tiny",
        "channels": 3,
        "backbone": [
            # [from, number, module,               args(out_channels, ...)]
            [-1,     1,      Conv2dNormActivation, [96,  4, 4, 0, 1, LayerNorm2d, None, 1, True, True]],  # 0 total_stage_blocks = 18
            [-1,     1,      ConvNeXtBlock,        [96,  1e-6, 0.1, 3, 0,  18]],                          # 1
            [-1,     1,      LayerNorm2d,          [96]],                                                 # 2
            [-1,     1,      Conv2d,               [192, 2, 2]],                                          # 3
            [-1,     1,      ConvNeXtBlock,        [192, 1e-6, 0.1, 3, 3,  18]],                          # 4
            [-1,     1,      LayerNorm2d,          [192]],                                                # 5
            [-1,     1,      Conv2d,               [384, 2, 2]],                                          # 6
            [-1,     1,      ConvNeXtBlock,        [384, 1e-6, 0.1, 9, 6,  18]],                          # 7
            [-1,     1,      LayerNorm2d,          [384]],                                                # 8
            [-1,     1,      Conv2d,               [768, 2, 2]],                                          # 9
            [-1,     1,      ConvNeXtBlock,        [768, 1e-6, 0.1, 3, 15, 18]],                          # 10
            [-1,     1,      AdaptiveAvgPool2d,    [1]],                                                  # 11
        ],
        "head"    : [
            [-1,     1,      ConvNeXtClassifier,   [768, LayerNorm2d]],                                   # 12
        ]
    },
    "convnext-small": {
        "name"    : "convnext-small",
        "channels": 3,
        "backbone": [
            # [from, number, module,               args(out_channels, ...)]
            [-1,     1,      Conv2dNormActivation, [96,  4, 4, 0, 1, LayerNorm2d, None, 1, True, True]],  # 0 total_stage_blocks = 36
            [-1,     1,      ConvNeXtBlock,        [96,  1e-6, 0.4, 3,  0,  36]],                         # 1
            [-1,     1,      LayerNorm2d,          [96]],                                                 # 2
            [-1,     1,      Conv2d,               [192, 2, 2]],                                          # 3
            [-1,     1,      ConvNeXtBlock,        [192, 1e-6, 0.4, 3,  3,  36]],                         # 4
            [-1,     1,      LayerNorm2d,          [192]],                                                # 5
            [-1,     1,      Conv2d,               [384, 2, 2]],                                          # 6
            [-1,     1,      ConvNeXtBlock,        [384, 1e-6, 0.4, 27, 6,  36]],                         # 7
            [-1,     1,      LayerNorm2d,          [384]],                                                # 8
            [-1,     1,      Conv2d,               [768, 2, 2]],                                          # 9
            [-1,     1,      ConvNeXtBlock,        [768, 1e-6, 0.4, 3,  33, 36]],                         # 10
            [-1,     1,      AdaptiveAvgPool2d,    [1]],                                                  # 11
        ],
        "head"    : [
            [-1,     1,      ConvNeXtClassifier,   [768, LayerNorm2d]],                                   # 12
        ]
    },
    "convnext-large": {
        "name"    : "convnext-large",
        "channels": 3,
        "backbone": [
            # [from, number, module,               args(out_channels, ...)]
            [-1,     1,      Conv2dNormActivation, [192, 4, 4, 0, 1, LayerNorm2d, None, 1, True, True]],  # 0 total_stage_blocks = 36
            [-1,     1,      ConvNeXtBlock,        [192, 1e-6, 0.5, 3,  0,  36]],                         # 1
            [-1,     1,      LayerNorm2d,          [192]],                                                # 2
            [-1,     1,      Conv2d,               [384, 2, 2]],                                          # 3
            [-1,     1,      ConvNeXtBlock,        [384, 1e-6, 0.5, 3,  3,  36]],                         # 4
            [-1,     1,      LayerNorm2d,          [384]],                                                # 5
            [-1,     1,      Conv2d,               [768, 2, 2]],                                          # 6
            [-1,     1,      ConvNeXtBlock,        [768, 1e-6, 0.5, 27, 6,  36]],                         # 7
            [-1,     1,      LayerNorm2d,          [768]],                                                # 8
            [-1,     1,      Conv2d,               [1536, 2, 2]],                                         # 9
            [-1,     1,      ConvNeXtBlock,        [1536, 1e-6, 0.5, 3,  33, 36]],                        # 10
            [-1,     1,      AdaptiveAvgPool2d,    [1]],                                                  # 11
        ],
        "head"    : [
            [-1,     1,      ConvNeXtClassifier,   [1536, LayerNorm2d]],                                  # 12
        ]
    },
}


@MODELS.register(name="convnext")
class ConvNeXt(ImageClassificationModel):
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
        "convnext-base-imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
            filename    = "convnext-base-imagenet.pth",
            num_classes = 1000,
        ),
        "convnext-tiny-imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
            filename    = "convnext-tiny-imagenet.pth",
            num_classes = 1000,
        ),
        "convnext-small-imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/convnext_small-0c510722.pth",
            filename    = "convnext-small-imagenet.pth",
            num_classes = 1000,
        ),
        "convnext-large-imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
            filename    = "convnext-large-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "convnext-base.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "convnext",
        fullname   : str          | None = "convnext-base",
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
            pretrained  = ConvNeXt.init_pretrained(pretrained),
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
                nn.init.trunc_normal_(m.conv.weight, std=0.02)
                if m.conv.bias is not None:
                    nn.init.zeros_(m.conv.bias)
            else:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        elif classname.find("Linear") != -1:
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
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
                if "features.0"     in k:
                    k = k.replace("features.",    "")
                elif "features.1"   in k:
                    k = k.replace("features.1",   "1.block")
                elif "features.2.0" in k:
                    k = k.replace("features.2.0", "2")
                elif "features.2.1" in k:
                    k = k.replace("features.2.1", "3")
                elif "features.3"   in k:
                    k = k.replace("features.3",   "4.block")
                elif "features.4.0" in k:
                    k = k.replace("features.4.0", "5")
                elif "features.4.1" in k:
                    k = k.replace("features.4.1", "6")
                elif "features.5"   in k:
                    k = k.replace("features.5",   "7.block")
                elif "features.6.0" in k:
                    k = k.replace("features.6.0", "8")
                elif "features.6.1" in k:
                    k = k.replace("features.6.1", "9")
                elif "features.7"   in k:
                    k = k.replace("features.7",   "10.block")
                elif "classifier"   in k:
                    continue
                model_state_dict[k] = v
            
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["12.norm.bias"]     = state_dict["classifier.0.bias"]
                model_state_dict["12.norm.weight"]   = state_dict["classifier.0.weight"]
                model_state_dict["12.linear.bias"]   = state_dict["classifier.2.bias"]
                model_state_dict["12.linear.weight"] = state_dict["classifier.2.weight"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()
