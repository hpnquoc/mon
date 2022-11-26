#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from one.nn import *

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Model ------------------------------------------------------------------

cfgs = {
    # Fused
    "mobileone-s0": {
        "name"    : "mobileone-s0",
        "channels": 3,
        "backbone": [
            # [from, number, module,              args(out_channels, ...)]
            [-1, 1, MobileOneConv2d, [48, 3, 2, 1, 1, 1, True, "zeros", None, None, True, False, 1]],  # 0
            [-1,     1,      MobileOneStage,      [48,   2,  0, True, 4]],                                          # 1
            [-1,     1,      MobileOneStage,      [128,  8,  0, True, 4]],                                          # 2
            [-1,     1,      MobileOneStage,      [256,  10, 0, True, 4]],                                          # 3
            [-1,     1,      MobileOneStage,      [1024, 1,  0, True, 4]],                                          # 4
        ],
        "head"    : [
            [-1,     1,      MobileOneClassifier, [1024]],                                                          # 5
        ]
    },
    "mobileone-s1": {
        "name"    : "mobileone-s1",
        "channels": 3,
        "backbone": [
            # [from, number, module,              args(out_channels, ...)]
            [-1, 1, MobileOneConv2d, [64, 3, 2, 1, 1, 1, True, "zeros", None, None, True, False, 1]],  # 0
            [-1,     1,      MobileOneStage,      [96,   2,  0, True, 1]],                                          # 1
            [-1,     1,      MobileOneStage,      [192,  8,  0, True, 1]],                                          # 2
            [-1,     1,      MobileOneStage,      [512,  10, 0, True, 1]],                                          # 3
            [-1,     1,      MobileOneStage,      [1280, 1,  0, True, 1]],                                          # 4
        ],
        "head"    : [
            [-1,     1,      MobileOneClassifier, [1280]],                                                          # 5
        ]
    },
    "mobileone-s2": {
        "name"    : "mobileone-s2",
        "channels": 3,
        "backbone": [
            # [from, number, module,              args(out_channels, ...)]
            [-1, 1, MobileOneConv2d, [64, 3, 2, 1, 1, 1, True, "zeros", None, None, True, False, 1]],  # 0
            [-1,     1,      MobileOneStage,      [96,   2,  0, True, 1]],                                          # 1
            [-1,     1,      MobileOneStage,      [256,  8,  0, True, 1]],                                          # 2
            [-1,     1,      MobileOneStage,      [640,  10, 0, True, 1]],                                          # 3
            [-1,     1,      MobileOneStage,      [2048, 1,  0, True, 1]],                                          # 4
        ],
        "head"    : [
            [-1,     1,      MobileOneClassifier, [2048]],                                                          # 5
        ]
    },
    "mobileone-s3": {
        "name"    : "mobileone-s3",
        "channels": 3,
        "backbone": [
            # [from, number, module,              args(out_channels, ...)]
            [-1, 1, MobileOneConv2d, [64, 3, 2, 1, 1, 1, True, "zeros", None,
                                      None, True, False, 1]],  # 0
            [-1,     1,      MobileOneStage,      [128,  2,  0, True, 1]],                                          # 1
            [-1,     1,      MobileOneStage,      [320,  8,  0, True, 1]],                                          # 2
            [-1,     1,      MobileOneStage,      [768,  10, 0, True, 1]],                                          # 3
            [-1,     1,      MobileOneStage,      [2048, 1,  0, True, 1]],                                          # 4
        ],
        "head"    : [
            [-1,     1,      MobileOneClassifier, [2048]],                                                          # 5
        ]
    },
    "mobileone-s4": {
        "name"    : "mobileone-s4",
        "channels": 3,
        "backbone": [
            # [from, number, module,              args(out_channels, ...)]
            [-1, 1, MobileOneConv2d, [64, 3, 2, 1, 1, 1, True, "zeros", None, None, True, True, 1]],   # 0
            [-1,     1,      MobileOneStage,      [192,  2,  0, True, 1]],                                          # 1
            [-1,     1,      MobileOneStage,      [448,  8,  0, True, 1]],                                          # 2
            [-1,     1,      MobileOneStage,      [896,  10, 5, True, 1]],                                          # 3
            [-1,     1,      MobileOneStage,      [2048, 1,  1, True, 1]],                                          # 4
        ],
        "head"    : [
            [-1,     1,      MobileOneClassifier, [2048]],                                                          # 5
        ]
    },
    
    # Unfused
    "mobileone-s0-unfused": {
        "name"    : "mobileone-s0-unfused",
        "channels": 3,
        "backbone": [
            # [from, number, module,              args(out_channels, ...)]
            [-1, 1, MobileOneConv2d, [48, 3, 2, 1, 1, 1, True, "zeros", None, None, False, False, 1]],  # 0
            [-1,     1,      MobileOneStage,      [48,   2,  0, False, 4]],                                          # 1
            [-1,     1,      MobileOneStage,      [128,  8,  0, False, 4]],                                          # 2
            [-1,     1,      MobileOneStage,      [256,  10, 0, False, 4]],                                          # 3
            [-1,     1,      MobileOneStage,      [1024, 1,  0, False, 4]],                                          # 4
        ],                                                                                                           
        "head"    : [
            [-1,     1,      MobileOneClassifier, [1024]],                                                           # 5
        ]
    },
    "mobileone-s1-unfused": {
        "name"    : "mobileone-s1-unfused",
        "channels": 3,
        "backbone": [
            # [from, number, module,              args(out_channels, ...)]
            [-1, 1, MobileOneConv2d, [64, 3, 2, 1, 1, 1, True, "zeros", None,
                                      None, False, False, 1]],  # 0
            [-1,     1,      MobileOneStage,      [96,   2,  0, False, 1]],                                          # 1
            [-1,     1,      MobileOneStage,      [192,  8,  0, False, 1]],                                          # 2
            [-1,     1,      MobileOneStage,      [512,  10, 0, False, 1]],                                          # 3
            [-1,     1,      MobileOneStage,      [1280, 1,  0, False, 1]],                                          # 4
        ],
        "head"    : [
            [-1,     1,      MobileOneClassifier, [1280]],                                                           # 5
        ]
    },
    "mobileone-s2-unfused": {
        "name"    : "mobileone-s2-unfused",
        "channels": 3,
        "backbone": [
            # [from, number, module,              args(out_channels, ...)]
            [-1, 1, MobileOneConv2d, [64, 3, 2, 1, 1, 1, True, "zeros", None, None, False, False, 1]],  # 0
            [-1,     1,      MobileOneStage,      [96,   2,  0, False, 1]],                                          # 1
            [-1,     1,      MobileOneStage,      [256,  8,  0, False, 1]],                                          # 2
            [-1,     1,      MobileOneStage,      [640,  10, 0, False, 1]],                                          # 3
            [-1,     1,      MobileOneStage,      [2048, 1,  0, False, 1]],                                          # 4
        ],
        "head"    : [
            [-1,     1,      MobileOneClassifier, [2048]],                                                           # 5
        ]
    },
    "mobileone-s3-unfused": {
        "name"    : "mobileone-s3-unfused",
        "channels": 3,
        "backbone": [
            # [from, number, module,              args(out_channels, ...)]
            [-1, 1, MobileOneConv2d, [64, 3, 2, 1, 1, 1, True, "zeros", None, None, False, False, 1]],  # 0
            [-1,     1,      MobileOneStage,      [128,  2,  0, False, 1]],                                          # 1
            [-1,     1,      MobileOneStage,      [320,  8,  0, False, 1]],                                          # 2
            [-1,     1,      MobileOneStage,      [768,  10, 0, False, 1]],                                          # 3
            [-1,     1,      MobileOneStage,      [2048, 1,  0, False, 1]],                                          # 4
        ],
        "head"    : [
            [-1,     1,      MobileOneClassifier, [2048]],                                                           # 5
        ]
    },
    "mobileone-s4-unfused": {
        "name"    : "mobileone-s4-unfused",
        "channels": 3,
        "backbone": [
            # [from, number, module,              args(out_channels, ...)]
            [-1, 1, MobileOneConv2d, [64, 3, 2, 1, 1, 1, True, "zeros", None, None, False, True, 1]],   # 0
            [-1,     1,      MobileOneStage,      [192,  2,  0, False, 1]],                                          # 1
            [-1,     1,      MobileOneStage,      [448,  8,  0, False, 1]],                                          # 2
            [-1,     1,      MobileOneStage,      [896,  10, 5, False, 1]],                                          # 3
            [-1,     1,      MobileOneStage,      [2048, 1,  1, False, 1]],                                          # 4
        ],
        "head"    : [
            [-1,     1,      MobileOneClassifier, [2048]],                                                           # 5
        ]
    },
}


@MODELS.register(name="mobileone")
class MobileOne(ImageClassificationModel):
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
        "mobileone-s0-imagenet": dict(
            name        = "imagenet",
            path        = "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s0.pth.tar",
            filename    = "mobileone-s0-imagenet.pth.tar",
            num_classes = 1000,
        ),
        "mobileone-s1-imagenet": dict(
            name        = "imagenet",
            path        = "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s1.pth.tar",
            filename    = "mobileone-s1-imagenet.pth.tar",
            num_classes = 1000,
        ),
        "mobileone-s2-imagenet": dict(
            name        = "imagenet",
            path        = "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s2.pth.tar",
            filename    = "mobileone-s2-imagenet.pth.tar",
            num_classes = 1000,
        ),
        "mobileone-s3-imagenet": dict(
            name        = "imagenet",
            path        = "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s3.pth.tar",
            filename    = "mobileone-s3-imagenet.pth.tar",
            num_classes = 1000,
        ),
        "mobileone-s4-imagenet": dict(
            name        = "imagenet",
            path        = "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s4.pth.tar",
            filename    = "mobileone-s4-imagenet.pth.tar",
            num_classes = 1000,
        ),
        "mobileone-s0-unfused-imagenet": dict(
            name        = "imagenet",
            path        = "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s0_unfused.pth.tar",
            filename    = "mobileone-s0-unfused-imagenet.pth.tar",
            num_classes = 1000,
        ),
        "mobileone-s1-unfused-imagenet": dict(
            name        = "imagenet",
            path        = "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s1_unfused.pth.tar",
            filename    = "mobileone-s1-unfused-imagenet.pth.tar",
            num_classes = 1000,
        ),
        "mobileone-s2-unfused-imagenet": dict(
            name        = "imagenet",
            path        = "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s2_unfused.pth.tar",
            filename    = "mobileone-s2-unfused-imagenet.pth.tar",
            num_classes = 1000,
        ),
        "mobileone-s3-unfused-imagenet": dict(
            name        = "imagenet",
            path        = "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s3_unfused.pth.tar",
            filename    = "mobileone-s3-unfused-imagenet.pth.tar",
            num_classes = 1000,
        ),
        "mobileone-s4-unfused-imagenet": dict(
            name        = "imagenet",
            path        = "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s4_unfused.pth.tar",
            filename    = "mobileone-s4-unfused-imagenet.pth.tar",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "mobileone-s0.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "mobileone",
        fullname   : str          | None = "mobileone-s0",
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
            pretrained  = MobileOne.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def reparameterize(self):
        for module in self.model.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()
    
    def init_weights(self, m: Module):
        pass
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained) and self.pretrained["name"] in ["imagenet", "imagenet-unfused"]:
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
                if "linear" in k:
                    k = k.replace("linear.", "5.fc.")
                    if self.pretrained["num_classes"] == self.num_classes:
                        model_state_dict[k] = v
                elif "stage0" in k:
                    k = k.replace("stage", "")
                    model_state_dict[k] = v
                else:
                    k      = k.replace("stage", "")
                    splits = k.split(".", 1)
                    k      = f"{splits[0]}.convs.{splits[1]}"
                    if "reduce" in k:
                        k = k.replace("reduce", "excitation.0")
                    elif "expand" in k:
                        k = k.replace("expand", "excitation.2")
                    model_state_dict[k] = v
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()
