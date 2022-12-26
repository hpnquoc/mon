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
    "shufflenet-v2-x0.5": {
        "name"    : "shufflenet-v2-x0.5",
        "channels": 3,
        "backbone": [
            # [from, number, module,           args(out_channels, ...)]
            # Stage 1
            [-1,     1,      Conv2d,           [24, 3, 2, 1, 1, 1, False]],    # 0
            [-1,     1,      BatchNorm2d,      [24]],                          # 1
            [-1,     1,      ReLU,             [True]],                        # 2
            [-1,     1,      MaxPool2d,        [3, 2, 1]],                     # 3
            # Stage 2                                                          
            [-1,     1,      InvertedResidual, [48,  2]],                      # 4
            [-1,     1,      InvertedResidual, [48,  1]],                      # 5
            [-1,     1,      InvertedResidual, [48,  1]],                      # 6
            [-1,     1,      InvertedResidual, [48,  1]],                      # 7
            # Stage 3
            [-1,     1,      InvertedResidual, [96,  2]],                      # 8
            [-1,     1,      InvertedResidual, [96,  1]],                      # 9
            [-1,     1,      InvertedResidual, [96,  1]],                      # 10
            [-1,     1,      InvertedResidual, [96,  1]],                      # 11
            [-1,     1,      InvertedResidual, [96,  1]],                      # 12
            [-1,     1,      InvertedResidual, [96,  1]],                      # 13
            [-1,     1,      InvertedResidual, [96,  1]],                      # 14
            [-1,     1,      InvertedResidual, [96,  1]],                      # 15
            # Stage 4                                                          
            [-1,     1,      InvertedResidual, [192, 2]],                      # 16
            [-1,     1,      InvertedResidual, [192, 1]],                      # 17
            [-1,     1,      InvertedResidual, [192, 1]],                      # 18
            [-1,     1,      InvertedResidual, [192, 1]],                      # 19
            # Stage 5
            [-1,     1,      Conv2d,           [1024, 1, 1, 0, 1, 1, False]],  # 20
            [-1,     1,      BatchNorm2d,      [1024]],                        # 21
            [-1,     1,      ReLU,             [True]],                        # 22
        ],
        "head"    : [
            [-1,     1,      ShuffleNetV2Classifier, [1024]],                  # 23
        ]
    },
    "shufflenet-v2-x1.0": {
        "name"    : "shufflenet-v2-x1.0",
        "channels": 3,
        "backbone": [
            # [from, number, module,           args(out_channels, ...)]
            # Stage 1
            [-1,     1,      Conv2d,           [24, 3, 2, 1, 1, 1, False]],    # 0
            [-1,     1,      BatchNorm2d,      [24]],                          # 1
            [-1,     1,      ReLU,             [True]],                        # 2
            [-1,     1,      MaxPool2d,        [3, 2, 1]],                     # 3
            # Stage 2
            [-1,     1,      InvertedResidual, [116, 2]],                      # 4
            [-1,     1,      InvertedResidual, [116, 1]],                      # 5
            [-1,     1,      InvertedResidual, [116, 1]],                      # 6
            [-1,     1,      InvertedResidual, [116, 1]],                      # 7
            # Stage 3
            [-1,     1,      InvertedResidual, [232, 2]],                      # 8
            [-1,     1,      InvertedResidual, [232, 1]],                      # 9
            [-1,     1,      InvertedResidual, [232, 1]],                      # 10
            [-1,     1,      InvertedResidual, [232, 1]],                      # 11
            [-1,     1,      InvertedResidual, [232, 1]],                      # 12
            [-1,     1,      InvertedResidual, [232, 1]],                      # 13
            [-1,     1,      InvertedResidual, [232, 1]],                      # 14
            [-1,     1,      InvertedResidual, [232, 1]],                      # 15
            # Stage 4
            [-1,     1,      InvertedResidual, [464, 2]],                      # 16
            [-1,     1,      InvertedResidual, [464, 1]],                      # 17
            [-1,     1,      InvertedResidual, [464, 1]],                      # 18
            [-1,     1,      InvertedResidual, [464, 1]],                      # 19
            # Stage 5
            [-1,     1,      Conv2d,           [1024, 1, 1, 0, 1, 1, False]],  # 20
            [-1,     1,      BatchNorm2d,      [1024]],                        # 21
            [-1,     1,      ReLU,             [True]],                        # 22
        ],
        "head"    : [
            [-1,     1,      ShuffleNetV2Classifier, [1024]],                  # 23
        ]
    },
    "shufflenet-v2-x1.5": {
        "name"    : "shufflenet-v2-x1.5",
        "channels": 3,
        "backbone": [
            # [from, number, module,           args(out_channels, ...)]
            # Stage 1
            [-1,     1,      Conv2d,           [24, 3, 2, 1, 1, 1, False]],    # 0
            [-1,     1,      BatchNorm2d,      [24]],                          # 1
            [-1,     1,      ReLU,             [True]],                        # 2
            [-1,     1,      MaxPool2d,        [3, 2, 1]],                     # 3
            # Stage 2
            [-1,     1,      InvertedResidual, [176, 2]],                      # 4
            [-1,     1,      InvertedResidual, [176, 1]],                      # 5
            [-1,     1,      InvertedResidual, [176, 1]],                      # 6
            [-1,     1,      InvertedResidual, [176, 1]],                      # 7
            # Stage 3
            [-1,     1,      InvertedResidual, [352, 2]],                      # 8
            [-1,     1,      InvertedResidual, [352, 1]],                      # 9
            [-1,     1,      InvertedResidual, [352, 1]],                      # 10
            [-1,     1,      InvertedResidual, [352, 1]],                      # 11
            [-1,     1,      InvertedResidual, [352, 1]],                      # 12
            [-1,     1,      InvertedResidual, [352, 1]],                      # 13
            [-1,     1,      InvertedResidual, [352, 1]],                      # 14
            [-1,     1,      InvertedResidual, [352, 1]],                      # 15
            # Stage 4
            [-1,     1,      InvertedResidual, [704, 2]],                      # 16
            [-1,     1,      InvertedResidual, [704, 1]],                      # 17
            [-1,     1,      InvertedResidual, [704, 1]],                      # 18
            [-1,     1,      InvertedResidual, [704, 1]],                      # 19
            # Stage 5
            [-1,     1,      Conv2d,           [1024, 1, 1, 0, 1, 1, False]],  # 20
            [-1,     1,      BatchNorm2d,      [1024]],                        # 21
            [-1,     1,      ReLU,             [True]],                        # 22
        ],
        "head"    : [
            [-1,     1,      ShuffleNetV2Classifier, [1024]],                  # 23
        ]
    },
    "shufflenet-v2-x2.0": {
        "name"    : "shufflenet-v2-x2.0",
        "channels": 3,
        "backbone": [
            # [from, number, module,           args(out_channels, ...)]
            # Stage 1
            [-1,     1,      Conv2d,           [24, 3, 2, 1, 1, 1, False]],    # 0
            [-1,     1,      BatchNorm2d,      [24]],                          # 1
            [-1,     1,      ReLU,             [True]],                        # 2
            [-1,     1,      MaxPool2d,        [3, 2, 1]],                     # 3
            # Stage 2
            [-1,     1,      InvertedResidual, [244, 2]],                      # 4
            [-1,     1,      InvertedResidual, [244, 1]],                      # 5
            [-1,     1,      InvertedResidual, [244, 1]],                      # 6
            [-1,     1,      InvertedResidual, [244, 1]],                      # 7
            # Stage 3
            [-1,     1,      InvertedResidual, [488, 2]],                      # 8
            [-1,     1,      InvertedResidual, [488, 1]],                      # 9
            [-1,     1,      InvertedResidual, [488, 1]],                      # 10
            [-1,     1,      InvertedResidual, [488, 1]],                      # 11
            [-1,     1,      InvertedResidual, [488, 1]],                      # 12
            [-1,     1,      InvertedResidual, [488, 1]],                      # 13
            [-1,     1,      InvertedResidual, [488, 1]],                      # 14
            [-1,     1,      InvertedResidual, [488, 1]],                      # 15
            # Stage 4
            [-1,     1,      InvertedResidual, [976, 2]],                      # 16
            [-1,     1,      InvertedResidual, [976, 1]],                      # 17
            [-1,     1,      InvertedResidual, [976, 1]],                      # 18
            [-1,     1,      InvertedResidual, [976, 1]],                      # 19
            # Stage 5
            [-1,     1,      Conv2d,           [2048, 1, 1, 0, 1, 1, False]],  # 20
            [-1,     1,      BatchNorm2d,      [2048]],                        # 21
            [-1,     1,      ReLU,             [True]],                        # 22
        ],
        "head"    : [
            [-1,     1,      ShuffleNetV2Classifier, [2048]],                  # 23
        ]
    },
}


@MODELS.register(name="shufflenet-v2")
class ShuffleNetV2(ImageClassificationModel):
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
        "shufflenet-v2-x0.5-imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
            filename    = "shufflenet-v2-x0.5-imagenet.pth",
            num_classes = 1000,
        ),
        "shufflenet-v2-x1.0-imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
            filename    = "shufflenet-v2-x1.0-imagenet.pth",
            num_classes = 1000,
        ),
        "shufflenet-v2-x1.5-imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth",
            filename    = "shufflenet-v2-x1.5-imagenet.pth",
            num_classes = 1000,
        ),
        "shufflenet-v2-x2.0-imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth",
            filename    = "shufflenet-v2-x2.0-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "shufflenet-v2-x0.5.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "shufflenet-v2",
        fullname   : str          | None = "shufflenet-v2-x0.5",
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
        cfg, variant = parse_cfg_variant(
            cfg     = cfg,
            cfgs    = cfgs,
            cfg_dir = CFG_DIR,
            to_dict = True
        )
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
            pretrained  = ShuffleNetV2.init_pretrained(pretrained),
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
        pass
    
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
                if "stage2.0" in k:
                    k = k.replace("stage2.0", "4")
                elif "stage2.1" in k:
                    k = k.replace("stage2.1", "5")
                elif "stage2.2" in k:
                    k = k.replace("stage2.2", "6")
                elif "stage2.3" in k:
                    k = k.replace("stage2.3", "7")
                elif "stage3.0" in k:
                    k = k.replace("stage3.0", "8")
                elif "stage3.1" in k:
                    k = k.replace("stage3.1", "9")
                elif "stage3.2" in k:
                    k = k.replace("stage3.2", "10")
                elif "stage3.3" in k:
                    k = k.replace("stage3.3", "11")
                elif "stage3.4" in k:
                    k = k.replace("stage3.4", "12")
                elif "stage3.5" in k:
                    k = k.replace("stage3.5", "13")
                elif "stage3.6" in k:
                    k = k.replace("stage3.6", "14")
                elif "stage3.7" in k:
                    k = k.replace("stage3.7", "15")
                elif "stage4.0" in k:
                    k = k.replace("stage4.0", "16")
                elif "stage4.1" in k:
                    k = k.replace("stage4.1", "17")
                elif "stage4.2" in k:
                    k = k.replace("stage4.2", "18")
                elif "stage4.3" in k:
                    k = k.replace("stage4.3", "19")
                elif "stage5.0" in k:
                    k = k.replace("stage5.0", "20")
                elif "stage5.1" in k:
                    k = k.replace("stage5.1", "21")
                else:
                    continue
                model_state_dict[k] = v
            model_state_dict["0.weight"]       = state_dict["conv1.0.weight"]
            model_state_dict["1.bias"]         = state_dict["conv1.1.bias"]
            model_state_dict["1.running_mean"] = state_dict["conv1.1.running_mean"]
            model_state_dict["1.running_var"]  = state_dict["conv1.1.running_var"]
            model_state_dict["1.weight"]       = state_dict["conv1.1.weight"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["23.linear.bias"]   = state_dict["fc.bias"]
                model_state_dict["23.linear.weight"] = state_dict["fc.weight"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()
