#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ultralytics SAM model for segmentation.

References:
    - Code: https://github.com/ultralytics/ultralytics
"""

__all__ = [
    "SAM",
    "SAM2",
    "SAM2_B",
    "SAM2_L",
    "SAM2_S",
    "SAM2_T",
    "SAM_B",
    "SAM_L",
]

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task, ZOO_DIR
from mon.core import pathlib
from ultralytics import SAM as SAM_

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- SAM -----
class SAM(SAM_, nn.ModelMixin):
    """Ultralytics SAM model for segmentation.
    
    References:
        - Code: https://github.com/ultralytics/ultralytics
    """
    
    arch     : str          = "sam"
    name     : str          = "sam"
    tasks    : list[Task]   = [Task.SEGMENT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: str = "sa1b", *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights)
        super().__init__(model=path, *args, **kwargs)


@MODELS.register(name="sam_b", arch="sam")
class SAM_B(SAM, nn.ModelMixin):
    
    arch: str  = "sam"
    name: str  = "sam_b"
    zoo : dict = box.Box({
        "sa1b": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/sam/sam_b/sa1b/sam_b_sa1b.pt",
            "num_classes": None,
        },
    })


@MODELS.register(name="sam_l", arch="sam")
class SAM_L(SAM, nn.ModelMixin):
    
    arch: str  = "sam"
    name: str  = "sam_l"
    zoo : dict = box.Box({
        "sa1b": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/sam/sam_l/sa1b/sam_l_sa1b.pt",
            "num_classes": None,
        },
    })


# ----- SAM2 -----
class SAM2(SAM_, nn.ModelMixin):
    
    arch: str  = "sam2"
    name: str  = "sam2"
    zoo : dict = box.Box()
    
    def __init__(self, weights: str = "sav", *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights)
        super().__init__(model=path, *args, **kwargs)


@MODELS.register(name="sam2_t", arch="sam2")
class SAM2_T(SAM, nn.ModelMixin):
    
    arch: str  = "sam2"
    name: str  = "sam2_t"
    zoo : dict = box.Box({
        "sa1b": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/sam2/sam2.1_t/sav/sam2.1_t_sav.pt",
            "num_classes": None,
        },
    })
    

@MODELS.register(name="sam2_s", arch="sam2")
class SAM2_S(SAM, nn.ModelMixin):
    
    arch: str  = "sam2"
    name: str  = "sam2_s"
    zoo : dict = box.Box({
        "sa1b": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/sam2/sam2.1_s/sav/sam2.1_s_sav.pt",
            "num_classes": None,
        },
    })
    
    
@MODELS.register(name="sam2_b", arch="sam2")
class SAM2_B(SAM, nn.ModelMixin):
    
    arch: str  = "sam2"
    name: str  = "sam2_b"
    zoo : dict = box.Box({
        "sa1b": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/sam2/sam2.1_b/sav/sam2.1_b_sav.pt",
            "num_classes": None,
        },
    })


@MODELS.register(name="sam2_l", arch="sam2")
class SAM2_L(SAM, nn.ModelMixin):
    
    arch: str  = "sam2"
    name: str  = "sam2_l"
    zoo : dict = box.Box({
        "sa1b": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/sam2/sam2.1_l/sav/sam2.1_l_sav.pt",
            "num_classes": None,
        },
    })
