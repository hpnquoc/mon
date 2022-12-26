#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Zero-ADCE
"""

from __future__ import annotations

from one.nn import *

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Loss -------------------------------------------------------------------

class CombinedLoss(BaseLoss):
    """
    Loss = SpatialConsistencyLoss
          + ExposureControlLoss
          + ColorConstancyLoss
          + IlluminationSmoothnessLoss
          + ChannelConsistencyLoss
          + EdgeLoss
    """
    
    def __init__(
        self,
        spa_weight    : Floats = 1.0,
        exp_patch_size: int    = 16,
        exp_mean_val  : float  = 0.6,
        exp_weight    : Floats = 10.0,
        col_weight    : Floats = 5.0,
        tv_weight     : Floats = 200.0,
        channel_weight: Floats = 5.0,
        edge_weight   : Floats = 5.0,
        reduction     : str    = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.name     = "combined_loss"
        self.loss_spa = SpatialConsistencyLoss(
            weight    = spa_weight,
            reduction = reduction,
        )
        self.loss_exp = ExposureControlLoss(
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
            weight     = exp_weight,
            reduction  = reduction,
        )
        self.loss_col = ColorConstancyLoss(
            weight    = col_weight,
            reduction = reduction,
        )
        self.loss_tv  = IlluminationSmoothnessLoss(
            weight    = tv_weight,
            reduction = reduction,
        )
        self.loss_channel = ChannelConsistencyLoss(
            weight    = channel_weight,
            reduction = reduction,
        )
        self.loss_edge = EdgeLoss(
            weight    = edge_weight,
            reduction = reduction,
        )
     
    def forward(self, input: Tensors, target: Sequence[Tensor], **_) -> Tensor:
        if isinstance(target, Sequence):
            a       = target[-2]
            enhance = target[-1]
        else:
            raise TypeError()
        
        loss_spa     = self.loss_spa(input=enhance, target=input)
        loss_exp     = self.loss_exp(input=enhance)
        loss_col     = self.loss_col(input=enhance)
        loss_tv      = self.loss_tv(input=a)
        loss_channel = self.loss_channel(input=enhance, target=input)
        loss_edge    = self.loss_edge(input=enhance, target=input)
        return loss_spa + loss_exp + loss_col + loss_tv + loss_channel + loss_edge


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "zero-adce-a": {
        "name"    : "zero-adce-a",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                     # 0  (x)
            [-1,      1,      ADCE,     [3, 32, partial(ABSConv2dS, act2=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                 # 2
        ]
    },
    "zero-adce-b": {
        "name"    : "zero-adce-b",
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                                 # 0  (x)
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                             # 2  (x1)
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                             # 4  (x2)
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                             # 6  (x3)
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                             # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                                 # 9
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None,  HalfInstanceNorm2d]], # 10
            [-1,      1,      ReLU,       [True]],                                                                             # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                                 # 12
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None,  HalfInstanceNorm2d]], # 13
            [-1,      1,      ReLU,       [True]],                                                                             # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                                 # 15
            [-1,      1,      Conv2d,     [3,  3, 1, 1]],                                                                      # 16 (a)
            [-1,      1,      Tanh,       []],                                                                                 # 17
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                               # 18
        ]
    },
    "zero-adce-c": {
        "name"    : "zero-adce-c",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                                              # 0  (x)
            [-1,      1,      ADCE,     [3, 32, partial(ABSConv2dS, act1=HalfInstanceNorm2d, act2=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                          # 2
        ],
    },  # old s5
    "zero-adce-d": {
        "name"    : "zero-adce-d",
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                                              # 0  (x)
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 1
            [-1,      1,      ReLU,       [True]],                                                                                          # 2  (x1)
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 3
            [-1,      1,      ReLU,       [True]],                                                                                          # 4  (x2)
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 5
            [-1,      1,      ReLU,       [True]],                                                                                          # 6  (x3)
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 7
            [-1,      1,      ReLU,       [True]],                                                                                          # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                                              # 9
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 10
            [-1,      1,      ReLU,       [True]],                                                                                          # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                                              # 12
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 13
            [-1,      1,      ReLU,       [True]],                                                                                          # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                                              # 15
            [-1,      1,      Conv2d,     [3,  3, 1, 1]],                                                                                   # 16 (a)
            [-1,      1,      Tanh,       []],                                                                                              # 17
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                                            # 18
        ]
    },
    "zero-adce-e": {
        "name"    : "zero-adce-e",
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                                               # 0  (x)
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                                           # 2  (x1)
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                                           # 4  (x2)
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                                           # 6  (x3)
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                                           # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                                               # 9
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 10
            [-1,      1,      ReLU,       [True]],                                                                                           # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                                               # 12
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 13
            [-1,      1,      ReLU,       [True]],                                                                                           # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                                               # 15
            [-1,      1,      ABSConv2dS, [3,  3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 16 (a)
            [-1,      1,      Tanh,       []],                                                                                               # 17
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                                             # 18
        ]
    },  # old s4
    
    "zero-adce-a-large": {
        "name"    : "zero-adce-a-large",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                     # 0  (x)
            [-1,      1,      ADCE,     [3, 64, partial(ABSConv2dS, act2=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                 # 2
        ]
    },
    "zero-adce-b-large": {
        "name"    : "zero-adce-b-large",
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                                 # 0  (x)
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                             # 2  (x1)
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                             # 4  (x2)
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                             # 6  (x3)
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                             # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                                 # 9
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None,  HalfInstanceNorm2d]], # 10
            [-1,      1,      ReLU,       [True]],                                                                             # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                                 # 12
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None,  HalfInstanceNorm2d]], # 13
            [-1,      1,      ReLU,       [True]],                                                                             # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                                 # 15
            [-1,      1,      Conv2d,     [3,  3, 1, 1]],                                                                      # 16 (a)
            [-1,      1,      Tanh,       []],                                                                                 # 17
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                               # 18
        ]
    },
    "zero-adce-c-large": {
        "name"    : "zero-adce-c-large",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                                              # 0  (x)
            [-1,      1,      ADCE,     [3, 64, partial(ABSConv2dS, act1=HalfInstanceNorm2d, act2=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                          # 2
        ],
    },
    "zero-adce-d-large": {
        "name"    : "zero-adce-d-large",
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                                              # 0  (x)
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 1
            [-1,      1,      ReLU,       [True]],                                                                                          # 2  (x1)
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 3
            [-1,      1,      ReLU,       [True]],                                                                                          # 4  (x2)
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 5
            [-1,      1,      ReLU,       [True]],                                                                                          # 6  (x3)
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 7
            [-1,      1,      ReLU,       [True]],                                                                                          # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                                              # 9
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 10
            [-1,      1,      ReLU,       [True]],                                                                                          # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                                              # 12
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 13
            [-1,      1,      ReLU,       [True]],                                                                                          # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                                              # 15
            [-1,      1,      Conv2d,     [3,  3, 1, 1]],                                                                                   # 16 (a)
            [-1,      1,      Tanh,       []],                                                                                              # 17
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                                            # 18
        ]
    },
    "zero-adce-e-large": {
        "name"    : "zero-adce-e-large",
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                                               # 0  (x)
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                                           # 2  (x1)
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                                           # 4  (x2)
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                                           # 6  (x3)
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                                           # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                                               # 9
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 10
            [-1,      1,      ReLU,       [True]],                                                                                           # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                                               # 12
            [-1,      1,      ABSConv2dS, [64, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 13
            [-1,      1,      ReLU,       [True]],                                                                                           # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                                               # 15
            [-1,      1,      ABSConv2dS, [3,  3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 16 (a)
            [-1,      1,      Tanh,       []],                                                                                               # 17
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                                             # 18
        ]
    },
    
    "zero-adce-a-tiny": {
        "name"    : "zero-adce-a-tiny",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                     # 0  (x)
            [-1,      1,      ADCE,     [3, 16, partial(ABSConv2dS, act2=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                 # 2
        ]
    },
    "zero-adce-b-tiny": {
        "name"    : "zero-adce-b-tiny",
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                                 # 0  (x)
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                             # 2  (x1)
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                             # 4  (x2)
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                             # 6  (x3)
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                             # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                                 # 9
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None,  HalfInstanceNorm2d]], # 10
            [-1,      1,      ReLU,       [True]],                                                                             # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                                 # 12
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, None,  HalfInstanceNorm2d]], # 13
            [-1,      1,      ReLU,       [True]],                                                                             # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                                 # 15
            [-1,      1,      Conv2d,     [3,  3, 1, 1]],                                                                      # 16 (a)
            [-1,      1,      Tanh,       []],                                                                                 # 17
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                               # 18
        ]
    },
    "zero-adce-c-tiny": {
        "name"    : "zero-adce-c-tiny",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                                              # 0  (x)
            [-1,      1,      ADCE,     [3, 16, partial(ABSConv2dS, act1=HalfInstanceNorm2d, act2=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                          # 2
        ],
    },
    "zero-adce-d-tiny": {
        "name"    : "zero-adce-d-tiny",
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                                              # 0  (x)
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 1
            [-1,      1,      ReLU,       [True]],                                                                                          # 2  (x1)
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 3
            [-1,      1,      ReLU,       [True]],                                                                                          # 4  (x2)
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 5
            [-1,      1,      ReLU,       [True]],                                                                                          # 6  (x3)
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 7
            [-1,      1,      ReLU,       [True]],                                                                                          # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                                              # 9
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 10
            [-1,      1,      ReLU,       [True]],                                                                                          # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                                              # 12
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]], # 13
            [-1,      1,      ReLU,       [True]],                                                                                          # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                                              # 15
            [-1,      1,      Conv2d,     [3,  3, 1, 1]],                                                                                   # 16 (a)
            [-1,      1,      Tanh,       []],                                                                                              # 17
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                                            # 18
        ]
    },
    "zero-adce-e-tiny": {
        "name"    : "zero-adce-e-tiny",
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                                               # 0  (x)
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                                           # 2  (x1)
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                                           # 4  (x2)
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                                           # 6  (x3)
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                                           # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                                               # 9
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 10
            [-1,      1,      ReLU,       [True]],                                                                                           # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                                               # 12
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 13
            [-1,      1,      ReLU,       [True]],                                                                                           # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                                               # 15
            [-1,      1,      ABSConv2dS, [3,  3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d, HalfInstanceNorm2d]],  # 16 (a)
            [-1,      1,      Tanh,       []],                                                                                               # 17
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                                             # 18
        ]
    },
    
    # EXPERIMENTAL #
    
    "zero-adce-abs1" : {
        "name"    : "zero-adce-abs1",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                     # 0  (x)
            [-1,      1,      ADCE,     [3, 32, partial(ABSConv2dS1, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                 # 2
        ]
    },
    "zero-adce-abs2" : {
        "name"    : "zero-adce-abs2",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                     # 0  (x)
            [-1,      1,      ADCE,     [3, 32, partial(ABSConv2dS2, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                 # 2
        ]
    },
    "zero-adce-abs3" : {
        "name"    : "zero-adce-abs3",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                     # 0  (x)
            [-1,      1,      ADCE,     [3, 32, partial(ABSConv2dS3, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                 # 2
        ]
    },
    "zero-adce-abs4" : {
        "name"    : "zero-adce-abs4",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                     # 0  (x)
            [-1,      1,      ADCE,     [3, 32, partial(ABSConv2dS4, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                 # 2
        ]
    },
    "zero-adce-abs5" : {
        "name"    : "zero-adce-abs5",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                     # 0  (x)
            [-1,      1,      ADCE,     [3, 32, partial(ABSConv2dS5, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                 # 2
        ]
    },
    "zero-adce-abs6" : {
        "name"    : "zero-adce-abs6",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                     # 0  (x)
            [-1,      1,      ADCE,     [3, 32, partial(ABSConv2dS6, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                 # 2
        ]
    },
    "zero-adce-abs7" : {
        "name"    : "zero-adce-abs7",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                     # 0  (x)
            [-1,      1,      ADCE,     [3, 32, partial(ABSConv2dS7, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                 # 2
        ]
    },
    "zero-adce-abs8" : {
        "name"    : "zero-adce-abs8",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                     # 0  (x)
            [-1,      1,      ADCE,     [3, 32, partial(ABSConv2dS8, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                 # 2
        ]
    },
    "zero-adce-abs9" : {
        "name"    : "zero-adce-abs9",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                     # 0  (x)
            [-1,      1,      ADCE,     [3, 32, partial(ABSConv2dS9, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                # 2
        ]
    },
    "zero-adce-abs10": {
        "name"    : "zero-adce-abs10",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                      # 0  (x)
            [-1,      1,      ADCE,     [3, 32, partial(ABSConv2dS10, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                  # 2
        ]
    },
    "zero-adce-abs11": {
        "name"    : "zero-adce-abs11",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                      # 0  (x)
            [-1,      1,      ADCE,     [3, 32, partial(ABSConv2dS11, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                  # 2
        ]
    },
    "zero-adce-abs12": {
        "name"    : "zero-adce-abs12",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                      # 0  (x)
            [-1,      1,      ADCE,     [3, 32, partial(ABSConv2dS12, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                  # 2
        ]
    },
    "zero-adce-abs13": {
        "name"    : "zero-adce-abs13",
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                      # 0  (x)
            [-1,      1,      ADCE,     [3, 32, partial(ABSConv2dS13, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head"    : [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                  # 2
        ]
    },
}


@MODELS.register(name="zero-adce")
class ZeroADCE(ImageEnhancementModel):
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
    
    model_zoo = {}
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "zero-adce-a",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "zero-adce",
        fullname   : str          | None = "zero-adce-a",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = CombinedLoss(tv_weight=1600.0),
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
            pretrained  = ZeroADCE.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss or CombinedLoss(tv_weight=1600.0),
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
                m.conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)
            else:
                m.weight.data.normal_(0.0, 0.02)
    
    def forward_loss(
        self,
        input : Tensor,
        target: Tensor,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        """
        Forward pass with loss value. Loss function may require more arguments
        beside the ground-truth and prediction values. For calculating the
        metrics, we only need the final predictions and ground-truth.

        Args:
            input (Tensor): Input of shape [B, C, H, W].
            target (Tensor): Ground-truth of shape [B, C, H, W].
            
        Returns:
            Predictions and loss value.
        """
        pred  = self.forward(input=input, *args, **kwargs)
        loss  = self.loss(input, pred) if self.loss else None
        loss += self.regularization_loss(alpha=0.1)
        return pred[-1], loss
    
    def regularization_loss(self, alpha: float = 0.1):
        loss = 0.0
        for sub_module in self.model.modules():
            if hasattr(sub_module, "regularization_loss"):
                loss += sub_module.regularization_loss()
        return alpha * loss


@MODELS.register(name="zero-adce-jit")
class ZeroADCEJIT(ImageEnhancementModel):
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
    
    model_zoo = {}
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "zero-adce-a",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "zero-adce",
        fullname   : str          | None = "zero-adce-a",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = CombinedLoss(tv_weight=1600.0),
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        super().__init__(
            cfg         = None,
            root        = root,
            project     = project,
            name        = name,
            fullname    = fullname,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = ZeroADCE.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss or CombinedLoss(tv_weight=1600.0),
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
        if cfg in ["zero-adce-a", "zero-adce-b"]:
            conv       = partial(ABSConv2dS, act2=HalfInstanceNorm2d)
            final_conv = Conv2d
        elif cfg in ["zero-adce-c", "zero-adce-d"]:
            conv       = partial(ABSConv2dS, ac1=HalfInstanceNorm2d, act2=HalfInstanceNorm2d)
            final_conv = Conv2d
        elif cfg in ["zero-adce-e"]:
            conv       = partial(ABSConv2dS, ac1=HalfInstanceNorm2d, act2=HalfInstanceNorm2d)
            final_conv = partial(ABSConv2dS, ac1=HalfInstanceNorm2d, act2=HalfInstanceNorm2d)
        else:
            raise ValueError(
                f"`cfg` must be one of: `zero-adce-[a, b, c, d, e]`. "
                f"But got: {cfg}."
            )
           
        self.relu  = ReLU(inplace=True)
        self.conv1 = conv(
            in_channels  = 3,
            out_channels = 32,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            dilation     = 1,
            groups       = 1,
            bias         = True,
            padding_mode = "zeros",
            device       = None,
            dtype        = None,
        )
        self.conv2 = conv(
            in_channels  = 32,
            out_channels = 32,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            dilation     = 1,
            groups       = 1,
            bias         = True,
            padding_mode = "zeros",
            device       = None,
            dtype        = None,
        )
        self.conv3 = ABSConv2dS(
            in_channels  = 32,
            out_channels = 32,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            dilation     = 1,
            groups       = 1,
            bias         = True,
            padding_mode = "zeros",
            device       = None,
            dtype        = None,
        )
        self.conv4 = conv(
            in_channels  = 32,
            out_channels = 32,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            dilation     = 1,
            groups       = 1,
            bias         = True,
            padding_mode = "zeros",
            device       = None,
            dtype        = None,
        )
        self.conv5 = conv(
            in_channels  = 32 * 2,
            out_channels = 32,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            dilation     = 1,
            groups       = 1,
            bias         = True,
            padding_mode = "zeros",
            device       = None,
            dtype        = None,
        )
        self.conv6 = conv(
            in_channels  = 32 * 2,
            out_channels = 32,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            dilation     = 1,
            groups       = 1,
            bias         = True,
            padding_mode = "zeros",
            device       = None,
            dtype        = None,
        )
        self.conv7 = final_conv(
            in_channels  = 32 * 2,
            out_channels = 3,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            dilation     = 1,
            groups       = 1,
            bias         = True,
            padding_mode = "zeros",
            device       = None,
            dtype        = None,
        )
        # Load pretrained
        if self.pretrained:
            self.load_pretrained()
        else:
            self.apply(self.init_weights)
    
    def init_weights(self, m: Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)
            else:
                m.weight.data.normal_(0.0, 0.02)
    
    def forward_loss(
        self,
        input : Tensor,
        target: Tensor,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        """
        Forward pass with loss value. Loss function may require more arguments
        beside the ground-truth and prediction values. For calculating the
        metrics, we only need the final predictions and ground-truth.

        Args:
            input (Tensor): Input of shape [B, C, H, W].
            target (Tensor): Ground-truth of shape [B, C, H, W].
            
        Returns:
            Predictions and loss value.
        """
        pred  = self.forward(input=input, *args, **kwargs)
        loss  = self.loss(input, pred) if self.loss else None
        loss += self.regularization_loss(alpha=0.1)
        return pred[-1], loss
    
    def forward_once(
        self,
        input    : Tensor,
        profile  : bool = False,
        out_index: int = -1,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass once. Implement the logic for a single forward pass.

        Args:
            input (Tensor): Input of shape [B, C, H, W].
            profile (bool): Measure processing time. Defaults to False.
            out_index (int): Return specific layer's output from `out_index`.
                Defaults to -1 means the last layer.
                
        Returns:
            Predictions.
        """
        x  = input
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        a  = torch.tanh(self.conv7(torch.cat([x1, x6], dim=1)))
        x  = self.enhance(x, a)
        return a, x
    
    def enhance(self, x: Tensor, a: Tensor) -> Tensor:
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        return x
    
    def regularization_loss(self, alpha: float = 0.1):
        loss = 0.0
        for conv in [
            self.conv1, self.conv2, self.conv3, self.conv4,
            self.conv5, self.conv6, self.conv7
        ]:
            if hasattr(conv, "regularization_loss"):
                loss += conv.regularization_loss()
        return alpha * loss
