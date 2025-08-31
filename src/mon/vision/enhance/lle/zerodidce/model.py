#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Zero-DiDCE model for low-light image enhancement.

References:
    - Paper: "Rethinking Zero-DCE for Low-Light Image Enhancement,"
      Neural Processing Letters 2024.
    - Code: https://github.com/Wenhui-Luo/Zero-DiDCE
"""

import box
import torch

from mon.constants import MODELS, ZOO_DIR
from mon.core import MLType, ModelMixin, nn, Path, Task
from mon.core.nn import functional as F

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="zerodidce", arch="zerodidce")
class ZeroDiDCE(nn.Module, ModelMixin):
    """Zero-DiDCE model for low-light image enhancement.
    
    References:
        - Paper: "Rethinking Zero-DCE for Low-Light Image Enhancement,"
          Neural Processing Letters 2024.
        - Code: https://github.com/Wenhui-Luo/Zero-DiDCE
    """
    
    arch     : str          = "zerodidce"
    name     : str          = "zerodidce"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box({
        "siceme": {
            "url"        : None,
            "path"       : ZOO_DIR / "vision/enhance/lle/zerodidce/zerodidce/siceme/zerodidce_siceme.pth",
            "num_classes": None,
        },
    })
    
    def __init__(self, weights: Any = None):
        super().__init__()
        in_channels   = 3
        hidden_dim    = 32
        out_channels  = 3
        self.e_conv1  = nn.Conv2d(in_channels,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(hidden_dim,     hidden_dim,   3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(hidden_dim,     hidden_dim,   3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(hidden_dim * 2, out_channels, 3, 1, 1, bias=True)
        self.relu     = nn.ReLU(inplace=True)
        self.maxpool  = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
        # Load weights
        self.load_weights(weights)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xx   = 1 - x
        x1   = self.relu(self.e_conv1(x))
        x2   = self.relu(self.e_conv2(x1))
        x3   = self.relu(self.e_conv3(x2))
        r    =    F.tanh(self.e_conv7(torch.cat([x1, x3], 1)))
        x11  = self.relu(self.e_conv1(xx))
        x21  = self.relu(self.e_conv2(x11))
        x31  = self.relu(self.e_conv3(x21))
        r1   =    F.tanh(self.e_conv7(torch.cat([x11, x31], 1)))
        r    = (r + r1) / 2

        xx1 = torch.mean(x).item()
        n1  = 0.63
        s   = xx1 * xx1
        n3  = -0.79 * s + 0.81 * xx1 + 1.4
        if xx1 < 0.1:
            b = -25 * xx1 + 10
        elif xx1 < 0.45:
            b = 17.14 * s - 15.14 * xx1 + 10
        else:
            b = 5.66 * s - 2.93 * xx1 + 7.2

        b = int(b)
        for i in range(b):
            x = x + r * (torch.pow(x, 2) - x) * ((n1 - torch.mean(x).item()) / (n3 - torch.mean(x).item()))  # + (n1-x)*0.01

        # xxx0 = 1 - x
        # xxx0 = xxx0 - 0.22
        # x = x + xxx0 * 0.15

        y = x
        return r, y
