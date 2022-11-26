#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code playground.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from one.core import assert_tensor_of_ndim
from one.vision.transformation import get_atmosphere_channel


y = torch.randn([3, 3, 5, 5])
q = get_atmosphere_channel(y, p=0.5)
print(q, q.shape)
