#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code playground.
"""

from __future__ import annotations

import torch

a      = torch.ones(10)
a[0:5] = 0
c      = torch.ones([4, 10, 5, 5])
b      = a.reshape(-1, 10, 1, 1)
print(b)
print(b * c)
