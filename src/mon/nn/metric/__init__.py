#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends ``torchmetrics`` for evaluation metrics."""

from __future__ import annotations

import mon.nn.metric.base
import mon.nn.metric.custom_ssim
import mon.nn.metric.efficiency
import mon.nn.metric.iqa
import mon.nn.metric.pytorch_msssim
import mon.nn.metric.torchmetric
from mon.nn.metric.base import *
from mon.nn.metric.efficiency import *
from mon.nn.metric.torchmetric import *
