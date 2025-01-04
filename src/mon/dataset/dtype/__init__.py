#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset Package.

This package implements the data structures for annotations, datasets, and
datamodules. The base classes are designed to be implemented by the user to
create their own custom labels, datasets, datamodules, and result writers.

We try to support all possible data types: :obj:`torch.Tensor`,
:obj:`numpy.ndarray`, or :obj:`Sequence`, but we prioritize :obj:`torch.Tensor`.
"""

from __future__ import annotations

import mon.dataset.dtype.annotation
import mon.dataset.dtype.datamodule
import mon.dataset.dtype.dataset
from mon.dataset.dtype.annotation import *
from mon.dataset.dtype.datamodule import *
from mon.dataset.dtype.dataset import *
