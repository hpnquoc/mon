#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Supports ML/DL research with vision, NLP, speech, built on ``PyTorch`` and
``Lightning``.
"""

# Interface to `torch.nn`. We import commonly used components so that everything can be
# accessed under one single import of ``from mon import nn``.
# noinspection PyUnresolvedReferences
from torch.nn import (
    common_types, Container, functional, init, Module, ModuleDict, ModuleList,
    ParameterDict, ParameterList, Sequential,
)
# noinspection PyUnresolvedReferences
from torch.nn.common_types import (
    _maybe_indices_t, _ratio_2_t, _ratio_3_t, _ratio_any_t, _size_1_t, _size_2_opt_t,
    _size_2_t, _size_3_opt_t, _size_3_t, _size_4_t, _size_5_t, _size_6_t,
    _size_any_opt_t, _size_any_t, _tensor_list_t,
)
# noinspection PyUnresolvedReferences
from torch.nn.parallel import DataParallel as DataParallel
# noinspection PyUnresolvedReferences
from torch.nn.parameter import (
    Parameter as Parameter,
    UninitializedBuffer as UninitializedBuffer,
    UninitializedParameter as UninitializedParameter,
)

# Import
from .callback import *
from .factory import *
from .logger import *
from .loss import *
from .lr_scheduler import *
from .metric import *
from .model import *
from .modules import *
from .modules import snn
from .optimizer import *
from .runner import *
from .strategy import *
from .utils import *
