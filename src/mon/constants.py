#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Defines global constants used across the package."""

__all__ = [
    "ALBUMENTATIONS",
    "DATASETS",
    "DATA_DIR",
    "DEPTH_SOURCE",
    "INFRARED_SOURCE",
    "MODELS",
    "MON_DIR",
    "MON_EXTRA_DIR",
    "ROOT_DIR",
    "SAVE_CKPT_EXT",
    "SAVE_DEBUG_DIR",
    "SAVE_IMAGE_DIR",
    "SAVE_IMAGE_EXT",
    "SAVE_LABEL_DIR",
    "SAVE_VISUALIZE_DIR",
    "SAVE_WEIGHTS_EXT",
    "ZOO_DIR",
]

from mon.core.enum import (
    DepthSource,
    ImageExtension,
    InfraredSource,
    WeightExtension,
)
from mon.core.factory import (
    ALBUMENTATIONS,
    DATASETS,
    MODELS,
)
from mon.core.pathlib import Path


# ----- Directory -----
current_file  = Path(__file__).absolute()
ROOT_DIR      = current_file.parents[2]     # ./mon
DATA_DIR      = ROOT_DIR / "data"           # ./mon/data
SRC_DIR       = ROOT_DIR / "src"            # ./mon/src
MON_DIR       = ROOT_DIR / "src/mon"        # ./mon/src/mon
MON_EXTRA_DIR = ROOT_DIR / "src/mon/extra"  # ./mon/src/mon/extra
ZOO_DIR       = ROOT_DIR / "zoo"            # ./mon/zoo


# ----- Constants -----
DEPTH_SOURCE       = DepthSource.DAv2_ViTB
INFRARED_SOURCE    = InfraredSource.INFRARED
SAVE_DEBUG_DIR     = "debug"
SAVE_IMAGE_DIR     = "pred"
SAVE_LABEL_DIR     = "label"
SAVE_VISUALIZE_DIR = "visualize"
SAVE_CKPT_EXT      = WeightExtension.CKPT.value
SAVE_IMAGE_EXT     = ImageExtension.JPG.value
SAVE_WEIGHTS_EXT   = WeightExtension.PT.value
