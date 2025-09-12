#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Globals.

This module defines all global constants used across :obj:`tss` framework.

Notes:
    * To avoid circular dependency, only define constants of basic/atomic types.
    * The same goes for type aliases.
    * The only exception is the enum and factory constants.
"""

__all__ = [
    "AppleRGB",
    "CLASSLABELS",
    "CONFIG_DIR",
    "DATA_DIR",
    "ID2CLASS",
    "ROOT_DIR",
    "SRC_DIR",
    "ZOO_DIR",
]

import mon
from mon import ClassLabels
from mon.core import types as DT


# region Directory

current_file = mon.Path(__file__).absolute()
ROOT_DIR     = current_file.parents[1]  # tss
SRC_DIR      = current_file.parents[0]  # tss/tss
CONFIG_DIR   = ROOT_DIR / "config"      # tss/config
DATA_DIR     = ROOT_DIR / "data"        # tss/data
ZOO_DIR      = ROOT_DIR / "zoo"         # tss/zoo

if not CONFIG_DIR.is_dir():
    raise Warning(f"Cannot locate the ``config`` directory.")
if not DATA_DIR.is_dir():
    raise Warning(f"Cannot locate the ``data`` directory.")
if not ZOO_DIR.is_dir():
    raise Warning(f"Cannot locate the ``zoo`` directory.")

# endregion


# region Enum

class AppleRGB(DT.Enum):
    """Apple's RGB colors."""
    
    BLACK       = (  0,   0,   0)
    BLUE        = (  0, 122, 255)
    BROWN       = (162, 132,  94)
    CYAN        = ( 50, 173, 230)
    GRAY        = (128, 128, 128)
    GRAY2       = (174, 174, 178)
    GRAY3       = (199, 199, 204)
    GRAY4       = (209, 209, 214)
    GRAY5       = (229, 229, 234)
    GRAY6       = (242, 242, 247)
    GREEN       = ( 52, 199,  89)
    INDIGO      = ( 85, 190, 240)
    MINT        = (  0, 199,  89)
    ORANGE      = (255, 149,   5)
    PINK        = (255,  45,  85)
    PURPLE      = ( 88,  86, 214)
    RED         = (255,  59,  48)
    TEAL        = ( 90, 200, 250)
    WHITE       = (255, 255, 255)
    YELLOW      = (255, 204,   0)
    DARK_BLUE   = (  0,  64, 221)
    DARK_BROWN  = (127, 101,  69)
    DARK_CYAN   = (  0, 113, 164)
    DARK_GRAY2  = ( 99,  99, 102)
    DARK_GRAY3  = ( 72,  72,  74)
    DARK_GRAY4  = ( 58,  58,  60)
    DARK_GRAY5  = ( 44,  44,  46)
    DARK_GRAY6  = ( 28,  28,  30)
    DARK_GREEN  = ( 36, 138,  61)
    DARK_INDIGO = ( 54,  52, 163)
    DARK_MINT   = ( 12, 129, 123)
    DARK_ORANGE = (201,  52,   0)
    DARK_PINK   = (211,  15,  69)
    DARK_PURPLE = (137,  68, 171)
    DARK_RED    = (255,  69,  58)
    DARK_TEAL   = (  0, 130, 153)
    DARK_YELLOW = (178,  80,   0)
    
# endregion


# region Constants

CLASSLABELS = ClassLabels([
    {"name": "unidentified",   "id": 0, "category": "void",           "category_id": 0, "train_id": 255, "color": AppleRGB.GRAY},
    {"name": "other",          "id": 1, "category": "void",           "category_id": 0, "train_id": 255, "color": AppleRGB.GRAY2},
    {"name": "pedestrian",     "id": 2, "category": "pedestrian",     "category_id": 1, "train_id": 0,   "color": AppleRGB.PURPLE},
    {"name": "micro-mobility", "id": 3, "category": "micro-mobility", "category_id": 2, "train_id": 1,   "color": AppleRGB.INDIGO},
    {"name": "car",            "id": 4, "category": "vehicle",        "category_id": 3, "train_id": 2,   "color": AppleRGB.RED},
    {"name": "bus",            "id": 5, "category": "vehicle",        "category_id": 3, "train_id": 3,   "color": AppleRGB.ORANGE},
    {"name": "small-truck",    "id": 6, "category": "vehicle",        "category_id": 3, "train_id": 4,   "color": AppleRGB.TEAL},
    {"name": "truck",          "id": 7, "category": "vehicle",        "category_id": 3, "train_id": 5,   "color": AppleRGB.BLUE},
])
ID2CLASS    = CLASSLABELS.id_to_class

# endregion
