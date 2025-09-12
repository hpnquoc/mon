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


# region Constants

CLASSLABELS = ClassLabels([
    {"name": "unidentified",   "id": 0, "category": "void",           "category_id": 0, "train_id": 255, "color": mon.AppleRGB.GRAY},
    {"name": "other",          "id": 1, "category": "void",           "category_id": 0, "train_id": 255, "color": mon.AppleRGB.GRAY2},
    {"name": "pedestrian",     "id": 2, "category": "pedestrian",     "category_id": 1, "train_id": 0,   "color": mon.AppleRGB.PURPLE},
    {"name": "micro-mobility", "id": 3, "category": "micro-mobility", "category_id": 2, "train_id": 1,   "color": mon.AppleRGB.INDIGO},
    {"name": "car",            "id": 4, "category": "vehicle",        "category_id": 3, "train_id": 2,   "color": mon.AppleRGB.RED},
    {"name": "bus",            "id": 5, "category": "vehicle",        "category_id": 3, "train_id": 3,   "color": mon.AppleRGB.ORANGE},
    {"name": "small-truck",    "id": 6, "category": "vehicle",        "category_id": 3, "train_id": 4,   "color": mon.AppleRGB.TEAL},
    {"name": "truck",          "id": 7, "category": "vehicle",        "category_id": 3, "train_id": 5,   "color": mon.AppleRGB.BLUE},
])
ID2CLASS    = CLASSLABELS.id_to_class

# endregion
