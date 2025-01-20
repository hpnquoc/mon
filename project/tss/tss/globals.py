#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Globals.

This module defines all global constants used across :obj:`tss` framework.

Notes:
    * To avoid circular dependency, only define constants of basic/atomic types.
    * The same goes for type aliases.
    * The only exception is the enum and factory constants.
"""

from __future__ import annotations

__all__ = [
    "CONFIG_DIR",
    "DATA_DIR",
    "ROOT_DIR",
    "SRC_DIR",
    "ZOO_DIR",
]

import mon
from mon.core import dtype as DT


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
    """Apple's 12 RGB colors."""
    
    GRAY   = (128, 128, 128)
    RED    = (255, 59 , 48)
    GREEN  = (52 , 199, 89)
    BLUE   = (0  , 122, 255)
    ORANGE = (255, 149, 5)
    YELLOW = (255, 204, 0)
    BROWN  = (162, 132, 94)
    PINK   = (255, 45 , 85)
    PURPLE = (88 , 86 , 214)
    TEAL   = (90 , 200, 250)
    INDIGO = (85 , 190, 240)
    BLACK  = (0  , 0  , 0)
    WHITE  = (255, 255, 255)

# endregion


# region Constants

# endregion
