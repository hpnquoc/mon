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
from mon.globals import *


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

# endregion


# region Constants

# endregion
