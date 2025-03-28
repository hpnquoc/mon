# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements image visualization functions.

References:
    - https://www.geeksforgeeks.org/concatenate-images-using-opencv-in-python/
"""

from __future__ import annotations

__all__ = [
    "get_grid_size",
    "move_figure",
    "plt",
]

import math

import matplotlib
from matplotlib import pyplot as plt

from mon import core

console = core.console


# mpl.use("wxAgg")

plt.ion()
plt.show()
# plt.switch_backend("qt6agg")
plt.rcParams["savefig.bbox"] = "tight"


# region Window Positioning

def get_grid_size(n: int, nrow: int = 4) -> list[int]:
    """Calculates grid size for displaying items.

    Args:
        n: Number of items.
        nrow: Items per row (grid size becomes ``(n / nrow, nrow)``).
            Default is ``4``; if ``0`` or negative, uses one row.

    Returns:
        List of [nrows, ncols] representing rows and columns.
    """
    ncols = nrow if nrow > 0 else n
    nrows = math.ceil(n / ncols)
    return [nrows, ncols]


def move_figure(x: int, y: int):
    """Moves the matplotlib figure to the specified window position.

    Args:
        x: X-coordinate for upper-left corner.
        y: Y-coordinate for upper-left corner.
    """
    mngr    = plt.get_current_fig_manager()
    backend = matplotlib.get_backend()
    if backend == "TkAgg":
        mngr.window.wm_geometry(f"+{x}+{y}")
    elif backend == "WXAgg":
        mngr.window.SetPosition((x, y))
    else:
        mngr.window.move(x, y)

# endregion
