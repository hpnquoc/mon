#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles system-wise utilities."""

__all__ = [
    "clear_terminal",
    "get_terminal_size",
    "set_random_seed",
    "set_terminal_size",
]

import fcntl
import os
import platform
import random
import shutil
import struct
import subprocess
import sys
import termios
from typing import Sequence

import numpy as np
import torch


# ----- Seed -----
def set_random_seed(seed: int | tuple[int, int]) -> None:
    """Sets random seeds for various libraries.

    Args:
        seed: An ``int``, or a ``tuple`` of :math:`(min, max)` for random selection.
    """
    if isinstance(seed, Sequence):
        seed = random.randint(seed[0], seed[1]) if len(seed) == 2 else seed[-1]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ----- Terminal -----
def clear_terminal():
    """Clears the terminal screen."""
    if platform.system() == "Windows":
        os.system("cls")
    elif platform.system() in ["Darwin", "Linux"]:
        os.system("clear")


def get_terminal_size() -> tuple[int, int]:
    """Gets the size of the terminal window in columns and rows.

    Returns:
        A ``tuple`` of :math:`(columns, rows)`.
    """
    size = shutil.get_terminal_size(fallback=(100, 40))
    return size.columns, size.lines


def set_terminal_size(rows: int = 40, cols: int = 100):
    """Sets the terminal window size to specified rows and columns.

    Args:
        rows: Number of rows for terminal. Default: ``40``.
        cols: Number of columns for terminal. Default: ``100``.
    """
    fd   = sys.stdout.fileno()
    size = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, size)
    subprocess.run(["stty", "rows", str(rows), "cols", str(cols)])
