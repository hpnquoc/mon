#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles system-level operations."""

__all__ = [
    "check_installed_package",
    "set_random_seed",
]

import importlib
import importlib.util
import os
import random

import numpy as np
import torch


# ----- Package -----
def check_installed_package(package_name: str, verbose: bool = False) -> bool:
    """Checks if a package is installed.

    Args:
        package_name: Name of the package to check.
        verbose: If ``True``, prints install status. Default is ``False``.

    Returns:
        ``True`` if package is installed, ``False`` otherwise.
    """
    try:
        importlib.import_module(package_name)
        if verbose:
            print(f"[{package_name}] is installed")
        return True
    except ImportError:
        if verbose:
            print(f"[{package_name}] is not installed")
        return False


# ----- Seed -----
def set_random_seed(seed: int | list[int] | tuple[int, int]) -> None:
    """Sets random seeds for various libraries.

    Args:
        seed: Int, list of ints, or tuple of two ints for range selection.
    """
    if isinstance(seed, (list, tuple)):
        seed = random.randint(seed[0], seed[1]) if len(seed) == 2 else seed[-1]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
