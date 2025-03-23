#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Logging Module.

This module extends Python's `logging` module.
"""

from __future__ import annotations

__all__ = [
    "disable_print",
    "enable_print",
    "get_logger",
    "logger",
]

import contextlib
import logging
import os
import sys

from rich import logging as r_logging

from mon.core import pathlib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disable TensorFlow logging


# region Logging

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(message)s",
    handlers = [r_logging.RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")
# logger.setLevel(logging.INFO)


def get_logger(path: pathlib.Path = None) -> logging.Logger:
    """Get access to the global ``logging.Logger`` object that uses ``rich``.
    Create a new one if it doesn't exist.

    Args:
        path: The path to store the log info. Default: ``None``.

    Returns:
        The global logger instance.
    """
    if path:
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            " %(asctime)s [%(file_name)s %(lineno)s] %(levelname)s: %(message)s"
        ))
        logger.addHandler(file_handler)
    
    return logger

# endregion


# region Print

def disable_print():
    """Temporarily disable printing to stdout by redirecting it to os.devnull."""
    # sys.stdout = open(os.devnull, "w")
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        yield


# Restore
def enable_print():
    """Restore printing to stdout."""
    sys.stdout = sys.__stdout__

# endregion
