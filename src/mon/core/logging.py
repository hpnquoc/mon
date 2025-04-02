#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends Python's ``logging`` module."""

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
from typing import Iterator

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
    """Retrieves or creates a global logger with ``rich`` support.

    Args:
        path: Path for log file, adds file handler if given. Default is ``None``.

    Returns:
        Global logger instance.
    """
    logger = logging.getLogger("global_logger")
    if path:
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)
    return logger

# endregion


# region Print

@contextlib.contextmanager
def disable_print() -> Iterator[None]:
    """Temporarily disables printing to stdout by redirecting it to ``os.devnull``.

    Yields:
        None, allowing use in a ``with`` statement to suppress output.

    Example:
        >>> with disable_print():
        >>>     print("This won't appear")
        >>> print("This will appear")
    """
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def enable_print():
    """Restores printing to stdout by resetting it to the original stream.

    Notes:
        Use this to undo manual redirection of ``sys.stdout`` (e.g., to ``os.devnull``).
    """
    sys.stdout = sys.__stdout__

# endregion
