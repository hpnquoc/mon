#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""JSON File Handler.

This module implements the JSON file handler by extending the ``json`` package.
"""

from __future__ import annotations

from json import *
from typing import Any, TextIO

import numpy as np

from mon.core import pathlib
from mon.core.file import base
from mon.globals import FILE_HANDLERS


# region JSON File Handler

@FILE_HANDLERS.register(name=".json")
class JSONHandler(base.FileHandler):
    """JSON file handler."""
    
    @staticmethod
    def set_default(obj: Any):
        """If an object is a ``set``, ``range``, ``numpy array``, or numpy generic,
        convert it to a ``list``.
        
        Args:
            A serializable object.
        """
        if isinstance(obj, (set, range, np.ndarray, np.generic)):
            return list(obj) if isinstance(obj, (set, range, np.ndarray)) else obj.item()
        raise TypeError(f"{type(obj)} is not supported for json dump.")
    
    # noinspection PyTypeChecker
    def read_from_fileobj(self, path: pathlib.Path | str | TextIO, **kwargs) -> Any:
        """Read data from a file object.

        Args:
            path: The path to the file. It can be a ``pathlib.Path``, ``str``,
                or ``TextIO`` object.
            **kwargs: Additional keyword arguments to pass to the ``json.load`` function.
    
        Returns:
            The data read from the file.
        """
        return load(path)
    
    # noinspection PyTypeChecker
    def write_to_fileobj(self, obj : Any, path: pathlib.Path | str | TextIO, **kwargs):
        """Write data to a file object.
    
        Args:
            obj: The data to write to the file.
            path: The path to the file. It can be a ``pathlib.Path``, ``str``,
                or ``TextIO`` object.
            **kwargs: Additional keyword arguments to pass to the ``json.dump`` function.
        """
        path = pathlib.Path(path)
        kwargs.setdefault("default", self.set_default)
        dump(obj, path, **kwargs)
    
    def write_to_string(self, obj: Any, **kwargs) -> str:
        """Write data to a JSON string.
    
        Args:
            obj: The data to convert to a JSON string.
            **kwargs: Additional keyword arguments to pass to the ``json.dumps`` function.
    
        Returns:
            The JSON string representation of the data.
        """
        kwargs.setdefault("default", self.set_default)
        return dumps(obj=obj, **kwargs)

# endregion
