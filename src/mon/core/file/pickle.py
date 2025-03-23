#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pickle File Handler.

This module implements a Pickle file handler by extending the ``pickle`` module.
"""

from __future__ import annotations

from pickle import *
from typing import Any, TextIO

from mon.core import pathlib
from mon.core.file import base
from mon.globals import FILE_HANDLERS


# region Pickle File Handler

@FILE_HANDLERS.register(name=".pickle")
@FILE_HANDLERS.register(name=".pkl")
class PickleHandler(base.FileHandler):
    """Pickle file handler."""
    
    def read_from_fileobj(self, path: pathlib.Path | str | TextIO, **kwargs) -> Any:
        """Read data from a file object.
    
        Args:
            path: The path to the file. It can be a ``pathlib.Path``, ``str``,
                or ``TextIO`` object.
            **kwargs: Additional keyword arguments to pass to the ``pickle.load`` function.
    
        Returns:
            The data read from the file.
        """
        return load(pathlib.Path(path), **kwargs)
    
    def write_to_fileobj(self, obj: Any, path: pathlib.Path | str | TextIO, **kwargs):
        """Write data to a file object.
    
        Args:
            obj: The data to write to the file.
            path: The path to the file. It can be a ``pathlib.Path``, ``str``,
                or ``TextIO`` object.
            **kwargs: Additional keyword arguments to pass to the ``pickle.dump`` function.
        """
        kwargs.setdefault("protocol", 4)
        dump(obj, pathlib.Path(path), **kwargs)
    
    def write_to_string(self, obj: Any, **kwargs) -> str:
        """Write data to a JSON string.
    
        Args:
            obj: The data to convert to a JSON string.
            **kwargs: Additional keyword arguments to pass to the ``pickle.dumps`` function.
    
        Returns:
            The JSON string representation of the data.
        """
        kwargs.setdefault("protocol", 2)
        return dumps(obj, **kwargs)
    
    def read_from_file(self, path: pathlib.Path | str, mode: str = "rb", **kwargs) -> Any:
        """Read data from a file.
    
        Args:
            path: The path to the file. It can be a ``pathlib.Path`` or ``str``.
            mode: The mode in which to open the file. Default: ``rb`` (read binary).
            **kwargs: Additional keyword arguments to pass to the parent class's
                ``read_from_file`` method.
    
        Returns:
            The data read from the file.
        """
        path = pathlib.Path(path)
        return super().read_from_file(path=path, mode=mode, **kwargs)
    
    def write_to_file(self, obj : Any, path: pathlib.Path | str, mode: str = "wb", **kwargs):
        """Write data to a file.
    
        Args:
            obj: The data to write to the file.
            path: The path to the file. It can be a ``pathlib.Path`` or ``str``.
            mode: The mode in which to open the file. Default: ``wb`` (write binary).
            **kwargs: Additional keyword arguments to pass to the parent class's
                ``write_to_file`` method.
        """
        path = pathlib.Path(path)
        super().write_to_file(obj=obj, path=path, mode=mode, **kwargs)

# endregion
