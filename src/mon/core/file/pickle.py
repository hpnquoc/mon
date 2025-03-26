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
    """Handler for Pickle file operations."""
    
    def read_from_fileobj(self, path: TextIO, **kwargs) -> Any:
        """Loads data from a file object.

        Args:
            path: File stream as ``TextIO`` (binary file-like object).
            **kwargs: Additional arguments for ``pickle.load``.

        Returns:
            Deserialized Pickle data.
        """
        return load(path, **kwargs)
    
    def write_to_fileobj(self, obj: Any, path: TextIO, **kwargs):
        """Writes data to a file object.

        Args:
            obj: Data to serialize.
            path: File stream as ``TextIO`` (binary file-like object).
            **kwargs: Additional arguments for ``pickle.dump``.
        """
        kwargs.setdefault("protocol", 4)
        dump(obj, path, **kwargs)
    
    def write_to_string(self, obj: Any, **kwargs) -> bytes:
        """Converts data to a Pickle byte string.

        Args:
            obj: Data to serialize.
            **kwargs: Additional arguments for ``pickle.dumps``.

        Returns:
            Pickle byte string representation of ``obj``.
        """
        kwargs.setdefault("protocol", 4)
        return dumps(obj, **kwargs)
    
    def read_from_file(self, path: pathlib.Path | str, mode: str = "rb", **kwargs) -> Any:
        """Loads data from a file.

        Args:
            path: File path as ``pathlib.Path`` or ``str``.
            mode: File open mode. Default is ``rb`` for read binary.
            **kwargs: Additional arguments for ``read_from_fileobj``.

        Returns:
            Deserialized Pickle data.
        """
        return super().read_from_file(path=pathlib.Path(path), mode=mode, **kwargs)
    
    def write_to_file(self, obj: Any, path: pathlib.Path | str, mode: str = "wb", **kwargs):
        """Writes data to a file.

        Args:
            obj: Data to serialize.
            path: File path as ``pathlib.Path`` or ``str``.
            mode: File open mode. Default is ``wb`` for write binary.
            **kwargs: Additional arguments for ``write_to_fileobj``.
        """
        super().write_to_file(obj=obj, path=pathlib.Path(path), mode=mode, **kwargs)

# endregion
