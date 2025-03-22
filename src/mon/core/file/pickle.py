#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pickle File Handler.

This module implements a Pickle file handler by extending the `pickle`
module.
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
        return load(pathlib.Path(path), **kwargs)
    
    def write_to_fileobj(self, obj: Any, path: pathlib.Path | str | TextIO, **kwargs):
        kwargs.setdefault("protocol", 4)
        dump(obj, pathlib.Path(path), **kwargs)
    
    def write_to_string(self, obj: Any, **kwargs) -> str:
        kwargs.setdefault("protocol", 2)
        return dumps(obj, **kwargs)
    
    def read_from_file(self, path: pathlib.Path | str, mode: str = "rb", **kwargs) -> Any:
        path = pathlib.Path(path)
        return super().read_from_file(path=path, mode=mode, **kwargs)
    
    def write_to_file(self, obj : Any, path: pathlib.Path | str, mode: str = "wb", **kwargs):
        path = pathlib.Path(path)
        super().write_to_file(obj=obj, path=path, mode=mode, **kwargs)

# endregion
