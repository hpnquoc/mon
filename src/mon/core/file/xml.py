#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""XML File Handler.

This module implements the XML file handler by extending the `xmltodict`
module.
"""

from __future__ import annotations

from typing import Any, TextIO

from xmltodict import *

from mon.core import pathlib
from mon.core.file import base
from mon.globals import FILE_HANDLERS


# region XML File Handler

@FILE_HANDLERS.register(name=".xml")
class XMLHandler(base.FileHandler):
    """XML file handler."""
    
    def read_from_fileobj(self, path: pathlib.Path | str | TextIO, **kwargs) -> Any:
        """Read data from a file object.

        Args:
            path: The path to the file. It can be a ``pathlib.Path``, ``str``,
                or ``TextIO`` object.
            **kwargs: Additional keyword arguments to pass to the ``xmltodict.parse`` function.
    
        Returns:
            The data read from the file.
        """
        return parse(path.read())
    
    def write_to_fileobj(self, obj : Any, path: pathlib.Path | str | TextIO, **kwargs):
        """Write data to a file object.
    
        Args:
            obj: The data to write to the file. Must be a dictionary.
            path: The path to the file. It can be a ``pathlib.Path``, ``str``,
                or ``TextIO`` object.
            **kwargs: Additional keyword arguments to pass to the ``xmltodict.unparse`` function.
    
        Raises:
            TypeError: If ``obj`` is not a dictionary.
        """
        if not isinstance(obj, dict):
            raise TypeError(f"`obj` must be a `dict`, but got {type(obj)}.")
        with open(path, "w") as f:
            f.write(unparse(input_dict=obj, pretty=True))
    
    def write_to_string(self, obj: Any, **kwargs) -> str:
        """Convert a dictionary to an XML string.
    
        Args:
            obj: The dictionary to convert to an XML string.
            **kwargs: Additional keyword arguments to pass to the ``xmltodict.unparse`` function.
    
        Returns:
            The XML string representation of the dictionary.
        """
        assert isinstance(obj, dict)
        return unparse(input_dict=obj, pretty=True)

# endregion
