#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""YAML File Handler.

This module implements the YAML file handler by extending the ``yaml`` module.
"""

from __future__ import annotations

from typing import Any, TextIO

from yaml import *

from mon.core import pathlib
from mon.core.file import base
from mon.globals import FILE_HANDLERS


# region YAML File Handler

@FILE_HANDLERS.register(name=".yaml")
@FILE_HANDLERS.register(name=".yml")
class YAMLHandler(base.FileHandler):
    """YAML file handler."""
    
    def read_from_fileobj(self, path: pathlib.Path | str | TextIO, **kwargs) -> Any:
        """Convert a dictionary to an XML string.
    
        Args:
            obj: The dictionary to convert to an XML string.
            **kwargs: Additional keyword arguments to pass to the ``xmltodict.unparse`` function.
    
        Returns:
            The XML string representation of the dictionary.
        """
        kwargs.setdefault("Loader", FullLoader)
        return load(stream=path, **kwargs)
    
    def write_to_fileobj(self, obj : Any, path: pathlib.Path | str | TextIO, **kwargs):
        """Write data to a YAML file object.
    
        Args:
            obj: The data to write to the file.
            path: The path to the file. It can be a ``pathlib.Path``, ``str``, or `
                `TextIO`` object.
            **kwargs: Additional keyword arguments to pass to the ``yaml.dump`` function.
        """
        kwargs.setdefault("Dumper", Dumper)
        dump(data=obj, stream=pathlib.Path(path), **kwargs)
    
    def write_to_string(self, obj: Any, **kwargs) -> str:
        """Convert a dictionary to a YAML string.
    
        Args:
            obj: The dictionary to convert to a YAML string.
            **kwargs: Additional keyword arguments to pass to the ``yaml.dump`` function.
    
        Returns:
            The YAML string representation of the dictionary.
        """
        kwargs.setdefault("Dumper", Dumper)
        return dump(data=obj, **kwargs)

# endregion
