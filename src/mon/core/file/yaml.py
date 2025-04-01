#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends ``yaml`` module for YAML file handling."""

from __future__ import annotations

from typing import Any, TextIO

from yaml import *

from mon.core.file import base
from mon.globals import FILE_HANDLERS


@FILE_HANDLERS.register(name=".yaml")
@FILE_HANDLERS.register(name=".yml")
class YAMLHandler(base.FileHandler):
    """Handler for YAML file operations."""
    
    def read_from_fileobj(self, path: TextIO, **kwargs) -> Any:
        """Loads data from a file object.

        Args:
            path: File stream as ``TextIO`` (text file-like object).
            **kwargs: Extra args for ``yaml.load()``.

        Returns:
            Parsed YAML data as ``Any``.
        """
        kwargs.setdefault("Loader", FullLoader)
        return load(path, **kwargs)
    
    def write_to_fileobj(self, obj: Any, path: TextIO, **kwargs):
        """Writes data to a file object.

        Args:
            obj: Data to serialize as YAML.
            path: File stream as ``TextIO`` (text file-like object).
            **kwargs: Extra args for ``yaml.dump()``.
        """
        kwargs.setdefault("Dumper", Dumper)
        dump(data=obj, stream=path, **kwargs)
    
    def write_to_string(self, obj: Any, **kwargs) -> str:
        """Converts data to a YAML string.

        Args:
            obj: Data to serialize as YAML.
            **kwargs: Extra args for ``yaml.dump()``.

        Returns:
            YAML string representation of ``obj``.
        """
        kwargs.setdefault("Dumper", Dumper)
        return dump(data=obj, stream=None, **kwargs)
