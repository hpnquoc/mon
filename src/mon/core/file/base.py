#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class and functions for file handlers with helper utilities."""

from __future__ import annotations

__all__ = [
    "FileHandler",
    "write_to_file",
    "read_from_file",
    "merge_files",
]

from abc import ABC, abstractmethod
from typing import Any, TextIO

from mon.core import dtype, pathlib
from mon.globals import FILE_HANDLERS


# region File Handler

class FileHandler(ABC):
    """Base class for reading and writing data in various file formats."""
    
    @abstractmethod
    def read_from_fileobj(self, path: pathlib.Path | str | TextIO, **kwargs) -> Any:
        """Loads content from a ``file`` object.
    
        Args:
            path: ``pathlib.Path``, ``str``, or ``TextIO`` stream.
            kwargs: Additional keyword arguments.
        
        Returns:
            Content from the ``file``.
        """
        pass
    
    @abstractmethod
    def write_to_fileobj(self, obj: Any, path: pathlib.Path | str | TextIO, **kwargs):
        """Writes a serializable object to a ``file`` object.

        Args:
            obj: Serializable object to write.
            path: ``pathlib.Path``, ``str``, or ``TextIO`` stream.
            kwargs: Additional keyword arguments.
        """
        pass
    
    @abstractmethod
    def write_to_string(self, obj: Any, **kwargs) -> str:
        """Converts a serializable object to a ``str``.

        Args:
            obj: Serializable object to convert.
            kwargs: Additional keyword arguments.

        Returns:
            String representation of the object.
        """
        pass
    
    def read_from_file(self, path: pathlib.Path | str, mode: str = "r", **kwargs) -> Any:
        """Loads content from a ``file``.

        Args:
            path: ``pathlib.Path`` or ``str`` file path.
            mode: File open ``mode``. Default is ``"r"`` for read-only.
            kwargs: Additional keyword arguments.
    
        Returns:
            Content from the ``file``.
        """
        with open(path, mode) as f:
            return self.read_from_fileobj(path=f, **kwargs)
    
    def write_to_file(self, obj: Any, path: pathlib.Path | str, mode: str = "w", **kwargs):
        """Writes a serializable object to a ``file``.
    
        Args:
            obj: Serializable object to write.
            path: ``pathlib.Path`` or ``str`` file path.
            mode: File open ``mode``. Default is ``"w"`` for write-only.
            kwargs: Additional keyword arguments.
        """
        with open(path, mode) as f:
            self.write_to_fileobj(obj=obj, path=f, **kwargs)


def write_to_file(
    obj        : Any,
    path       : pathlib.Path | str | TextIO,
    file_format: str = None,
    **kwargs
):
    """Writes a serializable object to a ``file``.

    Args:
        obj: Object to serialize.
        path: ``pathlib.Path``, ``str`` path, or ``TextIO`` stream.
        file_format: File format, inferred from ``path`` if ``None``.
            Default is ``None``.
        kwargs: Additional keyword arguments.

    Raises:
        ValueError: If ``file_format`` is not supported.
    """
    path_obj    = pathlib.Path(path) if isinstance(path, (pathlib.Path, str)) else path
    file_format = file_format or (path_obj.suffix if isinstance(path_obj, pathlib.Path) else "")
    if file_format not in FILE_HANDLERS:
        raise ValueError(f"[file_format] must be one of {list(FILE_HANDLERS.keys())}, "
                         f"got {file_format}")
    
    handler: FileHandler = FILE_HANDLERS.build(name=file_format)
    if hasattr(path, "write"):
        handler.write_to_fileobj(obj=obj, path=path, **kwargs)
    else:
        handler.write_to_file(obj=obj, path=path_obj, **kwargs)


def read_from_file(
    path       : pathlib.Path | str | TextIO,
    file_format: str = None,
    **kwargs
) -> Any:
    """Loads content from a ``file``.

    Args:
        path: ``pathlib.Path``, ``str`` path, or ``TextIO`` stream.
        file_format: File format, inferred from ``path`` if ``None``.
            Default is ``None``.
        kwargs: Additional keyword arguments.

    Returns:
        ``File`` content.

    Raises:
        TypeError: If ``path`` is not a valid type.
    """
    path_obj    = pathlib.Path(path) if isinstance(path, (pathlib.Path, str)) else path
    file_format = file_format or (path_obj.suffix if isinstance(path_obj, pathlib.Path) else "")
    
    handler: FileHandler = FILE_HANDLERS.build(name=file_format)
    if hasattr(path, "read"):
        return handler.read_from_fileobj(path=path, **kwargs)
    if isinstance(path_obj, (pathlib.Path, str)):
        return handler.read_from_file(path=path_obj, **kwargs)
    raise TypeError(f"[path] must be str, pathlib.Path, or file-like, "
                    f"got {type(path).__name__}.")


def merge_files(
    in_paths   : list[pathlib.Path | str | TextIO],
    out_path   : pathlib.Path | str | TextIO,
    file_format: str = None,
):
    """Merges content from multiple ``files`` into a single ``file``.

    Args:
        in_paths: List of input ``pathlib.Path``, ``str``, or ``TextIO`` streams.
        out_path: Output ``pathlib.Path``, ``str`` path, or ``TextIO`` stream.
        file_format: File format, inferred from ``out_path`` if ``None``.
            Default is ``None``.
        kwargs: Additional keyword arguments.

    Raises:
        TypeError: If content from ``in_paths`` is neither ``list`` nor ``dict``.
    """
    in_paths = [pathlib.Path(p) for p in dtype.to_list(in_paths)]
    data = None
    for input_path in in_paths:
        content = read_from_file(path=input_path)
        if isinstance(content, list):
            data = data or []
            data.extend(content)
        elif isinstance(content, dict):
            data = data or {}
            data.update(content)
        else:
            raise TypeError(f"[in_paths] content must be list or dict, "
                            f"got {type(content).__name__}.")
    
    write_to_file(obj=data, path=out_path, file_format=file_format)

# endregion
