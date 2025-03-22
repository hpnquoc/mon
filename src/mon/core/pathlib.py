#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pathlib Module.

This module extends Python `pathlib` module.
"""

from __future__ import annotations

__all__ = [
    "Path",
    "PosixPath",
    "PurePath",
    "PurePosixPath",
    "PureWindowsPath",
    "WindowsPath",
    "copy_file",
    "delete_cache",
    "delete_dir",
    "delete_files",
    "get_ckpt_file",
    "get_config_file",
    "get_files",
    "get_image_file",
    "get_next_version",
    "get_relative_path",
    "get_yaml_file",
    "hash_files",
    "is_url",
    "mkdirs",
    "rmdirs",
]

import glob
import os
import pathlib
import shutil
from pathlib import *

import validators

from mon.core import dtype, humps


# region Path

class Path(type(pathlib.Path())):
    """An extension of ``pathlib.Path`` with more functionalities.
    
    Notes:
        Most of the functions here should be properties, but we keep them as
        methods to be consistent with ``pathlib.Path``.
    """
    
    def is_basename(self) -> bool:
        """Return True if the current path is a basename of a file, otherwise False."""
        return str(self) == self.name
    
    def is_bmp_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a .bmp file, otherwise False."""
        return (not exist or self.is_file()) and self.suffix.lower() == ".bmp"
    
    def is_cache_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a cache file, otherwise False."""
        return (not exist or self.is_file()) and self.suffix.lower() == ".cache"
    
    def is_ckpt_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a checkpoint file, otherwise False."""
        return (not exist or self.is_file()) and self.suffix.lower() == ".ckpt"
    
    def is_config_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a .config or .cfg file, otherwise False."""
        from mon.globals import CONFIG_FILE_FORMATS
        return (not exist or self.is_file()) and self.suffix.lower() in CONFIG_FILE_FORMATS

    def is_dir_like(self) -> bool:
        """Return True if the path is a correct directory format."""
        return "" in self.suffix
    
    def is_file_like(self) -> bool:
        """Return True if the path is a correct file format."""
        return "." in self.suffix
    
    def is_image_file(self, exist: bool = True) -> bool:
        """Return True if the current path is an image file, otherwise False."""
        from mon.globals import IMAGE_FILE_FORMATS
        return (not exist or self.is_file()) and self.suffix.lower() in IMAGE_FILE_FORMATS
    
    def is_json_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a .json file, otherwise False."""
        return (not exist or self.is_file()) and self.suffix.lower() == ".json"
    
    def is_name(self) -> bool:
        """Return True if the current path is the same as the stem, otherwise False."""
        return self == self.stem
    
    def is_py_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a .py file, otherwise False."""
        return (not exist or self.is_file()) and self.suffix.lower() == ".py"
    
    def is_raw_image_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a raw image file, otherwise False."""
        return (not exist or self.is_file()) and self.suffix.lower() in [".dng", ".arw"]
    
    def is_stem(self) -> bool:
        """Return True if the current path is the same as the stem, otherwise False."""
        return str(self) == self.stem
    
    def is_torch_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a file, and the file extension is
        one of the following:
            - ``.pt``
            - ``.pt.tar``
            - ``.pth``
            - ``.pth.tar``
            - ``.weights``
            - ``.ckpt``
            - ``.onnx``
            - Otherwise, return False.
        """
        from mon.globals import TORCH_FILE_FORMATS
        return (not exist or self.is_file()) and self.suffix.lower() in TORCH_FILE_FORMATS
    
    def is_txt_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a text file, otherwise False."""
        return (not exist or self.is_file()) and self.suffix.lower() == ".txt"
    
    def is_url(self) -> bool:
        """Return True if the current path is a valid URL, otherwise False."""
        return not isinstance(validators.url(str(self)), validators.ValidationError)
    
    def is_url_or_file(self, exist: bool = True) -> bool:
        """Return True if the path is a file or a valid URL, otherwise False."""
        return (not exist or self.is_file()) or not isinstance(validators.url(self), validators.ValidationError)
    
    def is_video_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a video file, otherwise False."""
        from mon.globals import VIDEO_FILE_FORMATS
        return (not exist or self.is_file()) and self.suffix.lower() in VIDEO_FILE_FORMATS.values()
    
    def is_video_stream(self) -> bool:
        """Return True if the current path is a video stream, otherwise False."""
        return "rtsp" in str(self).lower()
    
    def is_weights_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a .pt or .pth file, otherwise False."""
        from mon.globals import WEIGHTS_FILE_FORMATS
        return (not exist or self.is_file()) and self.suffix.lower() in WEIGHTS_FILE_FORMATS
    
    def is_xml_file(self, exist: bool = True) -> bool:
        """Return True if the current path is an .xml file, otherwise False."""
        return (not exist or self.is_file()) and self.suffix.lower() == ".xml"
    
    def is_yaml_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a .yaml or .yml file, otherwise False."""
        return (not exist or self.is_file()) and self.suffix.lower() in [".yaml", ".yml"]
    
    def has_subdir(self, name: str) -> bool:
        """Return True if a directory has a subdirectory with the given name."""
        return name in [d.name for d in self.subdirs()]
    
    def subdirs(self, recursive: bool = False) -> list[Path]:
        """Return a list of subdirectories' paths inside the current directory."""
        path  = self.parent if self.is_file_like() else self
        paths = list(path.rglob("*")) if recursive else list(path.iterdir())
        return [p for p in paths if p.is_dir()]
    
    def files(self, recursive: bool = False) -> list[Path]:
        """Return a list of file paths inside the current directory."""
        path  = self.parent if self.is_file_like() else self
        paths = list(path.rglob("*")) if recursive else list(path.iterdir())
        return [p for p in paths if p.is_file()]
    
    def ckpt_file(self) -> Path | None:
        """Return the YAML file with the given path."""
        return get_ckpt_file(self)
    
    def config_file(self) -> Path:
        """Return the configuration file with the given path."""
        return get_config_file(self)
    
    def latest_file(self) -> Path | None:
        """Return the latest file based on creation time."""
        files = self.files()
        return max(files, key=os.path.getctime) if files else None
    
    def image_file(self) -> Path:
        """Return the image file with the given path."""
        return get_image_file(self)
    
    def relative_path(self, start_part: Path | str) -> Path:
        """Get the relative path from the given part."""
        return get_relative_path(self, start_part)
    
    def yaml_file(self) -> Path:
        """Return the YAML file with the given path."""
        return get_yaml_file(self)
    
    def copy_to(self, dst: Path | str, replace: bool = True):
        """Copy a file to a new location.
        
        Args:
            dst: The destination path.
            replace: If ``True`` replace the existing file at the destination
                location. Default: ``True``.
        """
        dst = Path(dst)
        if dst.is_url():
            raise NotImplementedError()
        mkdirs(dst, parents=True, exist_ok=True)
        dst = dst / self.name if dst.is_dir_like() else dst
        if replace:
            dst.unlink(missing_ok=True)
        shutil.copyfile(src=str(self), dst=str(dst))
    
    def replace(self, old: str, new: str, count: int = 1) -> Path:
        """Replace old string with new string in the path."""
        return Path(str(self).replace(old, new, count))
    
# endregion


# region Check

def is_url(url: str) -> bool:
    """Return True if the URL is valid, otherwise False."""
    return not isinstance(validators.url(url), validators.ValidationError)

# endregion


# region Obtainment

def get_ckpt_file(path: Path) -> Path:
    """Get the ckpt file from the given path."""
    temp = path.parent / f"{path.stem}.ckpt"
    return temp if temp.is_yaml_file() else path


def get_config_file(path: Path) -> Path:
    """Get the configuration file from the given path."""
    from mon.globals import CONFIG_FILE_FORMATS
    for ext in CONFIG_FILE_FORMATS:
        temp           = path.parent / f"{path.stem}{ext}"
        temp_snakecase = path.parent / f"{humps.snakecase(path.stem)}{ext}"
        if temp.is_config_file():
            return temp
        if temp_snakecase.is_config_file():
            return temp_snakecase
    return path


def get_files(regex: str, recursive: bool = False) -> list[Path]:
    """Get all files matching the given regular expression.
    
    Args:
        regex: A file path patterns.
        recursive: If ``True``, look for file in subdirectories. Default: ``False``.
        
    Returns:
        A ``list`` of unique file paths.
    """
    paths = [Path(p) for p in glob.glob(regex, recursive=recursive) if Path(p).is_file()]
    return dtype.unique(paths)


def get_image_file(path: Path) -> Path:
    """Get the image file from the given path."""
    from mon.globals import IMAGE_FILE_FORMATS
    for ext in IMAGE_FILE_FORMATS:
        temp = path.parent / f"{path.stem}{ext}"
        if temp.is_image_file():
            return temp
    return path


def get_next_version(path: Path | str, prefix: str = None) -> int:
    """Get the next version number of items in a directory.
    
    Args:
        path: The directory path containing files.
        prefix: The file prefix. Default: ``None``.
    """
    path = Path(path)
    existing_versions = [
        int(f.stem.split("-")[-1].replace("/", ""))
        for f in path.iterdir()
        if f.stem.startswith(("version-", "exp-", f"{prefix}-" if prefix else ""))
    ]
    return max(existing_versions, default=0) + 1


def get_relative_path(path: Path | str, start_part: Path | str) -> Path:
    """Get the relative path starting from the given `part`.
    
    Args:
        path: The path to the file.
        start_part: The starting part of the relative path.
    """
    path       = Path(path)
    start_part = str(start_part)
    if not start_part or start_part not in str(path):
        return path
    return Path(*path.parts[path.parts.index(start_part):])
    

def get_yaml_file(path: Path) -> Path:
    """Get the YAML file from the given path."""
    for ext in [".yaml", ".yml"]:
        temp = path.parent / f"{path.stem}{ext}"
        if temp.is_yaml_file():
            return temp
    return path


def hash_files(paths: list[Path | str]) -> int:
    """Return the total hash value of all the files (if it has one). Hash
    values are integers (in bytes) of all files.
    """
    paths = [Path(f) for f in dtype.to_list(paths) if f]
    return sum(f.stat().st_size for f in paths if f.is_file())

# endregion


# region Creation

def copy_file(src: Path | str, dst: Path | str):
    """Copy a file to a new location.
    
    Args:
        src: The path to the original file.
        dst: The destination path.
    """
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src=str(src), dst=str(dst))

# endregion


# region Alternation

def delete_cache(path: Path | str, recursive: bool = True):
    """Clears cache files in a directory and subdirectories.

    Args:
        path: The directory path containing the cache files.
        recursive: If ``True``, recursively look for cache files in subdirectories.
            Default: ``True``.
    """
    delete_files(path=path, regex=".cache", recursive=recursive)


def delete_dir(paths: Path | str | list[Path | str]):
    paths = dtype.unique(dtype.to_list(paths))
    for p in paths:
        p = Path(p)
        if p.exists():
            delete_files(path=p, regex="*", recursive=True)
            shutil.rmtree(p)


def delete_files(
    path     : Path | str,
    regex    : str  = None,
    recursive: bool = False
):
    """Delete all files matching the given regular expression.
    
    Args:
        path: A path to a directory to search for the files to delete.
        regex: A file path patterns. Default: ``None``.
        recursive: If ``True``, look for file in subdirectories. Default: ``False``.
    """
    path = Path(path)
    if regex:
        path  = path.parent if not path.is_dir() else path
        files = list(path.rglob(regex)) if recursive else list(path.glob(regex))
    else:
        files = [path]
    for f in files:
        try:
            f.unlink()
        except Exception as err:
            print(f"Cannot delete files: {err}.")


def mkdirs(
    paths   : Path | str | list[Path | str],
    mode    : int  = 0o777,
    parents : bool = True,
    exist_ok: bool = True,
    replace : bool = False,
):
    """Create a new directory at this given path. If mode is given, it is
    combined with the process' umask value to determine the file mode and access
    flags. If the path already exists, ``FileExistsError`` is raised.
    
    Args:
        paths: A `list` of directories' absolute paths.
        mode: If given, it is combined with the process' umask value to
            determine the file mode and access flags.
        parents:
            - If ``True`` (the default), any missing parents of this path are
              created as needed; they're created with the default permissions
              without taking mode into account (mimicking the POSIX mkdir -p
              command).
            - If ``False``, a missing parent raises ``FileNotFoundError``.
        exist_ok:
            - If ``True`` (the default), ``FileExistsError`` exceptions will be
              ignored (same behavior as the POSIX mkdir -p command), but only
            if the last path component isn't an existing non-directory file.
            - If ``False``, ``FileExistsError`` is raised if the target
              directory already exists.
        replace: If ``True``, delete existing directories and recreate.
            Default: ``False``.
    """
    paths = dtype.unique(dtype.to_list(paths))
    for p in paths:
        p = Path(p)
        if p.is_url():
            continue
        p = p.parent if p.is_file_like() else p
        if replace:
            delete_files(path=p, regex="*")
            p.rmdir()
        p.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)


def rmdirs(paths: Path | str | list[pathlib.Path | str]):
    """Delete directories.
    
    Args:
        paths: A ``list`` of directories' absolute paths.
    """
    paths = dtype.unique(dtype.to_list(paths))
    for p in paths:
        p = Path(p)
        if p.is_url():
            continue
        p = p.parent if p.is_file_like() else p
        delete_files(path=p, regex="*")
        try:
            p.rmdir()
        except Exception as err:
            print(f"Cannot delete directory: {err}.")

# endregion
