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
        """Check if the current path is a basename of a file.
    
        Returns:
            ``True`` if the current path is a basename, otherwise ``False``.
        """
        return str(self) == self.name
    
    def is_bmp_file(self, exist: bool = True) -> bool:
        """Check if the current path is a .bmp file.
    
        Args:
            exist: If ``True``, also check if the file exists. Default: ``True``.
    
        Returns:
           ``True`` if the current path is a .bmp file, otherwise ``False``.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".bmp"
    
    def is_cache_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a cache file, otherwise False.
    
        Args:
            exist: If ``True``, also check if the file exists. Default is ``True``.
    
        Returns:
            ``True`` if the current path is a cache file, otherwise ``False``.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".cache"
    
    def is_ckpt_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a checkpoint file, otherwise False.
    
        Args:
            exist: If ``True``, also check if the file exists. Default: ``True``.
    
        Returns:
            ``True`` if the current path is a checkpoint file, otherwise ``False``.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".ckpt"
    
    def is_config_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a .config or .cfg file, otherwise False.
    
        Args:
            exist: If ``True``, also check if the file exists. Default: ``True``.
    
        Returns:
            ``True`` if the current path is a .config or .cfg file, otherwise ``False``.
        """
        from mon.globals import CONFIG_FILE_FORMATS
        return (not exist or self.is_file()) and self.suffix.lower() in CONFIG_FILE_FORMATS

    def is_dir_like(self) -> bool:
        """Return True if the path is a correct directory format.
    
        Returns:
            ``True`` if the path is a correct directory format, otherwise ``False``.
        """
        return "" in self.suffix
    
    def is_file_like(self) -> bool:
        """Return True if the path is a correct file format.
    
        Returns:
            ``True`` if the path is a correct file format.
        """
        return "." in self.suffix
    
    def is_image_file(self, exist: bool = True) -> bool:
        """Return True if the current path is an image file, otherwise False.
    
        Args:
            exist: If ``True``, also check if the file exists. Default: ``True``.
    
        Returns:
            ``True`` if the current path is an image file, otherwise ``False``.
        """
        from mon.globals import IMAGE_FILE_FORMATS
        return (not exist or self.is_file()) and self.suffix.lower() in IMAGE_FILE_FORMATS
    
    def is_json_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a .json file, otherwise False.
    
        Args:
            exist: If ``True``, also check if the file exists. Default: ``True``.
    
        Returns:
            ``True`` if the current path is a .json file, otherwise ``False``.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".json"
    
    def is_name(self) -> bool:
        """Return True if the current path is the same as the stem, otherwise False.
    
        Returns:
            ``True`` if the current path is the same as the stem, otherwise ``False``.
        """
        return self == self.stem
    
    def is_py_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a .py file, otherwise False.
    
        Args:
            exist: If ``True``, also check if the file exists. Default: ``True``.
    
        Returns:
            ``True`` if the current path is a .py file, otherwise ``False``.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".py"
    
    def is_raw_image_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a raw image file, otherwise False.
    
        Args:
            exist: If ``True``, also check if the file exists. Default: ``True``.
    
        Returns:
            ``True`` if the current path is a raw image file, otherwise ``False``.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in [".dng", ".arw"]
    
    def is_stem(self) -> bool:
        """Return True if the current path is the same as the stem, otherwise False.
    
        Returns:
            ``True`` if the current path is the same as the stem, otherwise ``False``.
        """
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
    
        Args:
            exist: If ``True``, also check if the file exists. Default: ``True``.
    
        Returns:
            ``True`` if the current path is a file with one of the specified extensions, otherwise ``False``.
        """
        from mon.globals import TORCH_FILE_FORMATS
        return (not exist or self.is_file()) and self.suffix.lower() in TORCH_FILE_FORMATS
    
    def is_txt_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a text file, otherwise False.
    
        Args:
            exist: If ``True``, also check if the file exists. Default: ``True``.
    
        Returns:
            ``True`` if the current path is a text file, otherwise ``False``.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".txt"
    
    def is_url(self) -> bool:
        """Return True if the current path is a valid URL, otherwise False.
    
        Returns:
            ``True`` if the current path is a valid URL, otherwise ``False``.
        """
        return not isinstance(validators.url(str(self)), validators.ValidationError)
    
    def is_url_or_file(self, exist: bool = True) -> bool:
        """Return True if the path is a file or a valid URL, otherwise False.
    
        Args:
            exist: If ``True``, also check if the file exists. Default: ``True``.
    
        Returns:
            ``True`` if the path is a file or a valid URL, otherwise ``False``.
        """
        return (not exist or self.is_file()) or not isinstance(validators.url(self), validators.ValidationError)
    
    def is_video_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a video file, otherwise False.
    
        Args:
            exist: If ``True``, also check if the file exists. Default: ``True``.
    
        Returns:
            ``True`` if the current path is a video file, otherwise ``False``.
        """
        from mon.globals import VIDEO_FILE_FORMATS
        return (not exist or self.is_file()) and self.suffix.lower() in VIDEO_FILE_FORMATS.values()
    
    def is_video_stream(self) -> bool:
        """Return True if the current path is a video stream, otherwise False.
    
        Returns:
            ``True`` if the current path is a video stream, otherwise ``False``.
        """
        return "rtsp" in str(self).lower()
    
    def is_weights_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a .pt or .pth file, otherwise False.
    
        Args:
            exist: If ``True``, also check if the file exists. Default: ``True``.
    
        Returns:
            ``True`` if the current path is a .pt or .pth file, otherwise ``False``.
        """
        from mon.globals import WEIGHTS_FILE_FORMATS
        return (not exist or self.is_file()) and self.suffix.lower() in WEIGHTS_FILE_FORMATS
    
    def is_xml_file(self, exist: bool = True) -> bool:
        """Return True if the current path is an .xml file, otherwise False.
    
        Args:
            exist: If ``True``, also check if the file exists. Default: ``True``.
    
        Returns:
            ``True`` if the current path is an .xml file, otherwise ``False``.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".xml"
    
    def is_yaml_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a .yaml or .yml file, otherwise False.
    
        Args:
            exist: If ``True``, also check if the file exists. Default: ``True``.
    
        Returns:
            ``True`` if the current path is a .yaml or .yml file, otherwise ``False``.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in [".yaml", ".yml"]
    
    def has_subdir(self, name: str) -> bool:
        """Return True if a directory has a subdirectory with the given name.
    
        Args:
            name: The name of the subdirectory to check for.
    
        Returns:
            ``True`` if a subdirectory with the given name exists, otherwise ``False``.
        """
        return name in [d.name for d in self.subdirs()]
    
    def subdirs(self, recursive: bool = False) -> list[Path]:
        """Return a list of subdirectories' paths inside the current directory.
    
        Args:
            recursive: If ``True``, include subdirectories recursively. Default: ``False``.
    
        Returns:
            A list of subdirectories' paths.
        """
        path  = self.parent if self.is_file_like() else self
        paths = list(path.rglob("*")) if recursive else list(path.iterdir())
        return [p for p in paths if p.is_dir()]
    
    def files(self, recursive: bool = False) -> list[Path]:
        """Return a list of file paths inside the current directory.
    
        Args:
            recursive: If ``True``, include files in subdirectories. Default: ``False``.
    
        Returns:
            A list of file paths.
        """
        path  = self.parent if self.is_file_like() else self
        paths = list(path.rglob("*")) if recursive else list(path.iterdir())
        return [p for p in paths if p.is_file()]
    
    def ckpt_file(self) -> Path | None:
        """Return the checkpoint file with the given path.
    
        Returns:
            The checkpoint file path if found, otherwise ``None``.
        """
        return get_ckpt_file(self)
    
    def config_file(self) -> Path:
        """Return the configuration file with the given path.
    
        Returns:
            The configuration file path.
        """
        return get_config_file(self)
    
    def latest_file(self) -> Path | None:
        """Return the latest file based on creation time.
    
        Returns:
            The latest file path if files exist, otherwise ``None``.
        """
        files = self.files()
        return max(files, key=os.path.getctime) if files else None
    
    def image_file(self) -> Path:
        """Return the image file with the given path.
    
        Returns:
            The image file path.
        """
        return get_image_file(self)
    
    def relative_path(self, start_part: Path | str) -> Path:
        """Get the relative path from the given part.
    
        Args:
            start_part: The starting part of the relative path.
    
        Returns:
            The relative path from the given part.
        """
        return get_relative_path(self, start_part)
    
    def yaml_file(self) -> Path:
        """Return the YAML file with the given path.
    
        Returns:
            The YAML file path.
        """
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
        """Replace old string with new string in the path.
    
        Args:
            old: The old string to be replaced.
            new: The new string to replace the old string.
            count: The maximum number of occurrences to replace. Default: ``1``.
    
        Returns:
            The new path with the replaced string.
        """
        return Path(str(self).replace(old, new, count))
    
# endregion


# region Check

def is_url(url: str) -> bool:
    """Return True if the URL is valid, otherwise False.

    Args:
        url: The URL to validate.

    Returns:
        ``True`` if the URL is valid, otherwise ``False``.
    """
    return not isinstance(validators.url(url), validators.ValidationError)

# endregion


# region Obtainment

def get_ckpt_file(path: Path) -> Path:
    """Get the ckpt file from the given path.

    Args:
        path: The path to check for a ckpt file.

    Returns:
        The ckpt file path if found, otherwise the original path.
    """
    ckpt_path = path.with_suffix(".ckpt")
    return ckpt_path if ckpt_path.is_yaml_file() else path


def get_config_file(path: Path) -> Path:
    """Get the configuration file from the given path.

    This function attempts to find a configuration file with the same stem as
    the provided path. It checks for both the original stem and its snake_case
    version with various configuration file extensions.

    Args:
        path: The path to check for a configuration file.

    Returns:
        The configuration file path if found, otherwise the original path.
    """
    from mon.globals import CONFIG_FILE_FORMATS
    for ext in CONFIG_FILE_FORMATS:
        for stem in [path.stem, humps.snakecase(path.stem)]:
            config_path = path.with_name(f"{stem}{ext}")
            if config_path.is_config_file():
                return config_path
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
    """Get the image file from the given path.

    This function checks if the given path corresponds to an image file by
    iterating through a list of known image file extensions. If an image file
    is found, it returns the path to that file. Otherwise, it returns the
    original path.

    Args:
        path: The path to check for an image file.

    Returns:
        The path to the image file if found, otherwise the original path.
    """
    from mon.globals import IMAGE_FILE_FORMATS
    for ext in IMAGE_FILE_FORMATS:
        temp = path.with_suffix(ext)
        if temp.is_image_file():
            return temp
    return path


def get_relative_path(path: Path | str, start_part: Path | str) -> Path:
    """Get the relative path starting from the given ``part``.

    Args:
        path: The path to the file.
        start_part: The starting part of the relative path.

    Returns:
        The relative path from the given part.
    """
    path       = Path(path)
    start_part = str(start_part)
    if start_part not in str(path):
        return path
    return Path(*path.parts[path.parts.index(start_part):])
    

def get_yaml_file(path: Path) -> Path:
    """Get the YAML file from the given path.

    This function checks if a YAML file with the same stem as the provided path
    exists in the same directory. It looks for files with `.yaml` and `.yml`
    extensions.

    Args:
        path: The path to check for a YAML file.

    Returns:
        The path to the YAML file if found, otherwise the original path.
    """
    for ext in [".yaml", ".yml"]:
        temp = path.with_suffix(ext)
        if temp.is_yaml_file():
            return temp
    return path


def hash_files(paths: list[Path | str]) -> int:
    """Return the total hash value of all the files (if it has one). Hash
    values are integers (in bytes) of all files.

    Args:
        paths: A list of file paths to hash.

    Returns:
        The total hash value of all the files.
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
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

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
    """Delete directories.

    Args:
        paths: A list of directories' absolute paths.
    """
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
        if p.is_file_like():
            p = p.parent
        delete_files(path=p, regex="*")
        try:
            p.rmdir()
        except Exception as err:
            print(f"Cannot delete directory: {err}.")

# endregion
