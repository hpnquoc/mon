#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pathlib Module.

This module extends Python ``pathlib`` module.
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
    """Extended ``pathlib.Path`` with additional functionalities.

    Notes:
        Methods are kept as methods (not properties) for consistency with ``pathlib.Path``.
    """
    
    def is_basename(self) -> bool:
        """Checks if the path is a file basename.

        Returns:
            ``True`` if the path matches its basename, ``False`` otherwise.
        """
        return str(self) == self.name
    
    def is_bmp_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.bmp`` file.

        Args:
            exist: If ``True``, verifies file existence. Default is ``True``.

        Returns:
            ``True`` if the path is a ``.bmp`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".bmp"
    
    def is_cache_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.cache`` file.

        Args:
            exist: If ``True``, verifies file existence. Default is ``True``.

        Returns:
            ``True`` if the path is a ``.cache`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".cache"
    
    def is_ckpt_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.ckpt`` file.

        Args:
            exist: If ``True``, verifies file existence. Default is ``True``.

        Returns:
            ``True`` if the path is a ``.ckpt`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".ckpt"
    
    def is_config_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.config`` or ``.cfg`` file.

        Args:
            exist: If ``True``, verifies file existence. Default is ``True``.

        Returns:
            ``True`` if the path is a config file, ``False`` otherwise.
        """
        from mon.globals import CONFIG_FILE_FORMATS
        return (not exist or self.is_file()) and self.suffix.lower() in CONFIG_FILE_FORMATS
    
    def is_dir_like(self) -> bool:
        """Checks if the path resembles a directory format.

        Returns:
            ``True`` if the path has no suffix, ``False`` otherwise.
        """
        return self.suffix == ""
    
    def is_file_like(self) -> bool:
        """Checks if the path resembles a file format.

        Returns:
            ``True`` if the path has a suffix, ``False`` otherwise.
        """
        return "." in self.suffix
    
    def is_image_file(self, exist: bool = True) -> bool:
        """Checks if the path is an image file.

        Args:
            exist: If ``True``, verifies file existence. Default is ``True``.

        Returns:
            ``True`` if the path is an image file, ``False`` otherwise.
        """
        from mon.globals import IMAGE_FILE_FORMATS
        return (not exist or self.is_file()) and self.suffix.lower() in IMAGE_FILE_FORMATS
    
    def is_json_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.json`` file.

        Args:
            exist: If ``True``, verifies file existence. Default is ``True``.

        Returns:
            ``True`` if the path is a ``.json`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".json"
    
    def is_name(self) -> bool:
        """Checks if the path matches its stem.

        Returns:
            ``True`` if the path equals its stem, ``False`` otherwise.
        """
        return str(self) == self.stem
    
    def is_py_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.py`` file.

        Args:
            exist: If ``True``, verifies file existence. Default is ``True``.

        Returns:
            ``True`` if the path is a ``.py`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".py"
    
    def is_raw_image_file(self, exist: bool = True) -> bool:
        """Checks if the path is a raw image file (``.dng`` or ``.arw``).

        Args:
            exist: If ``True``, verifies file existence. Default is ``True``.

        Returns:
            ``True`` if the path is a raw image file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in [".dng", ".arw"]
    
    def is_stem(self) -> bool:
        """Checks if the path matches its stem.

        Returns:
            ``True`` if the path equals its stem, ``False`` otherwise.
        """
        return str(self) == self.stem
    
    def is_torch_file(self, exist: bool = True) -> bool:
        """Checks if the path is a Torch-compatible file.

        Args:
            exist: If ``True``, verifies file existence. Default is ``True``.

        Returns:
            ``True`` if the path has a Torch-compatible extension, ``False`` otherwise.
        """
        from mon.globals import TORCH_FILE_FORMATS
        return (not exist or self.is_file()) and self.suffix.lower() in TORCH_FILE_FORMATS
    
    def is_txt_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.txt`` file.

        Args:
            exist: If ``True``, verifies file existence. Default is ``True``.

        Returns:
            ``True`` if the path is a ``.txt`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".txt"
    
    def is_urlMerry(self) -> bool:
        """Checks if the path is a valid URL.

        Returns:
            ``True`` if the path is a valid URL, ``False`` otherwise.
        """
        return not isinstance(validators.url(str(self)), validators.ValidationError)
    
    def is_url_or_file(self, exist: bool = True) -> bool:
        """Checks if the path is a file or valid URL.

        Args:
            exist: If ``True``, verifies file existence. Default is ``True``.

        Returns:
            ``True`` if the path is a file or valid URL, ``False`` otherwise.
        """
        return (not exist or self.is_file()) or not isinstance(validators.url(str(self)), validators.ValidationError)
    
    def is_video_file(self, exist: bool = True) -> bool:
        """Checks if the path is a video file.

        Args:
            exist: If ``True``, verifies file existence. Default is ``True``.

        Returns:
            ``True`` if the path is a video file, ``False`` otherwise.
        """
        from mon.globals import VIDEO_FILE_FORMATS
        return (not exist or self.is_file()) and self.suffix.lower() in VIDEO_FILE_FORMATS.values()
    
    def is_video_stream(self) -> bool:
        """Checks if the path is a video stream.

        Returns:
            ``True`` if the path is a video stream (contains ``rtsp``), ``False`` otherwise.
        """
        return "rtsp" in str(self).lower()
    
    def is_weights_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.pt`` or ``.pth`` file.

        Args:
            exist: If ``True``, verifies file existence. Default is ``True``.

        Returns:
            ``True`` if the path is a weights file, ``False`` otherwise.
        """
        from mon.globals import WEIGHTS_FILE_FORMATS
        return (not exist or self.is_file()) and self.suffix.lower() in WEIGHTS_FILE_FORMATS
    
    def is_xml_file(self, exist: bool = True) -> bool:
        """Checks if the path is an ``.xml`` file.

        Args:
            exist: If ``True``, verifies file existence. Default is ``True``.

        Returns:
            ``True`` if the path is an ``.xml`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".xml"
    
    def is_yaml_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.yaml`` or ``.yml`` file.

        Args:
            exist: If ``True``, verifies file existence. Default is ``True``.

        Returns:
            ``True`` if the path is a ``.yaml`` or ``.yml`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in [".yaml", ".yml"]
    
    def has_subdir(self, name: str) -> bool:
        """Checks if the directory has a subdirectory with the given name.

        Args:
            name: Subdirectory name to check.

        Returns:
            ``True`` if the subdirectory exists, ``False`` otherwise.
        """
        return name in [d.name for d in self.subdirs()]
    
    def subdirs(self, recursive: bool = False) -> list['Path']:
        """Returns a list of subdirectory paths.

        Args:
            recursive: If ``True``, includes subdirectories recursively. Default is ``False``.

        Returns:
            List of subdirectory ``Path`` objects.
        """
        path = self.parent if self.is_file_like() else self
        paths = list(path.rglob("*")) if recursive else list(path.iterdir())
        return [p for p in paths if p.is_dir()]
    
    def files(self, recursive: bool = False) -> list['Path']:
        """Returns a list of file paths in the directory.

        Args:
            recursive: If ``True``, includes files in subdirectories. Default is ``False``.

        Returns:
            List of file ``Path`` objects.
        """
        path = self.parent if self.is_file_like() else self
        paths = list(path.rglob("*")) if recursive else list(path.iterdir())
        return [p for p in paths if p.is_file()]
    
    def ckpt_file(self) -> 'Path' | None:
        """Returns the checkpoint file path if found.

        Returns:
            Checkpoint file ``Path`` or ``None`` if not found.
        """
        return get_ckpt_file(self)
    
    def config_file(self) -> 'Path':
        """Returns the configuration file path.

        Returns:
            Configuration file ``Path``.
        """
        return get_config_file(self)
    
    def latest_file(self) -> 'Path' | None:
        """Returns the latest file based on creation time.

        Returns:
            Latest file ``Path`` or ``None`` if no files exist.
        """
        files = self.files()
        return max(files, key=os.path.getctime) if files else None
    
    def image_file(self) -> 'Path':
        """Returns the image file path.

        Returns:
            Image file ``Path``
        """
        return get_image_file(self)
    
    def relative_path(self, start_part: 'Path' | str) -> 'Path':
        """Returns the relative path from a given start part.

        Args:
            start_part: Starting path or string for relativity.

        Returns:
            Relative ``Path`` from ``start_part``
        """
        return get_relative_path(self, start_part)
    
    def yaml_file(self) -> 'Path':
        """Returns the YAML file path.

        Returns:
            YAML file ``Path``
        """
        return get_yaml_file(self)
    
    def copy_to(self, dst: 'Path' | str, replace: bool = True) -> None:
        """Copies the file to a new location.

        Args:
            dst: Destination path or string.
            replace: If ``True``, replaces the existing file. Default is ``True``.

        Raises:
            NotImplementedError: If ``dst`` is a URL.
        """
        dst = Path(dst)
        if dst.is_url():
            raise NotImplementedError("[dst] as a URL is not supported")
        mkdirs(dst, parents=True, exist_ok=True)
        dst = dst / self.name if dst.is_dir_like() else dst
        if replace:
            dst.unlink(missing_ok=True)
        shutil.copyfile(src=str(self), dst=str(dst))
    
    def replace(self, old: str, new: str, count: int = 1) -> 'Path':
        """Replaces occurrences of a string in the path.

        Args:
            old: String to replace.
            new: Replacement string.
            count: Maximum number of replacements. Default is ``1``.

        Returns:
            New ``Path`` with replaced string.
        """
        return Path(str(self).replace(old, new, count))

# endregion


# region Check

def is_url(url: str) -> bool:
    """Checks if a URL is valid.

    Args:
        url: URL string to validate.

    Returns:
        ``True`` if the URL is valid, ``False`` otherwise.
    """
    return not isinstance(validators.url(url), validators.ValidationError)

# endregion


# region Obtainment

def get_ckpt_file(path: Path) -> Path:
    """Gets the ``.ckpt`` file from the given path.

    Args:
        path: Path to check for a ``.ckpt`` file.

    Returns:
        ``Path`` to the ``.ckpt`` file if found, otherwise the original ``path``.
    """
    ckpt_path = path.with_suffix(".ckpt")
    return ckpt_path if ckpt_path.is_file() else path


def get_config_file(path: Path) -> Path:
    """Gets a configuration file based on the given path's stem.

    Args:
        path: Path to check for a config file.

    Returns:
        ``Path`` to the config file if found, otherwise the original ``path``.
    """
    from mon.globals import CONFIG_FILE_FORMATS
    for ext in CONFIG_FILE_FORMATS:
        for stem in [path.stem, humps.snakecase(path.stem)]:
            config_path = path.with_name(f"{stem}{ext}")
            if config_path.is_file():
                return config_path
    return path


def get_files(regex: str, recursive: bool = False) -> list[Path]:
    """Gets all files matching a regular expression pattern.

    Args:
        regex: File path pattern to match.
        recursive: If ``True``, searches subdirectories. Default is ``False``.

    Returns:
        List of unique ``Path`` objects for matching files.
    """
    paths = [Path(p) for p in glob.glob(regex, recursive=recursive) if Path(p).is_file()]
    return dtype.unique(paths)


def get_image_file(path: Path) -> Path:
    """Gets an image file based on the given path.

    Args:
        path: Path to check for an image file.

    Returns:
        ``Path`` to the image file if found, otherwise the original ``path``.
    """
    from mon.globals import IMAGE_FILE_FORMATS
    for ext in IMAGE_FILE_FORMATS:
        temp = path.with_suffix(ext)
        if temp.is_file():
            return temp
    return path


def get_relative_path(path: Path | str, start_part: Path | str) -> Path:
    """Gets the relative path starting from a given part.

    Args:
        path: Full path to relativize.
        start_part: Starting segment of the relative path.

    Returns:
        Relative ``Path`` from ``start_part`` or the original ``path`` if not found.
    """
    path       = Path(path)
    start_part = str(start_part)
    path_str   = str(path)
    if start_part not in path_str:
        return path
    start_idx = path_str.index(start_part)
    return Path(path_str[start_idx:])


def get_yaml_file(path: Path) -> Path:
    """Gets a YAML file based on the given path.

    Args:
        path: Path to check for a YAML file.

    Returns:
        ``Path`` to the YAML file if found, otherwise the original ``path``.
    """
    for ext in [".yaml", ".yml"]:
        temp = path.with_suffix(ext)
        if temp.is_file():
            return temp
    return path


def hash_files(paths: list[Path | str]) -> int:
    """Calculates the total hash value of files based on their sizes.

    Args:
        paths: List of file paths to hash.

    Returns:
        Integer sum of file sizes in bytes.
    """
    paths = [Path(f) for f in dtype.to_list(paths) if f]
    return sum(f.stat().st_size for f in paths if f.is_file())

# endregion


# region Creation

def copy_file(src: Path | str, dst: Path | str) -> None:
    """Copies a file to a new location.

    Args:
        src: Path to the source file.
        dst: Path to the destination.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

# endregion


# region Alternation

def delete_cache(path: Path | str, recursive: bool = True):
    """Clears cache files in a directory and optionally its subdirectories.

    Args:
        path: Directory path containing cache files.
        recursive: If ``True``, searches subdirectories. Default is ``True``.
    """
    delete_files(path=path, regex=".cache", recursive=recursive)


def delete_dir(paths: Path | str | list[Path | str]):
    """Deletes directories and their contents.

    Args:
        paths: Single path or list of directory paths.
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
    """Deletes files matching a pattern in a directory.

    Args:
        path: Directory path to search for files.
        regex: File path pattern. Default is ``None`` (deletes ``path`` if a file).
        recursive: If ``True``, searches subdirectories. Default is ``False``.
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
            print(f"Cannot delete file: [err].")


def mkdirs(
    paths   : Path | str | list[Path | str],
    mode    : int  = 0o777,
    parents : bool = True,
    exist_ok: bool = True,
    replace : bool = False,
):
    """Creates directories with specified options.

    Args:
        paths: Single path or list of directory paths.
        mode: File mode combined with umask. Default is ``0o777``.
        parents: If ``True``, creates parent directories. Default is ``True``.
        exist_ok: If ``True``, ignores existing directories. Default is ``True``.
        replace: If ``True``, deletes and recreates existing directories. Default is ``False``.
    """
    paths = dtype.unique(dtype.to_list(paths))
    for p in paths:
        p = Path(p)
        if p.is_url():
            continue
        p = p.parent if p.is_file_like() else p
        if replace and p.exists():
            delete_files(path=p, regex="*")
            p.rmdir()
        p.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)


def rmdirs(paths: Path | str | list[Path | str]):
    """Deletes directories and their contents.

    Args:
        paths: Single path or list of directory paths.
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
            print(f"Cannot delete directory: [err].")

# endregion
