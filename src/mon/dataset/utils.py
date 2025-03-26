#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset Utilities.

This module implements data i/o classes and functions.
"""

from __future__ import annotations

__all__ = [
    "parse_io_worker",
]

from mon import core, vision
from mon.dataset import dtype
from mon.globals import DATA_DIR, DATASETS, Split


# region Parsing

def parse_io_worker(
    src        : core.Path | str,
    dst        : core.Path | str,
    to_tensor  : bool            = False,
    denormalize: bool            = False,
    data_root  : core.Path | str = None,
    verbose    : bool            = False
) -> tuple[str, dtype.Dataset, vision.VideoWriterCV]:
    """Parses I/O worker for src and dst.

    Args:
        src: Source of input data.
        dst: Destination path.
        to_tensor: If ``True``, converts to tensor. Default is ``False``.
        denormalize: If ``True``, denormalizes to ``[0, 255]``. Default is ``False``.
        data_root: Dataset root dir (e.g., ``data/ntire_2025_llie``).
        verbose: If ``True``, enables verbose output. Default is ``False``.

    Returns:
        Tuple of data name, loader, and writer.

    Raises:
        ValueError: If ``[src]`` is invalid.
    """
    data_name   : str                  = ""
    data_loader : dtype.Dataset        = None
    data_writer : vision.VideoWriterCV = None
    
    src = core.Path(src)
    if src.stem in DATASETS:
        src = src.stem
        if data_root not in [None, "None", ""] and core.Path(data_root).is_dir():
            root = data_root
        else:
            defaults_dict = dict(zip(
                DATASETS[src].__init__.__code__.co_varnames[1:],
                DATASETS[src].__init__.__defaults__
            ))
            root = defaults_dict.get("root", None)
        if root and not root.is_dir():
            root = DATA_DIR
        
        config = {
            "name"     : src,
            "root"     : root,
            "split"    : Split.TEST,
            "to_tensor": to_tensor,
            "verbose"  : verbose,
        }
        data_name   = src
        data_loader = DATASETS.build(config=config)
    elif src.is_dir() and src.exists():
        data_name   = src.name
        data_loader = dtype.ImageLoader(
            root      = src,
            to_tensor = to_tensor,
            verbose   = verbose
        )
    elif src.is_video_file():
        data_name   = src.name
        data_loader = dtype.VideoLoaderCV(
            root      = src,
            to_tensor = to_tensor,
            verbose   = verbose
        )
        data_writer = vision.VideoWriterCV(
            dst         = core.Path(dst),
            image_size  = data_loader.imgsz,
            frame_rate  = data_loader.fps,
            fourcc      = "mp4v",
            save_image  = False,
            denormalize = denormalize,
            verbose     = verbose
        )
    else:
        raise ValueError(f"[src] is invalid: {src}")
    
    return data_name, data_loader, data_writer
    
# endregion
