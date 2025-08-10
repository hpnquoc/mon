#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements utilities for data I/O classes and functions."""

__all__ = [
    "list_datasets",
    "parse_data_loader",
    "parse_data_name",
]

import torch
from torch.utils import data

import mon.vision as vision
from mon.constants import DATASETS, Split, Task
from mon.core import load_project_defaults, pathlib, types


# ----- Retrieve -----
def list_datasets(task: str, mode: str, project_root: pathlib.Path = None) -> list[str]:
    """Lists all available datasets.

    Args:
        task: Task for which datasets are listed.
        mode: Mode of datasets (``train`` or ``test``).
        project_root: Root directory of the project. Default is ``None``.

    Returns:
        Sorted list of dataset names matching task and mode.
    """
    split    = Split("train" if mode == "train" else "test")
    task     = Task(task)
    datasets = sorted([
        d for d in DATASETS
        if task in DATASETS[d].tasks and split in DATASETS[d].splits
    ])
    
    default_configs = load_project_defaults(project_root)
    if default_configs.get("DATASETS"):
        datasets = [d for d in datasets if d in default_configs["DATASETS"]]
    return datasets


# ----- Convert -----
def parse_data_name(src: pathlib.Path | str) -> str:
    """Parses data name for data src.

    Args:
        src: Source of input data.

    Returns:
        Data name

    Raises:
        ValueError: If ``src`` is invalid.
    """
    src = pathlib.Path(src)

    # Existing dataset
    if src.stem in DATASETS:
        return src.stem
    # Direct input directory or file
    elif src.is_dir():
        data_name = src.stem
    elif src.is_video_file():
        data_name = src.name
    else:
        raise ValueError(f"[src] is invalid: {src}.")

    return data_name


def parse_data_loader(
    src       : pathlib.Path | str,
    data_root : pathlib.Path = None,
    to_tensor : bool         = False,
    batch_size: int          = 1,
    device    : torch.device = None,
    verbose   : bool         = False,
) -> tuple[str, types.Dataset]:
    """Parses I/O worker for data src.

    Args:
        src: Source of input data.
        data_root: Dataset root dir (e.g., ``data/ntire_2025_llie``). Default is ``None``.
        to_tensor: If ``True``, converts to tensor. Default is ``False``.
        batch_size: Number of samples per forward pass. Default is ``1``.
        device: Device to use for data loading. Default is ``None`` (uses CPU).
        verbose: If ``True``, enables verbose output. Default is ``False``.

    Returns:
        Tuple of data name, loader, and writer.

    Raises:
        ValueError: If ``src`` is invalid.
    """
    src = pathlib.Path(src)

    if src.stem in DATASETS:
        src         = src.stem
        root        = pathlib.parse_data_dir(root=data_root, data_dir=src)
        config      = {
            "name"     : src,
            "root"     : root,
            "split"    : Split.TEST,
            "to_tensor": to_tensor,
            "verbose"  : verbose,
        }
        data_name   = src
        data_loader = DATASETS.build(config=config)
    elif src.is_dir():
        data_name   = src.name
        data_loader = vision.ImageLoader(
            root      = src,
            to_tensor = to_tensor,
            verbose   = verbose
        )
    elif src.is_video_file():
        data_name   = src.name
        data_loader = vision.VideoLoaderCV(
            root      = src,
            to_tensor = to_tensor,
            verbose   = verbose
        )
    else:
        raise ValueError(f"[src] is invalid: {src}.")

    if batch_size > 1:
        data_loader = data.DataLoader(
            dataset            = data_loader,
            batch_size         = batch_size,
            shuffle            = False,
            num_workers        = 0 if device is None else 4,
            collate_fn         = getattr(data_loader, "collate_fn"),
            generator          = torch.Generator(device=device),
            persistent_workers = True
        )

    return data_name, data_loader
