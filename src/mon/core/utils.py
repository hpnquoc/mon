#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Provides utility functions and data structures."""

from __future__ import annotations

__all__ = [
    "Timer",
    "check_installed_package",
    "download_weights_from_url",
    "get_epoch_from_checkpoint",
    "get_global_step_from_checkpoint",
    "get_gpu_device_memory",
    "get_latest_checkpoint",
    "get_machine_memory",
    "get_model_device",
    "get_project_default_config",
    "is_extra_model",
    "is_rank_zero",
    "list_archs",
    "list_config_files",
    "list_configs",
    "list_cuda_devices",
    "list_datasets",
    "list_devices",
    "list_extra_archs",
    "list_extra_datasets",
    "list_extra_models",
    "list_models",
    "list_mon_archs",
    "list_mon_datasets",
    "list_mon_models",
    "list_tasks",
    "list_weights_files",
    "load_config",
    "parse_config_file",
    "parse_data_dir",
    "parse_device",
    "parse_menu_string",
    "parse_model_dir",
    "parse_model_fullname",
    "parse_model_name",
    "parse_save_dir",
    "parse_weights_file",
    "pynvml_available",
    "set_device",
    "set_random_seed",
]

import importlib
import importlib.util
import os
import random
import time
from typing import Any, Collection, Sequence

import numpy as np
import psutil
import torch
from torch import nn

from mon.core import dtype, file, humps, pathlib, rich
from mon.globals import MemoryUnit

try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False


# region Checkpoint

def get_epoch_from_checkpoint(ckpt: pathlib.Path) -> int:
    """Gets the epoch value from a checkpoint file.

    Args:
        ckpt: Path to the checkpoint file.

    Returns:
        Epoch value from checkpoint, or ``0`` if not found or invalid.
    """
    if ckpt is None:
        return 0
    
    ckpt = pathlib.Path(ckpt)
    if ckpt.is_torch_file():
        return torch.load(ckpt).get("epoch", 0)
    
    return 0


def get_global_step_from_checkpoint(ckpt: pathlib.Path) -> int:
    """Gets the global step from a checkpoint file.

    Args:
        ckpt: Path to the checkpoint file.

    Returns:
        Global step from checkpoint, or ``0`` if not found or invalid.
    """
    if ckpt is None:
        return 0
    
    ckpt = pathlib.Path(ckpt)
    if ckpt.is_torch_file():
        return torch.load(ckpt).get("global_step", 0)
    
    return 0


def get_latest_checkpoint(dirpath: pathlib.Path) -> str | None:
    """Gets the latest checkpoint file path in a directory.

    Args:
        dirpath: Directory path containing checkpoint files.

    Returns:
        Path to latest checkpoint as string, or ``None`` if none found.
    """
    dirpath = pathlib.Path(dirpath)
    ckpts = sorted(
        (ckpt for ckpt in dirpath.files(recursive=True) if ckpt.is_torch_file()),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not ckpts:
        rich.error_console.log(f"[red]Cannot find checkpoint file: {dirpath}.")
        return None
    
    return str(ckpts[0])

# endregion


# region Config

def get_project_default_config(project_root: str | pathlib.Path) -> dict:
    """Gets the default configuration of the project.

    Args:
        project_root: Root directory of the project.

    Returns:
        Dict with default config, or empty dict if invalid or not found.
    """
    if project_root in [None, "None", ""]:
        from mon.core.rich import error_console
        error_console.log(f"[project_root] is not a valid project directory: {project_root}.")
        return {}
    
    config_file = pathlib.Path(project_root) / "config" / "default.py"
    if not config_file.exists():
        return {}
    
    spec   = importlib.util.spec_from_file_location("default", str(config_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return {
        key: value
        for key, value in module.__dict__.items()
        if not key.startswith('__')
    }


def list_config_files(
    project_root: str | pathlib.Path,
    model_root  : str | pathlib.Path = None,
    model       : str                = None
) -> list[pathlib.Path]:
    """Lists configuration files in the project and/or model directory.

    Args:
        project_root: Root directory of the project.
        model_root: Root directory of the model. Default is ``None``.
        model: Name of the model to filter configs. Default is ``None``.

    Returns:
        Sorted list of config file ``Path`` objects.
    """
    def is_valid(x) -> bool:
        return x not in [None, "", "None"]

    def collect_config_files(root: str | pathlib.Path) -> list[pathlib.Path]:
        config_dir = pathlib.Path(root) / "config"
        return list(config_dir.files(recursive=True))

    config_files = []
    if is_valid(project_root):
        config_files += collect_config_files(project_root)
    if is_valid(model_root):
        config_files += collect_config_files(model_root)

    config_files = [
        cf for cf in config_files
        if cf.is_config_file() or (cf.is_py_file() and cf.name != "__init__.py")
    ]
    
    if is_valid(model):
        model_name   = parse_model_name(model)
        config_files = [cf for cf in config_files if model_name in cf.name]

    return sorted(dtype.unique(config_files))


def list_configs(
    project_root: str | pathlib.Path,
    model_root  : str | pathlib.Path = None,
    model       : str                = None
) -> list[str]:
    """Lists config file names in the project and/or model directory.

    Args:
        project_root: Root directory of the project.
        model_root: Root directory of the model. Default is ``None``.
        model: Name of the model to filter configs. Default is ``None``.

    Returns:
        Sorted list of config file names as strings.
    """
    config_files = list_config_files(
        project_root = project_root,
        model_root   = model_root,
        model        = model
    )
    return sorted(
        dtype.unique([str(cf.name) for cf in config_files]),
        key = lambda x: (os.path.splitext(x)[1], x)
    )


def parse_config_file(
    config      : str | pathlib.Path,
    project_root: str | pathlib.Path,
    model_root  : str | pathlib.Path = None,
    weights_path: str | pathlib.Path = None,
) -> pathlib.Path | None:
    """Parses the config file from the given paths.

    Args:
        config: Config file path or name.
        project_root: Root directory of the project.
        model_root: Root directory of the model. Default is ``None``.
        weights_path: Path to weights file. Default is ``None``.

    Returns:
        ``Path`` to config file if found, else ``None``.
    """
    from mon.core.rich import error_console
    
    def find_config_in_dirs(config, dirs):
        for config_dir in dirs:
            config_ = (config_dir / config.name).config_file()
            if config_.is_config_file():
                return config_
        return None
    
    if config:
        config = pathlib.Path(config)
        if config.is_config_file():
            return config
        config_ = config.config_file()
        if config_.is_config_file():
            return config_
        if project_root:
            config_dirs = [pathlib.Path(project_root / "config")] + \
                          pathlib.Path(project_root / "config").subdirs(recursive=True)
            config_ = find_config_in_dirs(config, config_dirs)
            if config_:
                return config_
        if model_root:
            config_dirs = [pathlib.Path(model_root / "config")] + \
                          pathlib.Path(model_root / "config").subdirs(recursive=True)
            config_ = find_config_in_dirs(config, config_dirs)
            if config_:
                return config_
    
    if weights_path:
        weights_path = pathlib.Path(weights_path[0] if isinstance(weights_path, list) else weights_path)
        if weights_path.is_weights_file():
            config_ = (weights_path.parent / "config.py").config_file()
            if config_.is_config_file():
                return config_
    
    error_console.log(
        f"Could not find configuration file given: "
        f"config={config}, project_root={project_root}, "
        f"model_root={model_root}, weights_path={weights_path}"
    )
    return None


def load_config(config: Any) -> dict:
    """Loads configuration from a given source.

    Args:
        config: Config source (dict, file path, or string).

    Returns:
        Dict with loaded config, or empty dict if loading fails.
    """
    from mon.core.rich import error_console, console
    
    if isinstance(config, dict):
        data = config
    elif isinstance(config, (pathlib.Path, str)):
        config = pathlib.Path(config)
        if config.is_py_file():
            spec   = importlib.util.spec_from_file_location(str(config.stem), str(config))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            data   = {key: value for key, value in module.__dict__.items() if not key.startswith("__")}
        else:
            data = file.read_from_file(path=config)
    else:
        data = None
    
    if data:
        console.log(f"Loaded configuration from: {config}.")
    else:
        error_console.log(f"Could not load configuration from: {config}. "
                          f"Returning empty dict.")
        data = {}

    return data

# endregion


# region Datasets

def list_mon_datasets(task: str, mode: str) -> list[str]:
    """Lists all available datasets in the ``mon`` framework.

    Args:
        task: Task for which datasets are listed.
        mode: Mode of datasets (``train`` or ``test``).

    Returns:
        Sorted list of dataset names matching task and mode.
    """
    from mon.globals import Task, Split, DATASETS

    split    = Split("train" if mode == "train" else "test")
    task     = Task(task)
    datasets = DATASETS

    return sorted([
        d for d in datasets
        if task in datasets[d].tasks and split in datasets[d].splits
    ])


def list_extra_datasets(task: str, mode: str) -> list[str]:
    """Lists all available datasets in the ``extra`` framework.

    Args:
        task: Task for which datasets are listed.
        mode: Mode of datasets (``train`` or ``test``).

    Returns:
        Sorted list of dataset names matching task and mode.
    """
    from mon.globals import Task, Split, EXTRA_DATASETS

    split    = Split("train" if mode == "train" else "test")
    task     = Task(task)
    datasets = EXTRA_DATASETS

    return sorted([
        d for d in datasets
        if task in datasets[d]["tasks"] and split in datasets[d]["splits"]
    ])


def list_datasets(
    task        : str,
    mode        : str,
    project_root: str | pathlib.Path = None
) -> list[str]:
    """Lists all available datasets.

    Args:
        task: Task for which datasets are listed.
        mode: Mode of datasets (``train`` or ``test``).
        project_root: Root directory of project. Default is ``None``.

    Returns:
        Sorted list of dataset names matching task and mode.
    """
    datasets        = sorted(list_mon_datasets(task, mode) + list_extra_datasets(task, mode))
    default_configs = get_project_default_config(project_root)
    if default_configs.get("DATASETS"):
        datasets = [d for d in datasets if d in default_configs["DATASETS"]]
    return datasets


def parse_data_dir(
    root    : str | pathlib.Path,
    data_dir: str | pathlib.Path
) -> str | pathlib.Path:
    """Parses the absolute data directory path from given components.

    Args:
        root: Root directory.
        data_dir: Data directory.

    Returns:
        Parsed the absolute path of the data directory.
    """
    from mon.globals import ROOT_DIR
    
    root     = pathlib.Path(root)
    data_dir = pathlib.Path(data_dir)
    if not data_dir.is_dir():
        if (ROOT_DIR / data_dir).is_dir():
            data_dir = ROOT_DIR / data_dir
        elif (root / data_dir).is_dir():
            data_dir = root / data_dir
    return data_dir

# endregion


# region Device

def is_rank_zero() -> bool:
    """Checks if current process is rank zero in distributed training.

    Notes:
        Based on PyTorch Lightning's DDP documentation, "LOCAL_RANK" and "NODE_RANK"
        environment variables indicate child processes for GPUs. Absence of both
        denotes the main process (rank zero).

    Returns:
        ``True`` if current process is rank zero, ``False`` otherwise.
    """
    return "LOCAL_RANK" not in os.environ and "NODE_RANK" not in os.environ


def list_cuda_devices() -> str | None:
    """Lists all available CUDA devices on the machine.

    Returns:
        String of CUDA devices (e.g., ``cuda:0,1,2``) or ``None`` if none.
    """
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        cuda_str    = "cuda:" + ",".join(str(i) for i in range(num_devices))
        return cuda_str
    return None


def list_devices() -> list[str]:
    """Lists all available devices on the machine.

    Returns:
        List of device strings including ``auto``, ``cpu``, and CUDA if available.
    """
    devices = ["auto", "cpu"]
    if torch.cuda.is_available():
        num_devices  = torch.cuda.device_count()
        devices.extend(f"cuda:{i}" for i in range(num_devices))
        all_cuda_str = "cuda:" + ",".join(str(i) for i in range(num_devices))
        if all_cuda_str != "cuda:0":
            devices.append(all_cuda_str)
    return devices


def set_device(device: Any, use_single_device: bool = True) -> torch.device:
    """Sets the device for the current process.

    Args:
        device: Device to set (e.g., CUDA index, list, or string).
        use_single_device: If ``True``, uses first device from list. Default is ``True``.

    Returns:
        Selected ``torch.device``, defaults to ``cpu`` if CUDA unavailable.
    """
    device = parse_device(device)
    if isinstance(device, list) and use_single_device:
        device = device[0]
    return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")


def get_machine_memory(unit: MemoryUnit = MemoryUnit.GB) -> list[int]:
    """Gets RAM status as a list of total, used, and free memory.

    Args:
        unit: Memory unit (e.g., ``GB``). Default is ``MemoryUnit.GB``.

    Returns:
        List of [total, used, free] memory values in specified unit.
    """
    memory = psutil.virtual_memory()
    ratio  = MemoryUnit.byte_conversion_mapping()[MemoryUnit.from_value(unit)]
    return [
        memory.total     / ratio,  # total
        memory.used      / ratio,  # used
        memory.available / ratio   # free
    ]


def get_gpu_device_memory(device: int = 0, unit: MemoryUnit = MemoryUnit.GB) -> list[int]:
    """Gets GPU memory status as a list of total, used, and free memory.

    Args:
        device: GPU device index. Default is ``0``.
        unit: Memory unit (e.g., ``GB``). Default is ``MemoryUnit.GB``.

    Returns:
        List of [total, used, free] memory values in specified unit.
    """
    pynvml.nvmlInit()
    unit  = MemoryUnit.from_value(unit)
    info  = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(device))
    ratio = MemoryUnit.byte_conversion_mapping()[unit]
    return [
        info.total / ratio,  # total
        info.used  / ratio,  # used
        info.free  / ratio   # free
    ]


def parse_device(device: Any) -> list[int] | int | str:
    """Parses a device spec into a list, integer, or string.

    Args:
        device: Device to parse (e.g., ``torch.device``, int, str, or ``None``).

    Returns:
        List of ints for multi-device, int for single, or str (``cpu`` or ``mps``).
    """
    if isinstance(device, torch.device):
        return device
    
    if not device or device in ["", "cpu"]:
        return "cpu"
    if device in ["mps", "mps:0"]:
        return device
    if isinstance(device, int):
        return [device]
    if isinstance(device, str):
        device = (device.lower()
                  .replace("cuda:", "")
                  .replace("none", "")
                  .translate(str.maketrans("", "", "()[ ]' ")))
        return [int(x) for x in device.split(",")] \
            if "," in device \
            else [0] if not device else device
    return device


def get_model_device(model: nn.Module) -> torch.device:
    """Gets the device of a model's parameters.

    Args:
        model: Model to check.

    Returns:
        ``torch.device`` where model parameters reside.
    """
    return next(model.parameters()).device
    
# endregion


# region Menu

def parse_menu_string(items: Sequence | Collection, num_columns: int = 4) -> str:
    """Parses a list of items into a formatted menu string.

    Args:
        items: Items to display in the menu.
        num_columns: Number of columns for menu layout. Default is ``4``.

    Returns:
        Formatted menu string.
    """
    s = "\n  "
    for i, item in enumerate(items):
        s += f"{f'{i}.':>6} {item}\n  "
    s += f"{f'Other.':} (please specify)\n  "
    return s

# endregion


# region Models

def is_extra_model(model: str) -> bool:
    """Checks if a model is an extra model.

    Args:
        model: Name of the model to check.

    Returns:
        ``True`` if model is extra, ``False`` otherwise.
    """
    from mon.globals import MODELS, EXTRA_MODELS, EXTRA_MODEL_STR
    
    model        = model.replace(f" {EXTRA_MODEL_STR}", "").strip()
    mon_models   = dtype.flatten_models_dict(MODELS)
    extra_models = dtype.flatten_models_dict(EXTRA_MODELS)
    return (
        f"{EXTRA_MODEL_STR}" in model
        or (model not in mon_models and model in extra_models)
    )


def list_mon_models(task: str = None, mode: str = None, arch: str = None) -> list[str]:
    """Lists all available models in the ``mon`` framework.

    Args:
        task: Task to filter models. Default is ``None``.
        mode: Mode of models (``train`` or ``None``). Default is ``None``.
        arch: Arch to filter models. Default is ``None``.

    Returns:
        Sorted list of model names matching task, mode, and arch.
    """
    from mon.globals import Task, MODELS, LType
    
    flatten_models = dtype.flatten_models_dict(MODELS)
    models         = list(flatten_models.keys())
    
    if task in Task.values():
        task   = Task(task)
        models = [m for m in models if task in flatten_models[m].tasks]
   
    if mode == "train":
        models = [m for m in models
                  if any(lt in LType.trainable() for lt in flatten_models[m].ltypes)]
    
    if arch:
        models = [m for m in models if arch in flatten_models[m].arch]
        
    return sorted(models)


def list_extra_models(task: str = None, mode: str = None, arch: str = None) -> list[str]:
    """Lists all available models in the ``extra`` framework.

    Args:
        task: Task to filter models. Default is ``None``.
        mode: Mode of models (``train`` or ``None``). Default is ``None``.
        arch: Arch to filter models. Default is ``None``.

    Returns:
        Sorted list of model names matching task, mode, and arch.
    """
    from mon.globals import Task, EXTRA_MODELS, LType
    
    flatten_models = dtype.flatten_models_dict(EXTRA_MODELS)
    models         = list(flatten_models.keys())
   
    if task in Task.values():
        task   = Task(task)
        models = [m for m in models if task in flatten_models[m]["tasks"]]
   
    if mode == "train":
        models = [m for m in models
                  if any(lt in LType.trainable() for lt in flatten_models[m]["ltypes"])]
    
    if arch:
        models = [m for m in models if arch in flatten_models[m]["arch"]]
   
    return sorted(models)


def list_models(
    task        : str = None,
    mode        : str = None,
    arch        : str = None,
    project_root: str | pathlib.Path = None
) -> list[str]:
    """Lists all available models in ``mon`` and ``extra`` frameworks.

    Args:
        task: Task to filter models. Default is ``None``.
        mode: Mode of models (``train`` or ``None``). Default is ``None``.
        arch: Arch to filter models. Default is ``None``.
        project_root: Root dir of project. Default is ``None``.

    Returns:
        Sorted list of model names matching task, mode, and arch.
    """
    from mon.globals import EXTRA_MODEL_STR
    
    models       = list_mon_models(task=task, mode=mode, arch=arch)
    extra_models = list_extra_models(task=task, mode=mode, arch=arch)
    
    default_configs = get_project_default_config(project_root=project_root)
    if default_configs.get("MODELS"):
        project_models = [humps.snakecase(m) for m in default_configs["MODELS"]]
        models         = [m for m in models       if humps.snakecase(m) in project_models]
        extra_models   = [m for m in extra_models if humps.snakecase(m) in project_models]
        
    for i, m in enumerate(extra_models):
        if m in models:
            extra_models[i] = f"{m} {EXTRA_MODEL_STR}"
            
    return sorted(models + extra_models)


def list_mon_archs(task: str = None, mode: str = None) -> list[str]:
    """Lists all available architectures in the ``mon`` framework.

    Args:
        task: Task to filter archs. Default is ``None``.
        mode: Mode of archs (``train`` or ``None``). Default is ``None``.

    Returns:
        Sorted list of unique arch names matching task and mode.
    """
    from mon.globals import Task, MODELS, LType
    
    flatten_models = dtype.flatten_models_dict(MODELS)
    models         = list(flatten_models.keys())
    
    if task in Task.values():
        task   = Task(task)
        models = [m for m in models if task in flatten_models[m].tasks]
    
    if mode == "train":
        models = [m for m in models
                  if any(lt in LType.trainable() for lt in flatten_models[m].ltypes)]
    
    archs = [flatten_models[m].arch.strip()
             for m in models
             if flatten_models[m].arch not in [None, "None", ""]]
    
    return sorted(dtype.unique(archs))


def list_extra_archs(task: str = None, mode: str = None) -> list[str]:
    """Lists all available architectures in the ``extra`` framework.

    Args:
        task: Task to filter archs. Default is ``None``.
        mode: Mode of archs (``train`` or ``None``). Default is ``None``.

    Returns:
        Sorted list of unique arch names matching task and mode.
    """
    from mon.globals import Task, EXTRA_MODELS, LType
    
    flatten_models = dtype.flatten_models_dict(EXTRA_MODELS)
    models         = list(flatten_models.keys())
    
    if task in Task.values():
        task   = Task(task)
        models = [m for m in models if task in flatten_models[m]["tasks"]]
    
    if mode == "train":
        models = [m for m in models
                  if any(lt in LType.trainable() for lt in flatten_models[m]["ltypes"])]
    
    archs = [flatten_models[m]["arch"].strip()
             for m in models if flatten_models[m]["arch"] not in [None, "None", ""]]
    
    return sorted(dtype.unique(archs))


def list_archs(
    task        : str = None,
    mode        : str = None,
    project_root: str | pathlib.Path = None
) -> list[str]:
    """Lists all available architectures in ``mon`` and ``extra`` frameworks.

    Args:
        task: Task to filter archs. Default is ``None``.
        mode: Mode of archs (``train`` or ``None``). Default is ``None``.
        project_root: Root dir of project. Default is ``None``.

    Returns:
        Sorted list of unique arch names matching task and mode.
    """
    from mon.globals import MODELS, EXTRA_MODELS
    
    models       = list_mon_models(task=task, mode=mode)
    extra_models = list_extra_models(task=task, mode=mode)
    
    default_configs = get_project_default_config(project_root=project_root)
    if default_configs.get("MODELS"):
        project_models = [humps.snakecase(m) for m in default_configs["MODELS"]]
        models         = [m for m in models       if humps.snakecase(m) in project_models]
        extra_models   = [m for m in extra_models if humps.snakecase(m) in project_models]
    
    flatten_mon_models   = dtype.flatten_models_dict(MODELS)
    flatten_extra_models = dtype.flatten_models_dict(EXTRA_MODELS)
    archs = (
        [flatten_mon_models[m].arch      for m in models] +
        [flatten_extra_models[m]["arch"] for m in extra_models]
    )
    archs = [a.strip() for a in archs if a not in [None, "None", ""]]
    
    return sorted(dtype.unique(archs))


def parse_model_dir(arch: str, model: str) -> pathlib.Path | None:
    """Parses the model's directory from given components.

    Args:
        arch: Architecture of the model.
        model: Name of the model.

    Returns:
        ``Path`` to model dir if found, else ``None``.
    """
    from mon.globals import EXTRA_MODELS, MODELS
    
    model_name = parse_model_name(model)
    model_dir  = (
        EXTRA_MODELS[arch][model_name].get("model_dir")
        if is_extra_model(model)
        else MODELS[arch][model_name].model_dir
    )
    return pathlib.Path(model_dir) if model_dir else None


def parse_model_name(model: str) -> str:
    """Parses the model's name from given components.

    Args:
        model: Model name to parse.

    Returns:
        Parsed model name as a string.
    """
    from mon.globals import EXTRA_MODEL_STR
    
    return model.replace(f" {EXTRA_MODEL_STR}", "").strip()


def parse_model_fullname(name: str, data: str, suffix: str = None) -> str:
    """Parses the model's full name as ``name-data-suffix`` from components.

    Args:
        name: Model's base name.
        data: Dataset name.
        suffix: Optional suffix for model name. Default is ``None``.

    Returns:
        Parsed full model name as a string.
    """
    if not name:
        rich.error_console.log("[name] must be provided for the model")
    
    fullname = name
    if data:
        fullname = f"{fullname}_{data}"
    if suffix:
        _fullname = humps.snakecase(fullname)
        _suffix   = humps.snakecase(suffix)
        if _suffix not in _fullname:
            fullname = f"{fullname}_{humps.kebabize(suffix)}"
    return fullname

# endregion


# region Package

def check_installed_package(package_name: str, verbose: bool = False) -> bool:
    """Checks if a package is installed.

    Args:
        package_name: Name of the package to check.
        verbose: If ``True``, prints install status. Default is ``False``.

    Returns:
        ``True`` if package is installed, ``False`` otherwise.
    """
    try:
        importlib.import_module(package_name)
        if verbose:
            print(f"[{package_name}] is installed")
        return True
    except ImportError:
        if verbose:
            print(f"[{package_name}] is not installed")
        return False

# endregion


# region Save Dir

def list_train_save_dirs(root: str | pathlib.Path) -> list[pathlib.Path]:
    """Lists all training save directories in the given project.

    Args:
        root: Root directory of the project.

    Returns:
        Sorted list of training save dir ``Path`` objects.
    """
    root = pathlib.Path(root)
    return sorted((root / "run" / "train").dirs())


def parse_save_dir(
    root : str | pathlib.Path,
    arch : str = None,
    model: str = None,
    data : str = None,
) -> str | pathlib.Path:
    """Parses a save dir in format: root/arch/model/data.

    Args:
        root: Project root.
        arch: Model architecture. Default is ``None``.
        model: Model name. Default is ``None``.
        data: Dataset name. Default is ``None``.

    Returns:
        Parsed save dir path as ``str`` or ``pathlib.Path``.
    """
    save_dir = pathlib.Path(root)
    if arch:
        save_dir /= arch
    if model:
        save_dir /= model
        if data:
            save_dir /= data
    return save_dir

# endregion


# region Seed

def set_random_seed(seed: int | list[int] | tuple[int, int]) -> None:
    """Sets random seeds for various libraries.

    Args:
        seed: Int, list of ints, or tuple of two ints for range selection.
    """
    if isinstance(seed, (list, tuple)):
        seed = random.randint(seed[0], seed[1]) if len(seed) == 2 else seed[-1]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# endregion


# region Tasks

def list_tasks(project_root: str | pathlib.Path) -> list[str]:
    """Lists all available tasks in the project.

    Args:
        project_root: Root directory of the project.

    Returns:
        Sorted list of task names as strings.
    """
    from mon.globals import Task
    
    tasks = Task.keys()
    
    default_configs = get_project_default_config(project_root)
    if default_configs.get("TASKS"):
        tasks = [t for t in tasks if t in default_configs["TASKS"]]
    
    return sorted(t.value for t in tasks)

# endregion


# region Timer

class Timer:
    """A simple timer.
    
    Attributes:
        start_time: The start time of the current call.
        end_time: The end time of the current call.
        total_time: The total time of the timer.
        calls: The number of calls.
        diff_time: The difference time of the call.
        avg_time: The total average time.
    """
    
    def __init__(self):
        self.start_time = 0.0
        self.end_time   = 0.0
        self.total_time = 0.0
        self.calls      = 0
        self.diff_time  = 0.0
        self.avg_time   = 0.0
        self.duration   = 0.0
    
    @property
    def total_time_m(self) -> float:
        return self.total_time / 60.0
    
    @property
    def total_time_h(self) -> float:
        return self.total_time / 3600.0
    
    @property
    def avg_time_m(self) -> float:
        return self.avg_time / 60.0
    
    @property
    def avg_time_h(self) -> float:
        return self.avg_time / 3600.0
    
    @property
    def duration_m(self) -> float:
        return self.duration / 60.0
    
    @property
    def duration_h(self) -> float:
        return self.duration / 3600.0
    
    def start(self):
        self.clear()
        self.tick()
    
    def end(self) -> float:
        self.tock()
        return self.avg_time
    
    def tick(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()
    
    def tock(self, average: bool = True) -> float:
        self.end_time    = time.time()
        self.diff_time   = self.end_time - self.start_time
        self.total_time += self.diff_time
        self.calls      += 1
        self.avg_time    = self.total_time / self.calls
        if average:
            self.duration = self.avg_time
        else:
            self.duration = self.diff_time
        return self.duration
    
    def clear(self):
        self.start_time = 0.0
        self.end_time   = 0.0
        self.total_time = 0.0
        self.calls      = 0
        self.diff_time  = 0.0
        self.avg_time   = 0.0
        self.duration   = 0.0

# endregion


# region Weights

def download_weights_from_url(
    url      : str,
    path     : pathlib.Path,
    overwrite: bool = False
) -> pathlib.Path:
    """Downloads weights from a URL to a local path.

    Args:
        url: URL to download weights from.
        path: Local file path to save weights.
        overwrite: If ``True``, overwrites existing file. Default is ``False``.

    Returns:
        Path to downloaded weights file.

    Raises:
        ValueError: If ``url`` is not a valid URL.
    """
    if not pathlib.Path(url).is_url():
        raise ValueError(f"[url] must be a valid URL, got {url}.")
    
    path = pathlib.Path(path)
    if not path.exists() or overwrite:
        pathlib.delete_files(path=path.parent, regex=path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, path, None, True)
    return path


def list_weights_files(
    model       : str,
    project_root: str | pathlib.Path = None
) -> list[pathlib.Path]:
    """Lists weights files for a model in project and ``zoo`` dirs.

    Args:
        model: Name of model to filter weights files.
        project_root: Root dir of project. Default is ``None``.

    Returns:
        Sorted list of weights file ``Path`` objects.
    """
    from mon.globals import ZOO_DIR
    
    def collect_weights_files(root: pathlib.Path) -> list[pathlib.Path]:
        return sorted(f for f in root.rglob("*") if f.is_weights_file())
    
    weights_files = []
    if project_root not in [None, "None", ""]:
        weights_files += collect_weights_files(pathlib.Path(project_root) / "run" / "train")
    
    weights_files += collect_weights_files(ZOO_DIR)
    
    model_name = parse_model_name(model)
    return sorted(dtype.unique([f for f in weights_files if model_name in str(f)]))


def parse_weights_file(
    root   : str | pathlib.Path,
    weights: str | pathlib.Path | Sequence[str | pathlib.Path]
) -> str | pathlib.Path | Sequence[str | pathlib.Path]:
    """Parses weights file path(s) from given components.

    Args:
        root: Root directory.
        weights: Weights file(s) to parse (str, ``Path``, or sequence).

    Returns:
        Parsed weights path(s) as single path or sequence, or ``None`` if empty.
    """
    from mon.globals import ROOT_DIR
    
    root = pathlib.Path(root)
    weights = dtype.to_list(weights)
    
    for i, w in enumerate(weights):
        w = pathlib.Path(w)
        if not w.is_weights_file(exist=True):
            weights[i] = (ROOT_DIR / w) \
                if (ROOT_DIR / w).is_weights_file(exist=True) \
                else (root / w)
    
    if len(weights) == 1:
        return weights[0]
    return weights or None

# endregion
