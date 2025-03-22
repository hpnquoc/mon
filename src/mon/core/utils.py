#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core Utilities Package.

This module implements various useful utilities functions and data structures.
"""

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

import mon
from mon.core import dtype, file, humps, pathlib, rich
from mon.globals import MemoryUnit

try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False


# region Checkpoint

def get_epoch_from_checkpoint(ckpt: pathlib.Path) -> int:
    """Get an epoch value stored in a checkpoint file."""
    if ckpt is None:
        return 0
    
    ckpt = pathlib.Path(ckpt)
    if ckpt.is_torch_file():
        return torch.load(ckpt).get("epoch", 0)
    
    return 0
 

def get_global_step_from_checkpoint(ckpt: pathlib.Path) -> int:
    """Get a global step stored in a checkpoint file."""
    if ckpt is None:
        return 0
    
    ckpt = pathlib.Path(ckpt)
    if ckpt.is_torch_file():
        return torch.load(ckpt).get("global_step", 0)
    
    return 0


def get_latest_checkpoint(dirpath: pathlib.Path) -> str | None:
    """Get the latest checkpoint (last saved) file path in a directory."""
    dirpath = pathlib.Path(dirpath)
    ckpts   = sorted(
        (ckpt for ckpt in dirpath.files(recursive=True) if ckpt.is_torch_file()),
        key     = lambda x: x.stat().st_mtime,
        reverse = True
    )
    
    if not ckpts:
        rich.error_console.log(f"[red]Cannot find checkpoint file: {dirpath}.")
        return None
    
    return ckpts[0]

# endregion


# region Config

def get_project_default_config(project_root: str | pathlib.Path) -> dict:
    """Get the default configuration of the project."""
    if project_root in [None, "None", ""]:
        from mon.core.rich import error_console
        error_console.log(f"{project_root} is not a valid project directory.")
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
    """List configuration files (absolute paths) in the given project and/or
    model directory.
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
    """List configuration files in the given project and/or model directory."""
    # List model's configuration files (absolute paths)
    config_files = list_config_files(project_root, model_root, model)
    
    # Sort and return the configuration names
    return sorted(
        dtype.unique([str(cf.name) for cf in config_files]),
        key=lambda x: (os.path.splitext(x)[1], x)
    )
    

def parse_config_file(
    config      : str | pathlib.Path,
    project_root: str | pathlib.Path,
    model_root  : str | pathlib.Path = None,
    weights_path: str | pathlib.Path = None,
) -> pathlib.Path | None:
    from mon.core.rich import error_console
    
    def find_config_in_dirs(config, dirs):
        for config_dir in dirs:
            config_ = (config_dir / config.name).config_file()
            if config_.is_config_file():
                return config_
        return None
    
    if config:
        # Check `config` itself.
        config = pathlib.Path(config)
        if config.is_config_file():
            return config
        # Check for other config file extensions in the same directory.
        config_ = config.config_file()
        if config_.is_config_file():
            return config_
        # Check for config file in `'config'` directory in `project_root`.
        if project_root:
            config_dirs = [pathlib.Path(project_root / "config")] + \
                           pathlib.Path(project_root / "config").subdirs(recursive=True)
            config_ = find_config_in_dirs(config, config_dirs)
            if config_:
                return config_
        # Check for config file in `'config'` directory in `model_root`.
        if model_root:
            config_dirs = [pathlib.Path(model_root / "config")] + \
                           pathlib.Path(model_root / "config").subdirs(recursive=True)
            config_ = find_config_in_dirs(config, config_dirs)
            if config_:
                return config_
    
    # Check for config file that comes along with `weights_path`.
    if weights_path:
        weights_path = pathlib.Path(weights_path[0] if isinstance(weights_path, list) else weights_path)
        if weights_path.is_weights_file():
            config_ = (weights_path.parent / "config.py").config_file()
            if config_.is_config_file():
                return config_
    
    # That's it.
    error_console.log(f"Could not find configuration file given: "
                      f"config={config}, project_root={project_root}, "
                      f"model_root={model_root}, weights_path={weights_path}.")
    
    return None


def load_config(config: Any) -> dict:
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
        error_console.log(f"Could not find configuration file at: {config}. Setting an empty dictionary.")
        data = {}

    return data

# endregion


# region Datasets

def list_mon_datasets(task: str, mode: str) -> list[str]:
    """List all available datasets in ``mon`` framework."""
    from mon.globals import Task, Split, DATASETS

    split    = Split("train" if mode == "train" else "test")
    task     = Task(task)
    datasets = DATASETS

    return sorted([
        d for d in datasets
        if task in datasets[d].tasks and split in datasets[d].splits
    ])


def list_extra_datasets(task: str, mode: str) -> list[str]:
    """List all available datasets in ``extra`` framework."""
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
    """List all available datasets."""
    datasets        = sorted(list_mon_datasets(task, mode) + list_extra_datasets(task, mode))
    default_configs = get_project_default_config(project_root)
    if default_configs.get("DATASETS"):
        datasets = [d for d in datasets if d in default_configs["DATASETS"]]
    return datasets


def parse_data_dir(root: str | pathlib.Path, data_dir: str | pathlib.Path) -> str | pathlib.Path:
    """Parse absolute data directory path from given components.
    
    Args:
        root: The root directory.
        data_dir: The data directory.
    """
    from mon.globals import ROOT_DIR
    root     = pathlib.Path(root)
    data_dir = pathlib.Path(data_dir)
    if not data_dir.is_dir():
        if (mon.ROOT_DIR / data_dir).is_dir():
            data_dir = ROOT_DIR / data_dir
        elif (root / data_dir).is_dir():
            data_dir = root / data_dir
    return data_dir

# endregion


# region Device

def is_rank_zero() -> bool:
    """From Pytorch Lightning Official Document on DDP, we know that PL
    intended call the main script multiple times to spin off the child
    processes that take charge of GPUs.

    They used the environment variable "LOCAL_RANK" and "NODE_RANK" to denote
    GPUs. So we can add conditions to bypass the code blocks that we don't want
    to get executed repeatedly.
    """
    return "LOCAL_RANK" not in os.environ and "NODE_RANK" not in os.environ


def list_cuda_devices() -> str | None:
    """List all available cuda devices in the current machine."""
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        cuda_str = "cuda:" + ",".join(str(i) for i in range(num_devices))
        return cuda_str
    return None


def list_devices() -> list[str]:
    """List all available devices in the current machine."""
    # Default devices: CPU and auto (for `pytorch.lighting`)
    devices = ["auto", "cpu"]
    
    # Get GPU devices if available
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        devices.extend(f"cuda:{i}" for i in range(num_devices))
        all_cuda_str = "cuda:" + ",".join(str(i) for i in range(num_devices))
        if all_cuda_str != "cuda:0":
            devices.append(all_cuda_str)
            
    return devices


def set_device(device: Any, use_single_device: bool = True) -> torch.device:
    """Set a cuda device in the current machine.
    
    Args:
        device: Cuda devices to set.
        use_single_device: If `True`, set a single-device cuda device in the list.
    
    Returns:
        A cuda device in the current machine.
    """
    device = parse_device(device)
    if isinstance(device, list) and use_single_device:
        device = device[0]
    # os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")


def get_machine_memory(unit: MemoryUnit = MemoryUnit.GB) -> list[int]:
    """Return the RAM status as a `list` of `[total, used, free]`.
    
    Args:
        unit: The memory unit. Default: `'GB'`.
    """
    memory = psutil.virtual_memory()
    ratio  = MemoryUnit.byte_conversion_mapping()[MemoryUnit.from_value(unit)]
    return [
        memory.total     / ratio,  # total
        memory.used      / ratio,  # free
        memory.available / ratio   # used
    ]


def get_gpu_device_memory(device: int = 0, unit: MemoryUnit = MemoryUnit.GB) -> list[int]:
    """Return the GPU memory status as a `list` of `[total, used, free]`.
    
    Args:
        device: The index of the GPU device. Default: `0`.
        unit: The memory unit. Default: `'GB'`.
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
        return [int(x) for x in device.split(",")] if "," in device else [0] if not device else device
    return device


def get_model_device(model: nn.Module) -> torch.device:
    """Get the device of the given model since nn.Module doesn't directly store
    a .device attribute.
    """
    return next(model.parameters()).device
    
# endregion


# region Menu

def parse_menu_string(items: Sequence | Collection, num_columns: int = 4) -> str:
    s = f"\n  "
    for i, item in enumerate(items):
        s += f"{f'{i}.':>6} {item}\n  "
    s += f"{f'Other.':} (please specify)\n  "
    return s

# endregion


# region Models

def is_extra_model(model: str) -> bool:
    """Check if the given model is an extra model."""
    from mon.globals import MODELS, EXTRA_MODELS, EXTRA_MODEL_STR
    model        = model.replace(f" {EXTRA_MODEL_STR}", "").strip()
    mon_models   = dtype.flatten_models_dict(MODELS)
    extra_models = dtype.flatten_models_dict(EXTRA_MODELS)
    return (
        f"{EXTRA_MODEL_STR}" in model
        or (model not in mon_models and model in extra_models)
    )


def list_mon_models(task: str = None, mode: str = None, arch: str = None) -> list[str]:
    from mon.globals import Task, MODELS, LType
    flatten_models = dtype.flatten_models_dict(MODELS)
    models         = list(flatten_models.keys())
    
    # Filter task
    if task in Task.values():
        task   = Task(task)
        models = [m for m in models if task in flatten_models[m].tasks]
   
    # Filter mode
    if mode == "train":
        models = [m for m in models if any(lt in LType.trainable() for lt in flatten_models[m].ltypes)]
    
    # Filter arch
    if arch:
        models = [m for m in models if arch in flatten_models[m].arch]
        
    # Sort
    return sorted(models)


def list_extra_models(task: str = None, mode: str = None, arch: str = None) -> list[str]:
    from mon.globals import Task, EXTRA_MODELS, LType
    flatten_models = dtype.flatten_models_dict(EXTRA_MODELS)
    models         = list(flatten_models.keys())
   
    # Filter task
    if task in Task.values():
        task   = Task(task)
        models = [m for m in models if task in flatten_models[m]["tasks"]]
   
    # Filter mode
    if mode == "train":
        models = [m for m in models if any(lt in LType.trainable() for lt in flatten_models[m]["ltypes"])]
    
    # Filter arch
    if arch:
        models = [m for m in models if arch in flatten_models[m]["arch"]]
   
    # Sort
    return sorted(models)


def list_models(
    task        : str = None,
    mode        : str = None,
    arch        : str = None,
    project_root: str | pathlib.Path = None
) -> list[str]:
    from mon.globals import EXTRA_MODEL_STR
    models       =   list_mon_models(task, mode, arch)
    extra_models = list_extra_models(task, mode, arch)
    
    # Filter models based on project's configuration
    default_configs = get_project_default_config(project_root=project_root)
    if default_configs.get("MODELS"):
        project_models = [humps.snakecase(m) for m in default_configs["MODELS"]]
        models         = [m for m in models       if humps.snakecase(m) in project_models]
        extra_models   = [m for m in extra_models if humps.snakecase(m) in project_models]
        
    # Rename extra models for clarity
    for i, m in enumerate(extra_models):
        if m in models:
            extra_models[i] = f"{m} {EXTRA_MODEL_STR}"
            
    # Sort
    return sorted(models + extra_models)


def list_mon_archs(task: str = None, mode: str = None) -> list[str]:
    from mon.globals import Task, MODELS, LType
    flatten_models = dtype.flatten_models_dict(MODELS)
    models         = list(flatten_models.keys())
    
    # Filter task
    if task in Task.values():
        task   = Task(task)
        models = [m for m in models if task in flatten_models[m].tasks]
    
    # Filter mode
    if mode == "train":
        models = [m for m in models if any(lt in LType.trainable() for lt in flatten_models[m].ltypes)]
    
    # Get archs
    archs = [flatten_models[m].arch.strip() for m in models if flatten_models[m].arch not in [None, "None", ""]]
    
    # Sort
    return sorted(dtype.unique(archs))


def list_extra_archs(task: str = None, mode: str = None) -> list[str]:
    from mon.globals import Task, EXTRA_MODELS, LType
    flatten_models = dtype.flatten_models_dict(EXTRA_MODELS)
    models         = list(flatten_models.keys())
   
    # Filter task
    if task in Task.values():
        task   = Task(task)
        models = [m for m in models if task in flatten_models[m]["tasks"]]
    
    # Filter mode
    if mode == "train":
        models = [m for m in models if any(lt in LType.trainable() for lt in flatten_models[m]["ltypes"])]
        
    # Get archs
    archs = [flatten_models[m]["arch"].strip() for m in models if flatten_models[m]["arch"] not in [None, "None", ""]]
    
    # Sort and return
    return sorted(dtype.unique(archs))


def list_archs(
    task        : str = None,
    mode        : str = None,
    project_root: str | pathlib.Path = None
) -> list[str]:
    from mon.globals import MODELS, EXTRA_MODELS
    models       =   list_mon_models(task, mode)
    extra_models = list_extra_models(task, mode)
    
    # Filter models based on project's configuration
    default_configs = get_project_default_config(project_root)
    if default_configs.get("MODELS"):
        project_models = [humps.snakecase(m) for m in default_configs["MODELS"]]
        models         = [m for m in models       if humps.snakecase(m) in project_models]
        extra_models   = [m for m in extra_models if humps.snakecase(m) in project_models]
    
    # Get archs
    flatten_mon_models   = dtype.flatten_models_dict(MODELS)
    flatten_extra_models = dtype.flatten_models_dict(EXTRA_MODELS)
    archs = ([flatten_mon_models[m].arch      for m in models] +
             [flatten_extra_models[m]["arch"] for m in extra_models])
    archs = [a.strip() for a in archs if a not in [None, "None", ""]]
    
    # Sort
    return sorted(dtype.unique(archs))


def parse_model_dir(arch: str, model: str) -> pathlib.Path | None:
    """Parse model's directory from given components."""
    from mon.globals import EXTRA_MODELS, MODELS
    model_name = parse_model_name(model)
    model_dir  = EXTRA_MODELS[arch][model_name].get("model_dir") \
        if is_extra_model(model) \
        else MODELS[arch][model_name].model_dir
    return pathlib.Path(model_dir) if model_dir else None


def parse_model_name(model: str) -> str:
    """Parse model's name from given components."""
    from mon.globals import EXTRA_MODEL_STR
    return model.replace(f" {EXTRA_MODEL_STR}", "").strip()


def parse_model_fullname(name: str, data: str, suffix: str = None) -> str:
    """Parse model's fullname from given components as ``name-data-suffix``.
    
    Args:
        name: The model's name.
        data: The dataset's name.
        suffix: The suffix of the model's name.
    """
    if not name:
        rich.error_console.log("Model's `name` must be given.")
    
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
    try:
        importlib.import_module(package_name)
        if verbose:
            print(f"`{package_name}` is installed.")
        return True
    except ImportError:
        if verbose:
            print(f"`{package_name}` is not installed.")
        return False

# endregion


# region Save Dir

def list_train_save_dirs(root: str | pathlib.Path) -> list[pathlib.Path]:
    """List all training save directories in the given project"""
    root = pathlib.Path(root)
    return sorted((root / "run" / "train").dirs())


def parse_save_dir(
    root : str | pathlib.Path,
    arch : str = None,
    model: str = None,
    data : str = None,
) -> str | pathlib.Path:
    """Parse ``save_dir`` in the following format:
        root
         |_ arch
             |_ model/fullname
                 |_ data
    
    Args:
        root: The project root.
        arch: The model's architecture.
        model: The model's name.
        data: The dataset's name.
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

def set_random_seed(seed: int | list[int] | tuple[int, int]):
    """Set random seeds."""
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
    from mon.globals import Task
    tasks = Task.keys()
    
    # Filter tasks based on project's configuration
    default_configs = get_project_default_config(project_root)
    if default_configs.get("TASKS"):
        tasks = [t for t in tasks if t in default_configs["TASKS"]]
        
    # Sort
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

def download_weights_from_url(url: str, path: pathlib.Path, overwrite: bool = False) -> pathlib.Path:
    """Download weights from the given `url` to the given `path`."""
    if not pathlib.is_url(url):
        raise ValueError("Both `url` and `path` must be given.")
    
    path = pathlib.Path(path)
    if not path.exists() or overwrite:
        pathlib.delete_files(path=path.parent, regex=path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, path, None, progress=True)
    return path


def list_weights_files(model: str, project_root: str | pathlib.Path = None) -> list[pathlib.Path]:
    from mon.globals import ZOO_DIR
    
    def collect_weights_files(root: pathlib.Path) -> list[pathlib.Path]:
        return sorted(f for f in root.rglob("*") if f.is_weights_file())
    
    # Collect weights files from project's `run/train` directory
    weights_files = []
    if project_root not in [None, "None", ""]:
        weights_files += collect_weights_files(pathlib.Path(project_root) / "run" / "train")
    
    # Collect weights files from `zoo` directory
    weights_files += collect_weights_files(ZOO_DIR)
    
    # Filter weights files based on model's name
    model_name    = parse_model_name(model)
    return sorted(dtype.unique([f for f in weights_files if model_name in str(f)]))


def parse_weights_file(
    root   : str | pathlib.Path,
    weights: str | pathlib.Path | Sequence[str | pathlib.Path]
) -> str | pathlib.Path | Sequence[str | pathlib.Path]:
    """Parse weights file. If the weights file is a relative path in the `zoo`
    directory, then it will be converted to the absolute path. If the weights
    file is a list with a single weights files, then it will be converted to a
    single weights.
    
    Args:
        root: The root directory.
        weights: The weights file to parse.
    """
    from mon.globals import ROOT_DIR
    root    = pathlib.Path(root)
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
