#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Parses config arguments and command line arguments."""

__all__ = [
    "list_configs",
    "parse_cli_args",
    "parse_config_file",
    "parse_predict_args",
    "parse_train_args",
    "read_config",
]

import argparse
import importlib.util
import socket
from typing import Any

from mon.core import pathlib, rich, serializers, type_extensions
from mon.core.device import parse_device


# ----- Retrieve -----
def list_configs(
    project_root : str | pathlib.Path,
    model_root   : str | pathlib.Path = None,
    model        : str  = None,
    absolute_path: bool = False,
) -> list[pathlib.Path]:
    """Lists configuration files in the project and/or model directory.

    Args:
        project_root: Root directory of the project.
        model_root: Root directory of the model. Default is ``None``.
        model: Name of the model to filter configs. Default is ``None``.
        absolute_path: If ``True``, returns absolute paths else file names.
            Default is ``False``.

    Returns:
        Sorted list of config file ``Path`` objects.
    """
    from mon import nn
    
    def is_valid(x) -> bool:
        return x not in [None, "", "None"]

    def collect_config_files(root: str | pathlib.Path) -> list[pathlib.Path]:
        config_dir = pathlib.Path(root) / "config"
        return list(config_dir.files(recursive=True))
    
    # List config files in project and model directories
    config_files = []
    if is_valid(project_root):
        config_files += collect_config_files(project_root)
    if is_valid(model_root):
        config_files += collect_config_files(model_root)
    
    # Filter
    config_files = [
        cf for cf in config_files
        if cf.is_config_file() or (cf.is_py_file() and cf.name != "__init__.py")
    ]
    
    if is_valid(model):
        model_name   = nn.parse_model_name(model)
        config_files = [cf for cf in config_files if model_name in cf.name]
    
    if not absolute_path:
        config_files = [cf.name for cf in config_files]
      
    return sorted(type_extensions.unique(config_files))


def read_config(config: Any) -> dict:
    """Loads configuration from a given source.

    Args:
        config: Config source (dict, file path, or string).

    Returns:
        Dict with loaded config, or empty dict if loading fails.
    """
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
            data = serializers.read_from_file(path=config)
    else:
        data = None
    
    if data:
        rich.console.log(f"Loaded configuration from: {config}.")
    else:
        rich.error_console.log(f"Could not load configuration from: {config}. "
                               f"Returning empty dict.")
        data = {}

    return data


# ----- Parse Config File -----
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
    
    rich.error_console.log(
        f"Could not find configuration file given: "
        f"config={config}, project_root={project_root}, "
        f"model_root={model_root}, weights_path={weights_path}"
    )
    return None


# ----- Parse Args -----
def _str_or_none(value: Any) -> str | None:
    """Converts a value to a string or ``None`` if value is ``"None"``.

    Args:
        value: Value to convert.

    Returns:
        String of ``value`` or ``None`` if ``value`` is ``"None"``.
    """
    return None if value == "None" else str(value)


def _int_or_none(value: Any) -> int | None:
    """Converts a value to an integer or ``None`` if value is ``"None"``.

    Args:
        value: Value to convert.

    Returns:
        Integer of ``value`` or ``None`` if ``value`` is ``"None"``.
    """
    return None if value == "None" else int(value)


def _float_or_none(value: Any) -> float | None:
    """Converts a value to a float or ``None`` if value is ``"None"``.

    Args:
        value: Value to convert.

    Returns:
        Float of ``value`` or ``None`` if ``value`` is ``"None"``.
    """
    return None if value == "None" else float(value)


def parse_cli_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="predict")
    # Basic
    parser.add_argument("--config",       type=_str_or_none, default=None, help="Model config.")
    parser.add_argument("--root",         type=_str_or_none, default=None, help="Root directory of the current run.")
    parser.add_argument("--arch",         type=_str_or_none, default=None, help="Model architecture or family.")
    parser.add_argument("--model",        type=_str_or_none, default=None, help="Model name.")
    parser.add_argument("--data",         type=_str_or_none, default=None, help="Dataset name or directory.")
    parser.add_argument("--fullname",     type=_str_or_none, default=None, help="Full name of the current run.")
    parser.add_argument("--save-dir",     type=_str_or_none, default=None, help="Saving directory. If not set, it will be `root/arch/model/data`.")
    parser.add_argument("--weights",      type=_str_or_none, default=None, help="Weights paths.")
    parser.add_argument("--device",       type=_str_or_none, default=None, help="Running devices.")
    parser.add_argument("--imgsz",        type=_int_or_none, default=None, help="Image sizes.")
    parser.add_argument("--resize",       action="store_true",             help="Resize the input image to `imgsz`.")
    parser.add_argument("--epochs",       type=_int_or_none, default=None, help="Training epochs.")
    parser.add_argument("--steps",        type=_int_or_none, default=None, help="Training steps.")
    parser.add_argument("--benchmark",    action="store_true",             help="Benchmark the model.")
    parser.add_argument("--save-image",   action="store_true",             help="Save the output image.")
    parser.add_argument("--save-debug",   action="store_true",             help="Save the debug information.")
    parser.add_argument("--use-fullname", action="store_true",             help="Use the full name for the save_dir.")
    parser.add_argument("--keep-subdirs", action="store_true",             help="Keep subdirectories in the save_dir.")
    parser.add_argument("--exist-ok",     action="store_true",             help="If ``False``, it will delete the save directory if it already exists.")
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("extra_args",     nargs=argparse.REMAINDER,        help="Additional arguments")
    args = parser.parse_args()
    return args


def parse_train_args(model_root: str | pathlib.Path = None) -> dict | argparse.Namespace:
    """Parse arguments for training."""
    from mon import nn
    
    hostname = socket.gethostname().lower()
    
    # Get input args
    cli_args = vars(parse_cli_args())
    config   = cli_args.get("config")
    root     = cli_args.get("root")
    root     = pathlib.Path(root) if root else None
    weights  = cli_args.get("weights")
    
    # Get config args
    config = parse_config_file(
        project_root = root,
        model_root   = model_root,
        weights_path = weights,
        config       = config,
    )
    args   = read_config(config)
    
    # Prioritize cli_args -> args
    root         = root                         or args.get("root")
    arch         = cli_args.get("arch")         or args.get("arch")
    model        = cli_args.get("model")        or args.get("model")
    data         = cli_args.get("data")         or args.get("data")
    fullname     = cli_args.get("fullname")     or args.get("fullname")
    save_dir     = cli_args.get("save_dir")     or args.get("save_dir")
    weights      = cli_args.get("weights")      or args.get("weights")
    device       = cli_args.get("device")       or args.get("device")
    imgsz        = cli_args.get("imgsz")        or args.get("imgsz")
    resize       = cli_args.get("resize")       or args.get("resize")
    epochs       = cli_args.get("epochs")       or args.get("epochs")
    steps        = cli_args.get("steps")        or args.get("steps")
    benchmark    = cli_args.get("benchmark")    or args.get("benchmark")
    save_image   = cli_args.get("save_image")   or args.get("save_image")
    save_debug   = cli_args.get("save_debug")   or args.get("save_debug")
    use_fullname =                                 args.get("use_fullname", False)
    keep_subdirs = cli_args.get("keep_subdirs") or args.get("keep_subdirs")
    exist_ok     = cli_args.get("exist_ok")     or args.get("exist_ok")
    verbose      = cli_args.get("verbose")      or args.get("verbose")
    extra_args   = cli_args.get("extra_args")
    
    # Parse arguments
    if save_dir in [None, ""]:
        save_dir = pathlib.parse_save_dir(root/"run"/"train", arch, model, data)
    else:
        save_dir = pathlib.Path(save_dir)
        if str("run/train") not in str(save_dir):
            save_dir = pathlib.Path(f"run/train/{save_dir}")
        if str(root) not in str(save_dir):
            save_dir = root / save_dir
            
    weights = nn.parse_weights_file(root/"run"/"train", weights)
    device  = parse_device(device)
    
    # Update arguments
    args["hostname"]     = hostname
    args["root"]         = root
    args["arch"]         = arch
    args["model"]        = model
    args["data"]         = data
    args["fullname"]     = fullname
    args["save_dir"]     = save_dir
    args["weights"]      = weights
    args["device"]       = device
    args["imgsz"]        = imgsz
    args["resize"]       = resize
    args["epochs"]       = epochs
    args["steps"]        = steps
    args["benchmark"]    = benchmark
    args["save_image"]   = save_image
    args["save_debug"]   = save_debug
    args["use_fullname"] = use_fullname
    args["keep_subdirs"] = keep_subdirs
    args["exist_ok"]     = exist_ok
    args["verbose"]      = verbose
    args |= extra_args
    
    # Save config file
    if not exist_ok:
        pathlib.delete_dir(paths=save_dir)
        
    save_dir.mkdir(parents=True, exist_ok=True)
    if config and config.is_config_file():
        # pathlib.copy_file(src=config, dst=save_dir / f"config{config.suffix}")
        pathlib.copy_file(src=config, dst=save_dir / f"{config.name}")
    
    # Return
    # args = argparse.Namespace(**args)
    return args


def parse_predict_args(model_root: str | pathlib.Path = None) -> dict | argparse.Namespace:
    """Parse arguments for predicting."""
    from mon import vision, nn
    
    hostname = socket.gethostname().lower()
    
    # Get input args
    cli_args = vars(parse_cli_args())
    config   = cli_args.get("config")
    root     = cli_args.get("root")
    root     = pathlib.Path(root) if root else None
    weights  = cli_args.get("weights")
    
    # Get config args
    config = parse_config_file(
        project_root = root,
        model_root   = model_root,
        weights_path = weights,
        config       = config,
    )
    args   = read_config(config)
    
    # Prioritize cli_args -> args
    root         = root                         or args.get("root")
    arch         = cli_args.get("arch")         or args.get("arch")
    model        = cli_args.get("model")        or args.get("model")
    data         = cli_args.get("data")         or args.get("data")
    fullname     = cli_args.get("fullname")     or args.get("fullname")
    save_dir     = cli_args.get("save_dir")     or args.get("save_dir")
    weights      = cli_args.get("weights")      or args.get("weights")
    device       = cli_args.get("device")       or args.get("device")
    imgsz        = cli_args.get("imgsz")        or args.get("imgsz")
    resize       = cli_args.get("resize")       or args.get("resize")
    epochs       = cli_args.get("epochs")       or args.get("epochs")
    steps        = cli_args.get("steps")        or args.get("steps")
    benchmark    = cli_args.get("benchmark")    or args.get("benchmark")
    save_image   = cli_args.get("save_image")   or args.get("save_image")
    save_debug   = cli_args.get("save_debug")   or args.get("save_debug")
    use_fullname =                                 args.get("use_fullname", False)
    keep_subdirs = cli_args.get("keep_subdirs") or args.get("keep_subdirs")
    exist_ok     = cli_args.get("exist_ok")     or args.get("exist_ok")
    verbose      = cli_args.get("verbose")      or args.get("verbose")
    extra_args   = cli_args.get("extra_args")
    
    # Parse arguments
    if save_dir in [None, ""]:
        if use_fullname:
            save_dir = pathlib.parse_save_dir(root/"run"/"predict", arch, fullname, None)
        else:
            save_dir = pathlib.parse_save_dir(root/"run"/"predict", arch, model,    None)
    else:
        save_dir = pathlib.Path(save_dir)
        save_dir = save_dir.replace("run/train/", "")
        if str("run/predict") not in str(save_dir):
            save_dir = pathlib.Path(f"run/predict/{save_dir}")
        if str(root) not in str(save_dir):
            save_dir = root / save_dir
        
    weights = nn.parse_weights_file(root, weights)
    device  = parse_device(device)
    imgsz   = vision.image_size(imgsz)
    
    # Update arguments
    args["hostname"]     = hostname
    args["root"]         = root
    args["arch"]         = arch
    args["model"]        = model
    args["data"]         = data
    args["fullname"]     = fullname
    args["save_dir"]     = save_dir
    args["weights"]      = weights
    args["device"]       = device
    args["imgsz"]        = imgsz
    args["resize"]       = resize
    args["epochs"]       = epochs
    args["steps"]        = steps
    args["benchmark"]    = benchmark
    args["save_image"]   = save_image
    args["save_debug"]   = save_debug
    args["use_fullname"] = use_fullname
    args["keep_subdirs"] = keep_subdirs
    args["exist_ok"]     = exist_ok
    args["verbose"]      = verbose
    args |= extra_args
    
    # Save config file
    if not exist_ok:
        pathlib.delete_dir(paths=save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    if config and config.is_config_file():
        # pathlib.copy_file(src=config, dst=save_dir / f"config{config.suffix}")
        pathlib.copy_file(src=config, dst=save_dir / f"{config.name}")
    
    # Return
    # args = argparse.Namespace(**args)
    return args
