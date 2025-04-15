#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Parses config arguments and command line arguments."""

__all__ = [
    "list_configs",
    "load_config",
    "load_project_defaults",
    "parse_config_file",
]

import importlib.util
from typing import Any

from mon.core import pathlib, rich, serializers, type_extensions


# ----- Retrieve -----
def load_project_defaults(project_root: str | pathlib.Path) -> dict:
    """Gets the default configuration of the project.

    Args:
        project_root: Root directory of the project.

    Returns:
        Dict with default config, or empty dict if invalid or not found.
    """
    if project_root in [None, "None", ""]:
        rich.error_console.log(f"[project_root] is not a valid project directory: {project_root}.")
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


def load_config(config: Any, verbose: bool = True) -> dict:
    """Loads configuration from a given source.

    Args:
        config: Config source (dict, file path, or string).
        verbose: If ``True``, prints verbose messages when loading.

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
            data = serializers.load_from_file(path=config)
    else:
        data = None
    
    if verbose:
        if data:
            rich.console.log(f"Loaded configuration from: {config}.")
        else:
            rich.error_console.log(f"Could not load configuration from: {config}. Returning empty dict.")

    return data or {}


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
