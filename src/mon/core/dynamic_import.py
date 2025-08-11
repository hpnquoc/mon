#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility module for dynamic submodule imports."""

__all__ = [
    "import_all_submodules",
    "import_parent_module",
]

import importlib
import os
import pkgutil


def import_all_submodules(package_name, package_dir):
    """Dynamically import all public names from submodules in the given package.

    Args:
        package_name (str): The full package name (e.g., 'mon.vision.enhance.lle').
        package_dir (str): The directory of the package's __init__.py.

    Returns:
        None: Imports names directly into the caller's global namespace.
    """
    for _, module_name, _ in pkgutil.iter_modules([package_dir]):
        if not module_name.startswith("_"):  # Skip private modules
            try:
                # Import the submodule relative to the package
                module = importlib.import_module(f".{module_name}", package=package_name)
                # Import all public names into the caller's namespace
                for name in dir(module):
                    # print(name)
                    if not name.startswith("_"):  # Skip private names
                        if name not in globals():  # Avoid name conflicts
                            globals()[name] = getattr(module, name)
            except ImportError as e:
                print(f"Warning: Failed to import {module_name}: {e}")
                continue


def import_parent_module(module_name: str, levels_up: int = 1) -> object:
    """Import a module from a parent directory relative to the current file using importlib.

    Args:
        module_name (str): Name of the module to import (without .py extension).
        levels_up (int, optional): Number of parent directory levels to traverse. Defaults to 1.

    Returns:
        object: The imported module.

    Raises:
        ValueError: If module_name is empty or levels_up is less than 1.
        FileNotFoundError: If the module file or parent directory does not exist.
        ImportError: If the module cannot be loaded or spec cannot be created.
    """
    if not module_name:
        raise ValueError("Module name cannot be empty")
    if levels_up < 1:
        raise ValueError("levels_up must be at least 1")

    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Traverse up the specified number of parent directories
    parent_dir = current_dir
    for _ in range(levels_up):
        parent_dir = os.path.dirname(parent_dir)

    # Construct the full path to the module
    module_path = os.path.join(parent_dir, f"{module_name}.py")

    # Check if the module file exists
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Module file {module_path} does not exist")

    # Get the module specification
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Cannot create spec for module {module_name}")

    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules to prevent re-import issues
    sys.modules[module_name] = module

    # Execute the module
    spec.loader.exec_module(module)

    return module
