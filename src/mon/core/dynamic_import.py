#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility module for dynamic submodule imports."""

__all__ = [
    "import_all_submodules",
]

import importlib
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
