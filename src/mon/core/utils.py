#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Provides utility functions and data structures."""

__all__ = [
    "get_project_defaults",
    "parse_menu_string",
]

import importlib
import importlib.util
from typing import Collection, Sequence

from mon.core import pathlib, rich


# ----- Retrieve -----
def get_project_defaults(project_root: str | pathlib.Path) -> dict:
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


# ----- Convert -----
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
