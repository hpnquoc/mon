#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Provides utility functions and data structures."""

__all__ = [
    "parse_menu_string",
]

from typing import Collection, Sequence


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
