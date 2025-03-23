#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Humps Module.

This module extends ``humps``.
"""

from __future__ import annotations

__all__ = [
	"camelize",
	"decamelize",
	"dekebabize",
	"depascalize",
	"is_camelcase",
	"is_kebabcase",
	"is_pascalcase",
	"is_snakecase",
	"kebabize",
	"pascalize",
	"snakecase",
]

from humps import *


def snakecase(x: str) -> str:
	"""Convert a string to snake_case by replacing spaces and hyphens with
	underscores.

	Args:
	    x: The input string to be converted.

	Returns:
	    The converted snake_case string.
	"""
	x = x.replace(" ", "_").replace("-", "_")
	return x
