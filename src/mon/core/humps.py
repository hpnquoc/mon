#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends ``humps`` module."""

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
	"""Converts a string to snake_case by replacing spaces and hyphens with underscores.

	Args:
		x: Input string to convert.

	Returns:
		Converted string in ``snake_case``.
	"""
	return x.replace(" ", "_").replace("-", "_")
