#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core Python ops: data types, file I/O, logging, etc.

This package implements the basic functionalities of Python operations. This is achieved
by extending `Python <https://www.python.org/>`__ built-in functions, including:
	- Data types and structures.
	- File I/O.
	- Filesystem handling.
	- Logging.
	- Managing devices.
	- Parsing.
	- Path handling.
	- etc.

Design Principle:
	- All submodules must be ATOMIC and self-contained.
	- Each submodule should extend a module and keep the same name.
"""

import mon.core.humps
import mon.core.logging
import mon.core.pathlib
import mon.core.rich
import mon.core.serializers
import mon.core.thop
from mon.core.factory import *
from mon.core.humps import *
from mon.core.logging import *
from mon.core.parse_args import *
from mon.core.pathlib import *
from mon.core.rich import (
	console, create_download_bar, create_progress_bar, error_console, get_terminal_size,
	print_dict, print_table, set_terminal_size,
)
from mon.core.serializers import *
from mon.core.type_extensions import *
from mon.core.types import *
from mon.core.utils import *
