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

from . import humps
from . import logging
from . import pathlib
from . import rich
from . import serializers
from . import thop
from .config import *
from .device import *
from .dynamic_import import *
from .factory import *
from .humps import *
from .logging import *
from .pathlib import *
from .rich import (
    console, create_download_bar, create_progress_bar, error_console, print_dict,
    print_table,
)
from .serializers import *
from .system import *
from .timer import *
from .type_extensions import *
from .types import *
