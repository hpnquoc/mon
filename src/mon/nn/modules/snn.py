#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements SNNs and wraps ``snntorch`` and ``spikingjelly`` libraries."""

import sys

from mon import core

try:
	import snntorch
	import spikingjelly
	from snntorch import *
	from spikingjelly import *
except ImportError as e:
	core.error_console.log(f"Missing library: {e.name}. Skipping execution.")
	sys.exit(0)  # Exit without error
