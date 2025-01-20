#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Spiking Neural Network (SNN).

This module implements Spiking Neural Networks (SNN), its variants and networks.
It also provides a wrapper around the `snntorch` and `spikingjelly` libraries.
"""

from __future__ import annotations

import sys

from mon import core

console       = core.console
error_console = core.error_console

try:
	import snntorch
	import spikingjelly
	from snntorch import *
	from spikingjelly import *
except ImportError as e:
	error_console.log(f"Missing library: {e.name}. Skipping execution.")
	sys.exit(0)  # Exit without error
