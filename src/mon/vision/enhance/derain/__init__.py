#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements denoising algorithms."""

import os
from mon.core.dynamic_import import import_all_submodules

# Call the reusable function to import all submodules
import_all_submodules(__name__, os.path.dirname(__file__))
