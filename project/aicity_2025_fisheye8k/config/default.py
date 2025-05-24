#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Project-level configurations."""

__all__ = [
    "DATASETS",
    "MODELS",
    "TASKS",
]

from mon.constants import Task

# List all tasks that are performed in this project.
TASKS = [
    Task.DETECT,
]

# List all models that are used in this project.
MODELS = [

]
# If unsure, run the following script:
# mon.print_table(mon.MODELS | mon.EXTRA_MODELS)

# List all datasets that are used in this project.
DATASETS = [
    "aicity_2025_fisheye8k",
]
# If unsure, run the following script:
# mon.print_table(mon.DATASETS | mon.DATASETS_EXTRA)
