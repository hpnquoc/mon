#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Defines scalar constants."""

__all__ = [
    "ACCELERATORS",
    "CALLBACKS",
    "DATAMODULES",
    "DATASETS",
    "DATA_DIR",
    "Enum",
    "LOGGERS",
    "LOSSES",
    "LR_SCHEDULERS",
    "METRICS",
    "MODELS",
    "MON_DIR",
    "MON_EXTRA_DIR",
    "OPTIMIZERS",
    "ROOT_DIR",
    "SAVE_CKPT_EXT",
    "SAVE_DEBUG_DIR",
    "SAVE_IMAGE_DIR",
    "SAVE_IMAGE_EXT",
    "SAVE_LABEL_DIR",
    "SAVE_VISUALIZE_DIR",
    "SAVE_WEIGHTS_EXT",
    "SERIALIZERS",
    "STRATEGIES",
    "TRANSFORMS",
    "ZOO_DIR",
]

from mon.constants.enums import *
from mon.core import factory, pathlib


# ----- Directory -----
current_file  = pathlib.Path(__file__).absolute()
ROOT_DIR      = current_file.parents[3]     # ./mon
DATA_DIR      = ROOT_DIR / "data"           # ./mon/data
SRC_DIR       = ROOT_DIR / "src"            # ./mon/src
MON_DIR       = ROOT_DIR / "src/mon"        # ./mon/src/mon
MON_EXTRA_DIR = ROOT_DIR / "src/mon/extra"  # ./mon/src/mon/extra
ZOO_DIR       = ROOT_DIR / "zoo"            # ./mon/zoo


# ----- Constants -----
SAVE_DEBUG_DIR     = "debug"
SAVE_IMAGE_DIR     = "pred"
SAVE_LABEL_DIR     = "label"
SAVE_VISUALIZE_DIR = "visualize"
SAVE_CKPT_EXT      = WeightExtension.CKPT.value
SAVE_IMAGE_EXT     = ImageExtension.JPG.value
SAVE_WEIGHTS_EXT   = WeightExtension.PT.value
# List 3rd party modules
EXTRA_MODELS   = {  # architecture/model (+ variant)
    "esdnet_snn": {
        "esdnet_snn": {
            "tasks"    : [Task.DERAIN],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "derain" / "esdnet_snn",
        },
    },
    "srgb_tir"  : {
        "srgb_tir": {
            "tasks"    : [Task.RGB2TIR],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "types" / "thermal" / "srgb_tir",
        },
    },
}


# ----- Factory -----
ACCELERATORS  = factory.Factory(name="Accelerators")
CALLBACKS     = factory.Factory(name="Callbacks")
DATAMODULES   = factory.Factory(name="DataModules")
DATASETS      = factory.Factory(name="Datasets")
LOGGERS       = factory.Factory(name="Loggers")
LOSSES        = factory.Factory(name="Losses")
LR_SCHEDULERS = factory.Factory(name="LRSchedulers")
METRICS       = factory.Factory(name="Metrics")
MODELS        = factory.ModelFactory(name="Models")
OPTIMIZERS    = factory.Factory(name="Optimizers")
SERIALIZERS   = factory.Factory(name="Serializers")
STRATEGIES    = factory.Factory(name="Strategies")
TRANSFORMS    = factory.Factory(name="Transforms")
