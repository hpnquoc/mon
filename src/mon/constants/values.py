#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Defines scalar constants."""

__all__ = [
    "ACCELERATORS",
    "CALLBACKS",
    "DATAMODULES",
    "DATASETS",
    "DATA_DIR",
    "EXTRA_MODELS",
    "EXTRA_STR",
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
EXTRA_STR      = "[extra]"
EXTRA_MODELS   = {  # architecture/model (+ variant)
    # region detect
    "deim"   : {
        "deim_dfine_l": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_dfine_m": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_dfine_n": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_dfine_s": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_dfine_x": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_rtdetrv2_r18vd": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_rtdetrv2_r34vd": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_rtdetrv2_r50vd": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_rtdetrv2_r50vd_m": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
        "deim_rtdetrv2_r101vd": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "deim",
        },
    },
    "dfine"  : {
        "dfine_l": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "dfine",
        },
        "dfine_m": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "dfine",
        },
        "dfine_n": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "dfine",
        },
        "dfine_s": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "dfine",
        },
        "dfine_x": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "dfine",
        },
    },
    "yolor"  : {
        "yolor_d6": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolor",
        },
        "yolor_e6": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolor",
        },
        "yolor_p6": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolor",
        },
        "yolor_w6": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolor",
        },
    },
    "yolov7" : {
        "yolov7"    : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
        "yolov7_d6" : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
        "yolov7_e6" : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
        "yolov7_e6e": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
        "yolov7_w6" : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
        "yolov7x"   : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
    },
    "yolov9" : {
        "gelan_c" : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov9",
        },
        "gelan_e" : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov9",
        },
        "yolov9_c": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov9",
        },
        "yolov9_e": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov9",
        },
    },
    # endregion
    # region enhance/derain
    "esdnet_snn": {
        "esdnet_snn": {
            "tasks"    : [Task.DERAIN],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "derain" / "esdnet_snn",
        },
    },
    # endregion
    # region types/thermal
    "srgb_tir": {
        "srgb_tir": {
            "tasks"    : [Task.RGB2TIR],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "types" / "thermal" / "srgb_tir",
        },
    },
    # endregion
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
