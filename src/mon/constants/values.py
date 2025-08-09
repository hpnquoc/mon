#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Defines scalar constants."""

__all__ = [
    "ACCELERATORS",
    "CALLBACKS",
    "DATAMODULES",
    "DATASETS",
    "DATA_DIR",
    "EXTRA_DATASETS",
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

'''
ZOO_DIR = None
for i, parent in enumerate(current_file.parents):
    if (parent / "zoo").is_dir():
        ZOO_DIR = parent / "zoo"
        break
    if i >= 5:
        break
if ZOO_DIR is None:
    raise Warning(f"Cannot locate the ``zoo`` directory.")

DATA_DIR = os.getenv("DATA_DIR", None)
DATA_DIR = pathlib.Path(DATA_DIR) if DATA_DIR else None
DATA_DIR = DATA_DIR or pathlib.Path("/data")
DATA_DIR = DATA_DIR if DATA_DIR.is_dir() else ROOT_DIR / "data"
if not DATA_DIR.is_dir():
    raise Warning(f"Cannot locate the ``data`` directory.")
'''


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
EXTRA_DATASETS = {}
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
    "yolov8" : {
        "yolov8n": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov8s": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov8m": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov8l": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov8x": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
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
    "yolov11": {
        "yolov11n"    : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov11s"    : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov11m"    : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov11l"    : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov11x"    : {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov11n_obb": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov11s_obb": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov11m_obb": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov11l_obb": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov11x_obb": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov11n_seg": {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov11s_seg": {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov11m_seg": {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov11l_seg": {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
        },
        "yolov11x_seg": {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "extra" / "ultralytics",
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
    # region segment
    "sam" : {
        "sam_vit_b": {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam",
        },
        "sam_vit_h": {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam",
        },
        "sam_vit_l": {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam",
        },
    },
    "sam2": {
        "sam2_hiera_b+": {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam2",
        },
        "sam2_hiera_l" : {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam2",
        },
        "sam2_hiera_s" : {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam2",
        },
        "sam2_hiera_t" : {
            "tasks"    : [Task.SEGMENT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam2",
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
