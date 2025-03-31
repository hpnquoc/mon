#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Defines global constants for ``mon`` package.

Notes:
    * To avoid circular dependency, only define constants of basic/atomic types.
    * The same goes for type aliases.
    * The only exception is the enum and factory constants.
"""

from __future__ import annotations

__all__ = [
    "ACCELERATORS",
    "CALLBACKS",
    "CONFIG_FILE_FORMATS",
    "DATAMODULES",
    "DATASETS",
    "DATA_DIR",
    "DEPTH_DATA_SOURCES",
    "DETECTORS",
    "DISTANCES",
    "EMBEDDERS",
    "EXTRA_DATASETS",
    "EXTRA_DATASET_STR",
    "EXTRA_MODELS",
    "EXTRA_MODEL_STR",
    "FILE_HANDLERS",
    "IMAGE_FILE_FORMATS",
    "LOGGERS",
    "LOSSES",
    "LR_SCHEDULERS",
    "METRICS",
    "MODELS",
    "MON_DIR",
    "MON_EXTRA_DIR",
    "MOTIONS",
    "OBJECTS",
    "OPTIMIZERS",
    "ROOT_DIR",
    "STRATEGIES",
    "TORCH_FILE_FORMATS",
    "TRACKERS",
    "TRANSFORMS",
    "VIDEO_FILE_FORMATS",
    "WEIGHTS_FILE_FORMATS",
    "ZOO_DIR",
]

from mon.core import factory, pathlib, enum


# region Directory

current_file = pathlib.Path(__file__).absolute()
ROOT_DIR      = current_file.parents[2]     # ./mon
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

# endregion


# region Factory

ACCELERATORS  = factory.Factory(name="Accelerators")
CALLBACKS     = factory.Factory(name="Callbacks")
DATAMODULES   = factory.Factory(name="DataModules")
DATASETS      = factory.Factory(name="Datasets")
DETECTORS     = factory.Factory(name="Detectors")
DISTANCES     = factory.Factory(name="Distances")
EMBEDDERS     = factory.Factory(name="Embedders")
FILE_HANDLERS = factory.Factory(name="FileHandlers")
LOGGERS       = factory.Factory(name="Loggers")
LOSSES        = factory.Factory(name="Losses")
LR_SCHEDULERS = factory.Factory(name="LRSchedulers")
METRICS       = factory.Factory(name="Metrics")
MODELS        = factory.ModelFactory(name="Models")
MOTIONS       = factory.Factory(name="Motions")
OBJECTS       = factory.Factory(name="Objects")
OPTIMIZERS    = factory.Factory(name="Optimizers")
STRATEGIES    = factory.Factory(name="Strategies")
TRACKERS      = factory.Factory(name="Trackers")
TRANSFORMS    = factory.Factory(name="Transforms")

# endregion


# region Constants

CONFIG_FILE_FORMATS  = [".config", ".cfg", ".yaml", ".yml", ".py", ".json", ".names", ".txt"]
IMAGE_FILE_FORMATS   = [".arw", ".bmp", ".dng", ".jpg", ".jpeg", ".png", ".ppm", ".raf", ".tif", ".tiff"]
VIDEO_FILE_FORMATS   = [".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".wmv"]
TORCH_FILE_FORMATS   = [".pt", ".pth", ".weights", ".ckpt", ".tar", ".onnx"]
WEIGHTS_FILE_FORMATS = [".pt", ".pth", ".onnx"]
DEPTH_DATA_SOURCES   = [None, "dav2_vitb", "dav2_vitl"]

# List 3rd party modules
EXTRA_DATASET_STR = "[extra]"
EXTRA_MODEL_STR   = "[extra]"
EXTRA_DATASETS    = {}
EXTRA_MODELS      = {                   # architecture/model (+ variant)
    # region dtype/depth
    "depth_anything_v2": {
        "depth_anything_v2_vitb": {
            "tasks"    : [enum.Task.DEPTH],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "dtype" / "depth" / "depth_anything_v2",
        },
        "depth_anything_v2_vits": {
            "tasks"    : [enum.Task.DEPTH],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "dtype" / "depth" / "depth_anything_v2",
        },
        "depth_anything_v2_vitl": {
            "tasks"    : [enum.Task.DEPTH],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "dtype" / "depth" / "depth_anything_v2",
        },
        "depth_anything_v2_vitg": {
            "tasks"    : [enum.Task.DEPTH],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "dtype" / "depth" / "depth_anything_v2",
        },
    },
    "depth_pro"        : {
        "depth_pro": {
            "tasks"    : [enum.Task.DEPTH],
            "ltypes"   : [enum.LType.ZERO_SHOT],
            "model_dir": MON_DIR / "vision" / "dtype" / "depth" / "depth_pro",
        },
    },
    # endregion
    # region detect
    "yolor" : {
        "yolor_d6": {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolor",
        },
        "yolor_e6": {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolor",
        },
        "yolor_p6": {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolor",
        },
        "yolor_w6": {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolor",
        },
    },
    "yolov7": {
        "yolov7"    : {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
        "yolov7_d6" : {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
        "yolov7_e6" : {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
        "yolov7_e6e": {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
        "yolov7_w6" : {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
        "yolov7x"   : {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov7",
        },
    },
    "yolov8": {
        "yolov8n": {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
        },
        "yolov8s": {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
        },
        "yolov8m": {
            "tasks"    : [enum.Task.DETECT],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
        },
        "yolov8l": {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
        },
        "yolov8x": {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
        },
    },
    "yolov9": {
        "gelan_c" : {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov9",
        },
        "gelan_e" : {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov9",
        },
        "yolov9_c": {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov9",
        },
        "yolov9_e": {
            "tasks"    : [enum.Task.DETECT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "detect" / "yolov9",
        },
    },
    # endregion
    # region enhance/dehaze
    "zid"   : {
        "zid": {
            "tasks"    : [enum.Task.DEHAZE],
            "ltypes"   : [enum.LType.ZERO_SHOT],
            "model_dir": MON_DIR / "vision" / "enhance" / "dehaze" / "zid",
        },
    },
    # endregion
    # region enhance/demoire
    "esdnet": {
        "esdnet": {
            "tasks"    : [enum.Task.DEMOIRE, enum.Task.LLIE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "demoire" / "esdnet",
        },
        "esdnet_l": {
            "tasks"    : [enum.Task.DEMOIRE, enum.Task.LLIE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "demoire" / "esdnet",
        },
    },
    # endregion
    # region enhance/derain
    "esdnet_snn": {
        "esdnet_snn": {
            "tasks"    : [enum.Task.DERAIN, enum.Task.LLIE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "derain" / "esdnet_snn",
        },
    },
    # endregion
    # region enhance/llie
    "colie"        : {
        "colie": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.ZERO_SHOT],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "colie",
        },
    },
    "dccnet"       : {
        "dccnet": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "dccnet",
        },
    },
    "enlightengan" : {
        "enlightengan": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.UNSUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "enlightengan",
        },
    },
    "fourllie"     : {
        "fourllie": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "fourllie",
        },
    },
    "hvi_cidnet"   : {
        "hvi_cidnet": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "hvi_cidnet",
        },
    },
    "lime"         : {
        "lime": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.TRADITIONAL],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "lime",
        },
    },
    "llflow"       : {
        "llflow": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "llflow",
        },
    },
    "llunet++"     : {
        "llunet++": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "llunetpp",
        },
    },
    "nerco"        : {
        "nerco": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.UNSUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "nerco",
        },
    },
    "pairlie"      : {
        "pairlie": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.UNSUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "pairlie",
        },
    },
    "pie"          : {
        "pie": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.TRADITIONAL],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "pie",
        },
    },
    "psenet"       : {
        "psenet": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "psenet",
        },
    },
    "quadprior"    : {
        "quadprior": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.ZERO_SHOT],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "quadprior",
        }
    },
    "retinexformer": {
        "retinexformer": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "retinexformer",
        },
    },
    "retinexnet"   : {
        "retinexnet": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "retinexnet",
        },
    },
    "rsfnet"       : {
        "rsfnet": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.UNSUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "rsfnet",
        },
    },
    "ruas"         : {
        "ruas": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.UNSUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "ruas",
        },
    },
    "sci"          : {
        "sci": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.UNSUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "sci",
        },
    },
    "sgz"          : {
        "sgz": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.UNSUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "sgz",
        },
    },
    "snr_net"      : {
        "snr_net": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "snr_net",
        },
    },
    "uretinexnet"  : {
        "uretinexnet": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "uretinexnet",
        },
    },
    "utvnet"       : {
        "utvnet": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "utvnet",
        },
    },
    "zero_dce"     : {
        "zero_dce"  : {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.UNSUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "zero_dce",
        },
    },
    "zero_dce++"   : {
        "zero_dce++": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.UNSUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "zero_dce++",
        },
    },
    "zero_didce"   : {
        "zero_didce": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.UNSUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "zero_didce",
        },
    },
    "zero_ig"      : {
        "zero_ig": {
            "tasks"    : [enum.Task.LLIE],
            "ltypes"   : [enum.LType.ZERO_SHOT],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "zero_ig",
        },
    },
    # endregion
    # region enhance/multitask
    "airnet"   : {
        "airnet": {
            "tasks"    : [enum.Task.DENOISE, enum.Task.DERAIN, enum.Task.DEHAZE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "multitask" / "airnet",
        },
    },
    "restormer": {
        "restormer": {
            "tasks"    : [enum.Task.DEBLUR, enum.Task.DENOISE, enum.Task.DERAIN, enum.Task.DESNOW, enum.Task.LLIE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "multitask" / "restormer",
        },
    },
    # endregion
    # region enhance/retouch
    "neurop": {
        "neurop": {
            "tasks"    : [enum.Task.RETOUCH, enum.Task.LLIE],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "retouch" / "neurop",
        },
    },
    # endregion
    # region enhance/sr
    "sronet": {
        "sronet": {
            "tasks"    : [enum.Task.SR],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "sr" / "sronet",
        },
    },
    # endregion
    # region segment
    "sam" : {
        "sam_vit_b": {
            "tasks"    : [enum.Task.SEGMENT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam",
        },
        "sam_vit_h": {
            "tasks"    : [enum.Task.SEGMENT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam",
        },
        "sam_vit_l": {
            "tasks"    : [enum.Task.SEGMENT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam",
        },
    },
    "sam2": {
        "sam2_hiera_b+": {
            "tasks"    : [enum.Task.SEGMENT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam2",
        },
        "sam2_hiera_l" : {
            "tasks"    : [enum.Task.SEGMENT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam2",
        },
        "sam2_hiera_s" : {
            "tasks"    : [enum.Task.SEGMENT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam2",
        },
        "sam2_hiera_t" : {
            "tasks"    : [enum.Task.SEGMENT],
            "ltypes"   : [enum.LType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "segment" / "sam2",
        },
    },
    # endregion
}

# endregion
