#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Defines scalar constants."""

__all__ = [
    "ACCELERATORS",
    "CALLBACKS",
    "DATAMODULES",
    "DATASETS",
    "DATA_DIR",
    "DETECTORS",
    "DISTANCES",
    "EMBEDDERS",
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
    "MOTIONS",
    "OBJECTS",
    "OPTIMIZERS",
    "ROOT_DIR",
    "SAVE_CKPT_EXT",
    "SAVE_IMAGE_EXT",
    "SAVE_WEIGHTS_EXT",
    "SERIALIZERS",
    "STRATEGIES",
    "TRACKERS",
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
SAVE_CKPT_EXT    = TorchExtension.CKPT.value
SAVE_IMAGE_EXT   = ImageExtension.JPG.value
SAVE_WEIGHTS_EXT = TorchExtension.PT.value
# List 3rd party modules
EXTRA_STR      = "[extra]"
EXTRA_DATASETS = {}
EXTRA_MODELS   = {  # architecture/model (+ variant)
    # region detect
    "yolor" : {
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
    "yolov7": {
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
    "yolov8": {
        "yolov8n": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
        },
        "yolov8s": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
        },
        "yolov8m": {
            "tasks"    : [Task.DETECT],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
        },
        "yolov8l": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
        },
        "yolov8x": {
            "tasks"    : [Task.DETECT],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
        },
    },
    "yolov9": {
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
    # region enhance/dehaze
    "zid"   : {
        "zid": {
            "tasks"    : [Task.DEHAZE],
            "mltypes"  : [MLType.ZERO_SHOT],
            "model_dir": MON_DIR / "vision" / "enhance" / "dehaze" / "zid",
        },
    },
    # endregion
    # region enhance/demoire
    "esdnet": {
        "esdnet"  : {
            "tasks"    : [Task.DEMOIRE, Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "demoire" / "esdnet",
        },
        "esdnet_l": {
            "tasks"    : [Task.DEMOIRE, Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "demoire" / "esdnet",
        },
    },
    # endregion
    # region enhance/derain
    "esdnet_snn": {
        "esdnet_snn": {
            "tasks"    : [Task.DERAIN, Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "derain" / "esdnet_snn",
        },
    },
    # endregion
    # region enhance/llie
    "colie"        : {
        "colie": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.ZERO_SHOT],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "colie",
        },
    },
    "dccnet"       : {
        "dccnet": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "dccnet",
        },
    },
    "enlightengan" : {
        "enlightengan": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.UNPAIRED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "enlightengan",
        },
    },
    "fourllie"     : {
        "fourllie": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "fourllie",
        },
    },
    "hvi_cidnet"   : {
        "hvi_cidnet": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "hvi_cidnet",
        },
    },
    "li2025"       : {
        "li2025": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.UNPAIRED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "li2025",
        },
    },
    "lime"         : {
        "lime": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.TRADITIONAL],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "lime",
        },
    },
    "llflow"       : {
        "llflow": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "llflow",
        },
    },
    "llunet++"     : {
        "llunet++": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "llunet++",
        },
    },
    "nerco"        : {
        "nerco": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.UNPAIRED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "nerco",
        },
    },
    "pairlie"      : {
        "pairlie": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.UNPAIRED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "pairlie",
        },
    },
    "pie"          : {
        "pie": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.TRADITIONAL],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "pie",
        },
    },
    "psenet"       : {
        "psenet": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "psenet",
        },
    },
    "quadprior"    : {
        "quadprior": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.ZERO_SHOT],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "quadprior",
        }
    },
    "retinexformer": {
        "retinexformer": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "retinexformer",
        },
    },
    "retinexnet"   : {
        "retinexnet": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "retinexnet",
        },
    },
    "rsfnet"       : {
        "rsfnet": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.ZERO_REFERENCE],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "rsfnet",
        },
    },
    "ruas"         : {
        "ruas": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.ZERO_REFERENCE],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "ruas",
        },
    },
    "sci"          : {
        "sci": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.ZERO_REFERENCE],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "sci",
        },
    },
    "sgz"          : {
        "sgz": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.ZERO_REFERENCE],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "sgz",
        },
    },
    "snr_net"      : {
        "snr_net": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "snr_net",
        },
    },
    "uretinexnet"  : {
        "uretinexnet": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "uretinexnet",
        },
    },
    "uretinexnet++": {
        "uretinexnet++": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "uretinexnet++",
        },
    },
    "utvnet"       : {
        "utvnet": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "utvnet",
        },
    },
    "zero_dce"     : {
        "zero_dce"  : {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.ZERO_REFERENCE],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "zero_dce",
        },
    },
    "zero_dce++"   : {
        "zero_dce++": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.ZERO_REFERENCE],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "zero_dce++",
        },
    },
    "zero_didce"   : {
        "zero_didce": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.ZERO_REFERENCE],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "zero_didce",
        },
    },
    "zero_ig"      : {
        "zero_ig": {
            "tasks"    : [Task.LLIE],
            "mltypes"  : [MLType.ZERO_SHOT],
            "model_dir": MON_DIR / "vision" / "enhance" / "llie" / "zero_ig",
        },
    },
    # endregion
    # region enhance/multitask
    "airnet"   : {
        "airnet": {
            "tasks"    : [Task.DENOISE, Task.DERAIN, Task.DEHAZE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "multitask" / "airnet",
        },
    },
    "restormer": {
        "restormer": {
            "tasks"    : [Task.DEBLUR, Task.DENOISE, Task.DERAIN, Task.DESNOW, Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "multitask" / "restormer",
        },
    },
    # endregion
    # region enhance/retouch
    "neurop": {
        "neurop": {
            "tasks"    : [Task.RETOUCH, Task.LLIE],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "retouch" / "neurop",
        },
    },
    # endregion
    # region enhance/rr
    "rdnet": {
        "rdnet": {
            "tasks"    : [Task.RR],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "rr" / "rdnet",
        },
    },
    # endregion
    # region enhance/sr
    "sronet": {
        "sronet": {
            "tasks"    : [Task.SR],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "enhance" / "sr" / "sronet",
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
    # region types/depth
    "depth_anything_v2": {
        "depth_anything_v2_vitb": {
            "tasks"    : [Task.DEPTH],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "types" / "depth" / "depth_anything_v2",
        },
        "depth_anything_v2_vits": {
            "tasks"    : [Task.DEPTH],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "types" / "depth" / "depth_anything_v2",
        },
        "depth_anything_v2_vitl": {
            "tasks"    : [Task.DEPTH],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "types" / "depth" / "depth_anything_v2",
        },
        "depth_anything_v2_vitg": {
            "tasks"    : [Task.DEPTH],
            "mltypes"  : [MLType.SUPERVISED],
            "model_dir": MON_DIR / "vision" / "types" / "depth" / "depth_anything_v2",
        },
    },
    "depth_pro"        : {
        "depth_pro": {
            "tasks"    : [Task.DEPTH],
            "mltypes"  : [MLType.ZERO_SHOT],
            "model_dir": MON_DIR / "vision" / "types" / "depth" / "depth_pro",
        },
    },
    # endregion
}


# ----- Factory -----
ACCELERATORS  = factory.Factory(name="Accelerators")
CALLBACKS     = factory.Factory(name="Callbacks")
DATAMODULES   = factory.Factory(name="DataModules")
DATASETS      = factory.Factory(name="Datasets")
DETECTORS     = factory.Factory(name="Detectors")
DISTANCES     = factory.Factory(name="Distances")
EMBEDDERS     = factory.Factory(name="Embedders")
LOGGERS       = factory.Factory(name="Loggers")
LOSSES        = factory.Factory(name="Losses")
LR_SCHEDULERS = factory.Factory(name="LRSchedulers")
METRICS       = factory.Factory(name="Metrics")
MODELS        = factory.ModelFactory(name="Models")
MOTIONS       = factory.Factory(name="Motions")
OBJECTS       = factory.Factory(name="Objects")
OPTIMIZERS    = factory.Factory(name="Optimizers")
SERIALIZERS   = factory.Factory(name="Serializers")
STRATEGIES    = factory.Factory(name="Strategies")
TRACKERS      = factory.Factory(name="Trackers")
TRANSFORMS    = factory.Factory(name="Transforms")
