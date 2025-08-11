#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ultralytics YOLOs model for object detection, classification, segmentation,
orientation bounding box detection, and pose estimation.

References:
    - Code: https://github.com/ultralytics/ultralytics
"""

__all__ = [
    "YOLOv11l",
    "YOLOv11l_CLS",
    "YOLOv11l_OBB",
    "YOLOv11l_POSE",
    "YOLOv11l_SEG",
    "YOLOv11m",
    "YOLOv11m_CLS",
    "YOLOv11m_OBB",
    "YOLOv11m_POSE",
    "YOLOv11m_SEG",
    "YOLOv11n",
    "YOLOv11n_CLS",
    "YOLOv11n_OBB",
    "YOLOv11n_POSE",
    "YOLOv11n_SEG",
    "YOLOv11s",
    "YOLOv11s_CLS",
    "YOLOv11s_OBB",
    "YOLOv11s_POSE",
    "YOLOv11s_SEG",
    "YOLOv11x",
    "YOLOv11x_CLS",
    "YOLOv11x_OBB",
    "YOLOv11x_POSE",
    "YOLOv11x_POSE",
    "YOLOv11x_SEG",
    "YOLOv12",
    "YOLOv12l",
    "YOLOv12m",
    "YOLOv12n",
    "YOLOv12s",
    "YOLOv12x",
]

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task, ZOO_DIR
from mon.core import pathlib
from ultralytics import YOLO

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- YOLOv11 -----
class YOLOv11(YOLO, nn.ModelMixin):
    
    arch     : str          = "yolov11"
    name     : str          = "yolov11"
    tasks    : list[Task]   = [Task.DETECT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: str = "coco80", *args, **kwargs):
        if weights in self.zoo:
            model = self.zoo[weights].path
        elif pathlib.Path(weights).is_weights_file(exist=True):
            model = weights
        else:
            model = f"{self.name}.pt"
        super().__init__(model=model, *args, **kwargs)
        

@MODELS.register(name="yolov11n", arch="yolov11")
class YOLOv11n(YOLOv11, nn.ModelMixin):
    
    arch     : str          = "yolov11"
    name     : str          = "yolov11n"
    tasks    : list[Task]   = [Task.DETECT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11n/coco80/yolov11n_coco80.pt",
            "num_classes": 80,
        },
        "widerface": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11n/widerface/yolov11n_widerface.pt",
            "num_classes": 1,
        },
    })
    

@MODELS.register(name="yolov11s", arch="yolov11")
class YOLOv11s(YOLOv11, nn.ModelMixin):
    
    arch     : str          = "yolov11"
    name     : str          = "yolov11s"
    tasks    : list[Task]   = [Task.DETECT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11s/coco80/yolov11s_coco80.pt",
            "num_classes": 80,
        },
        "widerface": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11s/widerface/yolov11s_widerface.pt",
            "num_classes": 1,
        },
    })


@MODELS.register(name="yolov11m", arch="yolov11")
class YOLOv11m(YOLOv11, nn.ModelMixin):
    
    arch     : str          = "yolov11"
    name     : str          = "yolov11m"
    tasks    : list[Task]   = [Task.DETECT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11m/coco80/yolov11m_coco80.pt",
            "num_classes": 80,
        },
    })


@MODELS.register(name="yolov11l", arch="yolov11")
class YOLOv11l(YOLOv11, nn.ModelMixin):
    
    arch     : str          = "yolov11"
    name     : str          = "yolov11l"
    tasks    : list[Task]   = [Task.DETECT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11l/coco80/yolov11l_coco80.pt",
            "num_classes": 80,
        },
    })


@MODELS.register(name="yolov11x", arch="yolov11")
class YOLOv11x(YOLOv11, nn.ModelMixin):
    
    arch     : str          = "yolov11"
    name     : str          = "yolov11x"
    tasks    : list[Task]   = [Task.DETECT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11x/coco80/yolov11x_coco80.pt",
            "num_classes": 80,
        },
    })


# ----- YOLOv11-OBB -----
@MODELS.register(name="yolov11n-obb", arch="yolov11-obb")
class YOLOv11n_OBB(YOLOv11, nn.ModelMixin):
    
    arch     : str          = "yolov11-obb"
    name     : str          = "yolov11n-obb"
    tasks    : list[Task]   = [Task.OBB]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11n_obb/dotav1/yolov11n_obb_dotav1.pt",
            "num_classes": 15,
        },
    })


@MODELS.register(name="yolov11s-obb", arch="yolov11-obb")
class YOLOv11s_OBB(YOLOv11, nn.ModelMixin):
    
    arch     : str          = "yolov11-obb"
    name     : str          = "yolov11s-obb"
    tasks    : list[Task]   = [Task.OBB]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11s_obb/dotav1/yolov11s_obb_dotav1.pt",
            "num_classes": 15,
        },
    })


@MODELS.register(name="yolov11m-obb", arch="yolov11-obb")
class YOLOv11m_OBB(YOLOv11, nn.ModelMixin):
    
    arch     : str          = "yolov11-obb"
    name     : str          = "yolov11m-obb"
    tasks    : list[Task]   = [Task.OBB]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11m_obb/dotav1/yolov11m_obb_dotav1.pt",
            "num_classes": 15,
        },
    })


@MODELS.register(name="yolov11l-obb", arch="yolov11-obb")
class YOLOv11l_OBB(YOLOv11, nn.ModelMixin):
    
    arch     : str          = "yolov11-obb"
    name     : str          = "yolov11l-obb"
    tasks    : list[Task]   = [Task.OBB]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11l_obb/dotav1/yolov11l_obb_dotav1.pt",
            "num_classes": 15,
        },
    })


@MODELS.register(name="yolov11x-obb", arch="yolov11-obb")
class YOLOv11x_OBB(YOLOv11, nn.ModelMixin):
    
    arch     : str          = "yolov11-obb"
    name     : str          = "yolov11x-obb"
    tasks    : list[Task]   = [Task.OBB]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11x_obb/dotav1/yolov11x_obb_dotav1.pt",
            "num_classes": 15,
        },
    })


# ----- YOLOv11-SEG -----
@MODELS.register(name="yolov11n-seg", arch="yolov11-seg")
class YOLOv11n_SEG(YOLOv11, nn.ModelMixin):
    
    arch     : str          = "yolov11-seg"
    name     : str          = "yolov11n-seg"
    tasks    : list[Task]   = [Task.SEGMENT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11n_seg/coco80/yolov11n_seg_coco80.pt",
            "num_classes": 80,
        },
    })


@MODELS.register(name="yolov11s-seg", arch="yolov11-seg")
class YOLOv11s_SEG(YOLOv11, nn.ModelMixin):
    
    arch     : str          = "yolov11-seg"
    name     : str          = "yolov11s-seg"
    tasks    : list[Task]   = [Task.SEGMENT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11s_seg/coco80/yolov11s_seg_coco80.pt",
            "num_classes": 80,
        },
    })
    
    
@MODELS.register(name="yolov11m-seg", arch="yolov11-seg")
class YOLOv11m_SEG(YOLOv11, nn.ModelMixin):
    
    arch     : str          = "yolov11-seg"
    name     : str          = "yolov11m-seg"
    tasks    : list[Task]   = [Task.SEGMENT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11m_seg/coco80/yolov11m_seg_coco80.pt",
            "num_classes": 80,
        },
    })


@MODELS.register(name="yolov11l-seg", arch="yolov11-seg")
class YOLOv11l_SEG(YOLOv11, nn.ModelMixin):
    
    arch     : str          = "yolov11-seg"
    name     : str          = "yolov11l-seg"
    tasks    : list[Task]   = [Task.SEGMENT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11l_seg/coco80/yolov11l_seg_coco80.pt",
            "num_classes": 80,
        },
    })


@MODELS.register(name="yolov11x-seg", arch="yolov11-seg")
class YOLOv11x_SEG(YOLOv11, nn.ModelMixin):
    
    arch     : str          = "yolov11-seg"
    name     : str          = "yolov11l-seg"
    tasks    : list[Task]   = [Task.SEGMENT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11x_seg/coco80/yolov11x_seg_coco80.pt",
            "num_classes": 80,
        },
    })


# ----- YOLOv11-CLS -----
class YOLOv11_CLS(YOLO, nn.ModelMixin):
    
    arch     : str          = "yolov11-cls"
    name     : str          = "yolov11n-cls"
    tasks    : list[Task]   = [Task.CLASSIFY]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: str = "imagenet", *args, **kwargs):
        if weights in self.zoo:
            model = self.zoo[weights].path
        elif pathlib.Path(weights).is_weights_file(exist=True):
            model = weights
        else:
            model = f"{self.name}.pt"
        super().__init__(model=model, *args, **kwargs)
        
        
@MODELS.register(name="yolov11n-cls", arch="yolov11-cls")
class YOLOv11n_CLS(YOLOv11_CLS, nn.ModelMixin):
    
    arch     : str          = "yolov11-cls"
    name     : str          = "yolov11n-cls"
    tasks    : list[Task]   = [Task.CLASSIFY]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "imagenet": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11n_cls/imagenet/yolov11n_cls_imagenet.pt",
            "num_classes": 1000,
        },
    })


@MODELS.register(name="yolov11s-cls", arch="yolov11-cls")
class YOLOv11s_CLS(YOLOv11_CLS, nn.ModelMixin):
    
    arch     : str          = "yolov11-cls"
    name     : str          = "yolov11s-cls"
    tasks    : list[Task]   = [Task.CLASSIFY]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "imagenet": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11s_cls/imagenet/yolov11s_cls_imagenet.pt",
            "num_classes": 1000,
        },
    })


@MODELS.register(name="yolov11m-cls", arch="yolov11-cls")
class YOLOv11m_CLS(YOLOv11_CLS, nn.ModelMixin):
    
    arch     : str          = "yolov11-cls"
    name     : str          = "yolov11m-cls"
    tasks    : list[Task]   = [Task.CLASSIFY]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "imagenet": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11m_cls/imagenet/yolov11m_cls_imagenet.pt",
            "num_classes": 1000,
        },
    })


@MODELS.register(name="yolov11l-cls", arch="yolov11-cls")
class YOLOv11l_CLS(YOLOv11_CLS, nn.ModelMixin):
    
    arch     : str          = "yolov11-cls"
    name     : str          = "yolov11l-cls"
    tasks    : list[Task]   = [Task.CLASSIFY]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "imagenet": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11l_cls/imagenet/yolov11l_cls_imagenet.pt",
            "num_classes": 1000,
        },
    })


@MODELS.register(name="yolov11x-cls", arch="yolov11-cls")
class YOLOv11x_CLS(YOLOv11_CLS, nn.ModelMixin):
    
    arch     : str          = "yolov11-cls"
    name     : str          = "yolov11x-cls"
    tasks    : list[Task]   = [Task.CLASSIFY]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "imagenet": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11x_cls/imagenet/yolov11x_cls_imagenet.pt",
            "num_classes": 1000,
        },
    })


# ----- YOLOv11-POSE -----
class YOLOv11_POSE(YOLO, nn.ModelMixin):
    
    arch     : str          = "yolov11-pose"
    name     : str          = "yolov11n-pose"
    tasks    : list[Task]   = [Task.POSE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: str = "coco1", *args, **kwargs):
        if weights in self.zoo:
            model = self.zoo[weights].path
        elif pathlib.Path(weights).is_weights_file(exist=True):
            model = weights
        else:
            model = f"{self.name}.pt"
        super().__init__(model=model, *args, **kwargs)
        
        
@MODELS.register(name="yolov11n-pose", arch="yolov11-pose")
class YOLOv11n_POSE(YOLOv11_POSE, nn.ModelMixin):
    
    arch     : str          = "yolov11-pose"
    name     : str          = "yolov11n-pose"
    tasks    : list[Task]   = [Task.POSE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco1": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11n_pose/coco1/yolov11n_pose_coco1.pt",
            "num_classes": 1,
        },
    })


@MODELS.register(name="yolov11s-pose", arch="yolov11-pose")
class YOLOv11s_POSE(YOLOv11_POSE, nn.ModelMixin):
    
    arch     : str          = "yolov11-pose"
    name     : str          = "yolov11s-pose"
    tasks    : list[Task]   = [Task.POSE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco1": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11s_pose/coco1/yolov11s_pose_coco1.pt",
            "num_classes": 1,
        },
    })


@MODELS.register(name="yolov11m-pose", arch="yolov11-pose")
class YOLOv11m_POSE(YOLOv11_POSE, nn.ModelMixin):
    
    arch     : str          = "yolov11-pose"
    name     : str          = "yolov11m-pose"
    tasks    : list[Task]   = [Task.POSE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco1": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11m_pose/coco1/yolov11m_pose_coco1.pt",
            "num_classes": 1,
        },
    })


@MODELS.register(name="yolov11l-pose", arch="yolov11-pose")
class YOLOv11l_POSE(YOLOv11_POSE, nn.ModelMixin):
    
    arch     : str          = "yolov11-pose"
    name     : str          = "yolov11l-pose"
    tasks    : list[Task]   = [Task.POSE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco1": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11l_pose/coco1/yolov11l_pose_coco1.pt",
            "num_classes": 1,
        },
    })
    

@MODELS.register(name="yolov11x-pose", arch="yolov11-pose")
class YOLOv11x_POSE(YOLOv11_POSE, nn.ModelMixin):
    
    arch     : str          = "yolov11-pose"
    name     : str          = "yolov11x-pose"
    tasks    : list[Task]   = [Task.POSE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco1": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov11/yolov11x_pose/coco1/yolov11x_pose_coco1.pt",
            "num_classes": 1,
        },
    })


# ----- YOLOv12 -----
class YOLOv12(YOLO, nn.ModelMixin):
    
    arch     : str          = "yolov12"
    name     : str          = "yolov12"
    tasks    : list[Task]   = [Task.DETECT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: str = "coco80", *args, **kwargs):
        if weights in self.zoo:
            model = self.zoo[weights].path
        elif pathlib.Path(weights).is_weights_file(exist=True):
            model = weights
        else:
            model = f"{self.name}.pt"
        super().__init__(model=model, *args, **kwargs)
        

@MODELS.register(name="yolov12n", arch="yolov12")
class YOLOv12n(YOLOv12, nn.ModelMixin):
    
    arch     : str          = "yolov12"
    name     : str          = "yolov12n"
    tasks    : list[Task]   = [Task.DETECT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov12/yolov12n/coco80/yolov12n_coco80.pt",
            "num_classes": 80,
        },
    })
    

@MODELS.register(name="yolov12s", arch="yolov12")
class YOLOv12s(YOLOv12, nn.ModelMixin):
    
    arch     : str          = "yolov12"
    name     : str          = "yolov12s"
    tasks    : list[Task]   = [Task.DETECT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov12/yolov12s/coco80/yolov12s_coco80.pt",
            "num_classes": 80,
        },
    })


@MODELS.register(name="yolov12m", arch="yolov12")
class YOLOv12m(YOLOv12, nn.ModelMixin):
    
    arch     : str          = "yolov12"
    name     : str          = "yolov12m"
    tasks    : list[Task]   = [Task.DETECT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov12/yolov12m/coco80/yolov12m_coco80.pt",
            "num_classes": 80,
        },
    })


@MODELS.register(name="yolov12l", arch="yolov12")
class YOLOv12l(YOLOv12, nn.ModelMixin):
    
    arch     : str          = "yolov12"
    name     : str          = "yolov12l"
    tasks    : list[Task]   = [Task.DETECT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov12/yolov12l/coco80/yolov12l_coco80.pt",
            "num_classes": 80,
        },
    })


@MODELS.register(name="yolov12x", arch="yolov12")
class YOLOv12x(YOLOv12, nn.ModelMixin):
    
    arch     : str          = "yolov12"
    name     : str          = "yolov12x"
    tasks    : list[Task]   = [Task.DETECT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ZOO_DIR / "extra/ultralytics/yolov12/yolov12x/coco80/yolov12x_coco80.pt",
            "num_classes": 80,
        },
    })
