#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements core components for datasets."""

__all__ = [
    "BaseDataset",
    "BaseTensorOrArray",
    "Classes",
    "DATASETS",
    "DEPTH_SOURCE",
    "DataLoader",
    "DefaultDepthMap",
    "DefaultInfraredMap",
    "DepthMap",
    "DepthSource",
    "Frame",
    "HBBs",
    "INFRARED_SOURCE",
    "Image",
    "ImageLoader",
    "InfraredMap",
    "InfraredSource",
    "Modalities",
    "Modality",
    "Probs",
    "SemanticMask",
    "Split",
    "Task",
    "VideoLoader",
    "VideoLoaderCV",
    "VideoWriter",
    "VideoWriterCV",
    "VideoWriterFFmpeg",
    "VisionDataset",
]

from functools import partial

from mon.constants import DATASETS, DEPTH_SOURCE, INFRARED_SOURCE
from mon.core.data import (
    BaseDataset,
    Classes,
    DataLoader,
    ImageLoader,
    Modalities,
    Modality,
    VideoLoader,
    VideoLoaderCV,
    VisionDataset,
)
from mon.core.dtypes import (
    BaseTensorOrArray,
    DepthMap,
    Frame,
    HBBs,
    Image,
    InfraredMap,
    Probs,
    SemanticMask,
    VideoWriter,
    VideoWriterCV,
    VideoWriterFFmpeg,
)
from mon.core.enum import DepthSource, InfraredSource, Split, Task

DefaultDepthMap    = partial(DepthMap,    source=DEPTH_SOURCE)
DefaultInfraredMap = partial(InfraredMap, source=INFRARED_SOURCE)
