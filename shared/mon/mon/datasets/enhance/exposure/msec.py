#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Multi-Scale Exposure Correction (MSEC) datasets.

References:
    - Paper: "Learning Multi-Scale Photo Exposure Correction," CVPR 2021.
    - Code: https://github.com/mahmoudnafifi/Exposure_Correction
"""

__all__ = [
    "MSEC",
]

from mon.core import rich
from mon.datasets.core import *


@DATASETS.register(name="msec")
class MSEC(VisionDataset):
    """MSEC dataset."""
    
    root_name : str         = "msec"
    tasks     : list[Task]  = [Task.EXPOSURE, Task.MEF]
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: Modalities  = {
        "image"     : Modality(name="image_0",    type="image", module=Image, in_test=True, primary=True),
        "image_n1.5": Modality(name="image_n1.5", type="image", module=Image, in_test=True),
        "image_n1"  : Modality(name="image_n1",   type="image", module=Image, in_test=True),
        "image_p1"  : Modality(name="image_p1",   type="image", module=Image, in_test=True),
        "image_p1.5": Modality(name="image_p1.5", type="image", module=Image, in_test=True),
        "ref"       : Modality(name="ref_c",      type="image", module=Image, in_test=True),
    }
    classes   : Classes     = None
    
    def list_primary_data(self) -> list:
        patterns = [self.root / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))
        
        return images
