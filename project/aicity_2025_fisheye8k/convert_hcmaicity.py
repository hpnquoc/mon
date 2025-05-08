#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert VisDrone bbox to yolo format.

VisDrone format:

    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    
    
        Name                                                  Description
    -------------------------------------------------------------------------------------------------------------------------------
     <bbox_left>	     The x coordinate of the top-left corner of the predicted bounding box
    
     <bbox_top>	     The y coordinate of the top-left corner of the predicted object bounding box
    
     <bbox_width>	     The width in pixels of the predicted object bounding box
    
    <bbox_height>	     The height in pixels of the predicted object bounding box
    
       <score>	     The score in the DETECTION file indicates the confidence of the predicted bounding box enclosing
                         an object instance.
                         The score in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding box is considered in evaluation,
                         while 0 indicates the bounding box will be ignored.
                         
    <object_category>    The object category indicates the type of annotated object, (i.e., ignored regions(0), pedestrian(1),
                         people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10),
                         others(11))
                         
    <truncation>	     The score in the DETECTION result file should be set to the constant -1.
                         The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame
                         (i.e., no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% ~ 50%)).
                         
    <occlusion>	     The score in the DETECTION file should be set to the constant -1.
                         The score in the GROUNDTRUTH file indicates the fraction of objects being occluded (i.e., no occlusion = 0
                         (occlusion ratio 0%), partial occlusion = 1 (occlusion ratio 1% ~ 50%), and heavy occlusion = 2
                         (occlusion ratio 50% ~ 100%)).
"""

from __future__ import annotations

import cv2

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]

map_classes  = {
    "0": 1,    # HCM AICity bike, bicycle    -> ignore bicycle, Fisheye8K bike
    "1": 2,    # HCM AICity car              -> Fisheye8K car
    "2": 0,    # HCM AICity bus              -> Fisheye8K bus
    "3": 4,    # HCM AICity truck            -> Fisheye8K truck
}


def hcmaicity_to_fisheye8k(split: str):
    images_dir     = current_dir / "data" / "hcmaicity" / split / "images"
    labels_old_dir = current_dir / "data" / "hcmaicity" / split / "labels_old_cls"
    label_dir      = current_dir / "data" / "hcmaicity" / split / "labels"
    
    assert mon.Path(images_dir).is_dir()
    assert mon.Path(labels_old_dir).is_dir()
    
    image_files = list(images_dir.rglob("*"))
    image_files = sorted([f for f in image_files if f.is_image_file()])
    with mon.create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence    = enumerate(image_files),
            total       = len(image_files),
            description = f"[bright_yellow] Processing"
        ):
            image   = cv2.imread(str(image_file))
            h, w, c = image.shape
            
            label_old_file = labels_old_dir / f"{image_file.stem}.txt"
            label_file     = label_dir      / f"{image_file.stem}.txt"
            label_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Read the annotation file
            with open(label_old_file, "r") as f:
                bboxes = f.readlines()
            
            # Open the new label file
            f = open(label_file, "w")
            for bbox in bboxes:
                args     = bbox.split(" ")
                category = str(args[0])
                category = map_classes[category]
                b1       = round(float(args[1]), 6)
                b2       = round(float(args[2]), 6)
                b3       = round(float(args[3]), 6)
                b4       = round(float(args[4]), 6)
                # Ignored classes
                if category == -1:
                    continue
                f.write("{} {} {} {} {}\n".format(category, b1, b2, b3, b4))
            f.close()


if __name__ == "__main__":
    hcmaicity_to_fisheye8k("train")
