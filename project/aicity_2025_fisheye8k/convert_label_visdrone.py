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

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]

map_classes  = {
    "0":  -1,  # VisDrone ignored regions  -> ignore
    "1":   3,  # VisDrone pedestrian       -> Fisheye8K pedestrian
    "2":  -1,  # VisDrone human            -> ignore
    "3":  -1,  # VisDrone bicycle          -> ignore
    "4":   2,  # VisDrone car              -> Fisheye8K car
    "5":   2,  # VisDrone van              -> Fisheye8K car
    "6":   4,  # VisDrone truck            -> Fisheye8K truck
    "7":  -1,  # VisDrone tricycle         -> ignore
    "8":  -1,  # VisDrone awning-tricycle  -> ignore
    "9":   0,  # VisDrone bus              -> Fisheye8K bus
    "10":  1,  # VisDrone motor            -> Fisheye8K bike
    "11": -1,  # VisDrone others           -> ignore
}


def convert_label_visdrone2fisheye8k(split: str):
    images_dir     = current_dir / "data" / "fisheye8k" / "extra" / "visdrone" / "images"
    labels_old_dir = current_dir / "data" / "fisheye8k" / "extra" / "visdrone" / "labels_old_dir"
    label_dir      = current_dir / "data" / "fisheye8k" / "extra" / "visdrone" / "labels"
    
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
            h, w, c = mon.read_image_shape(image_file)
            
            labels_old_file = labels_old_dir / f"{image_file.stem}.txt"
            label_file      = label_dir      / f"{image_file.stem}.txt"
            label_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Read the annotation file
            with open(labels_old_file, "r") as f:
                bboxes = f.readlines()
            
            # Open the new label file
            f = open(label_file, "w")
            for bbox in bboxes:
                args     = bbox.split(",")
                left     = int(args[0])
                top      = int(args[1])
                bbox_w   = int(args[2])
                bbox_h   = int(args[3])
                score    = int(args[4])
                category = str(args[5])
                category = map_classes[category]
                # Ignored classes
                if category == -1:
                    continue
                # VisDrone Ignored bounding boxes
                if score == 0:
                    continue
                # Convert
                center_x = left + bbox_w // 2
                center_y = top  + bbox_h // 2
                center_x = round(float(center_x) / w, 6)
                center_y = round(float(center_y) / h, 6)
                bbox_w   = round(float(bbox_w)   / w, 6)
                bbox_h   = round(float(bbox_h)   / h, 6)
                f.write("{} {} {} {} {}\n".format(category, center_x, center_y, bbox_w, bbox_h))
            f.close()


if __name__ == "__main__":
    convert_label_visdrone2fisheye8k("train")
    #convert_visdrone_to_fisheye8k("val")
    #convert_visdrone_to_fisheye8k("test_dev")
