#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert UA-DETRAC bbox to yolo format."""

from __future__ import annotations

import cv2

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]

map_classes  = {
    "0": 4,    # UA-DETRAC truck -> Fisheye8K truck
    "1": 2,    # UA-DETRAC car   -> Fisheye8K car
    "2": 2,    # UA-DETRAC van   -> Fisheye8K car
    "3": 0,    # UA-DETRAC bus   -> Fisheye8K bus
}


def convert_ua_detrac_to_fisheye8k(split: str):
    images_old_dir = current_dir / "data" / "ua_detrac" / "images_old"
    labels_old_dir = current_dir / "data" / "ua_detrac" / "labels_old_cls"
    images_dir     = current_dir / "data" / "ua_detrac" / "images"
    label_dir      = current_dir / "data" / "ua_detrac" / "labels"
    
    assert mon.Path(images_old_dir).is_dir()
    assert mon.Path(labels_old_dir).is_dir()
    
    image_files = list(images_old_dir.rglob("*"))
    image_files = sorted([f for f in image_files if f.is_image_file()])
    with mon.create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence    = enumerate(image_files),
            total       = len(image_files),
            description = f"[bright_yellow] Processing"
        ):
            if i % 25 != 0:  # The video is recorded at 25 fps -> Every second takes 1 frame.
                continue
                
            image   = cv2.imread(str(image_file))
            h, w, c = image.shape

            label_old_file = labels_old_dir / f"{image_file.stem}.txt"
            label_file     = label_dir      / f"{image_file.stem}.txt"
            label_file.parent.mkdir(parents=True, exist_ok=True)
            if not label_old_file.is_file():
                continue
            
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
            
            # Save image
            images_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(images_dir / f"{image_file.stem}.jpg"), image)
            

if __name__ == "__main__":
    convert_ua_detrac_to_fisheye8k("train")
    # convert_ua_detrac_to_fisheye8k("val")
