#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert UA-DETRAC bbox to yolo format."""

from __future__ import annotations

import cv2

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]

map_classes  = {
    "0": 4,  # UA-DETRAC truck -> Fisheye8K truck
    "1": 2,  # UA-DETRAC car   -> Fisheye8K car
    "2": 2,  # UA-DETRAC van   -> Fisheye8K car
    "3": 0,  # UA-DETRAC bus   -> Fisheye8K bus
}


def convert_label_uadetrac2fisheye8k(split: str):
    image_old_dir = current_dir / "data" / "fisheye8k" / "extra" / "ua_detrac" / "image_old"
    label_old_dir = current_dir / "data" / "fisheye8k" / "extra" / "ua_detrac" / "label_old_cls"
    image_dir     = current_dir / "data" / "fisheye8k" / "extra" / "ua_detrac" / "image"
    label_dir     = current_dir / "data" / "fisheye8k" / "extra" / "ua_detrac" / "label"
    
    assert mon.Path(image_old_dir).is_dir()
    assert mon.Path(label_old_dir).is_dir()
    
    image_files = sorted([f for f in list(image_old_dir.rglob("*")) if f.is_image_file()])
    with mon.create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence    = enumerate(image_files),
            total       = len(image_files),
            description = f"[bright_yellow] Processing"
        ):
            if i % 25 != 0:  # The video is recorded at 25 fps -> Every second takes 1 frame.
                continue

            image   = cv2.imread(str(image_file))
            h, w, _ = image.shape

            label_old_file = label_old_dir / f"{image_file.stem}.txt"
            label_file     = label_dir     / f"{image_file.stem}.txt"
            label_file.parent.mkdir(parents=True, exist_ok=True)
            if not label_old_file.is_file():
                continue
            
            # Read the annotation file
            with open(label_old_file, "r") as f:
                bs = f.readlines()
            
            # Open the new label file
            f = open(label_file, "w")
            for b in bs:
                args = b.split(" ")
                c    = str(args[0])
                c    = map_classes[c]
                b1   = round(float(args[1]), 6)
                b2   = round(float(args[2]), 6)
                b3   = round(float(args[3]), 6)
                b4   = round(float(args[4]), 6)
                # Ignored classes
                if c == -1:
                    continue
                f.write("{} {} {} {} {}\n".format(c, b1, b2, b3, b4))
            f.close()
            
            # Save image
            image_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_dir / f"{image_file.stem}.jpg"), image)
            

if __name__ == "__main__":
    convert_label_uadetrac2fisheye8k("train")
    # convert_ua_detrac_to_fisheye8k("val")
