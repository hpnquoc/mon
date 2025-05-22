#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert YOLO bbox to COCO format:
{
    "info": {
        "year": "2020",
        "version": "1",
        "description": "Exported from roboflow.ai",
        "contributor": "Roboflow",
        "url": "https://app.roboflow.ai/datasets/hard-hat-sample/1",
        "date_created": "2000-01-01T00:00:00+00:00"
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "name": "Public Domain"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "Workers",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "head",
            "supercategory": "Workers"
        },
        {
            "id": 2,
            "name": "helmet",
            "supercategory": "Workers"
        },
        {
            "id": 3,
            "name": "person",
            "supercategory": "Workers"
        }
    ],
    "images": [
        {
            "id": 0,
            "license": 1,
            "file_name": "0001.jpg",
            "height": 275,
            "width": 490,
            "date_captured": "2020-07-20T19:39:26+00:00"
        }
    ],
    "annotations": [
        {
            "id": 0,
            "image_id": 0,
            "category_id": 2,
            "bbox": [
                45,
                2,
                85,
                85
            ],
            "area": 7225,
            "segmentation": [],
            "iscrowd": 0
        },
        {
            "id": 1,
            "image_id": 0,
            "category_id": 2,
            "bbox": [
                324,
                29,
                72,
                81
            ],
            "area": 5832,
            "segmentation": [],
            "iscrowd": 0
        }
    ]
}
"""

import argparse
import json

import numpy as np

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


def convert_label_to_coco(split: str):
    code        = mon.ShapeCode.from_value(value=f"yolo_to_coco")
    image_dir   = current_dir / "data" / split / "image"
    label_dir   = current_dir / "data" / split / "label"
    json_file   = current_dir / "data" / split / f"{split}.json"
    
    assert mon.Path(image_dir).is_dir()
    assert mon.Path(label_dir).is_dir()
    
    image_files = sorted([f for f in list(image_dir.rglob("*")) if f.is_image_file()])
    
    # COCO JSON Format
    info        = {
        "year"        : "2025",
        "version"     : "1",
        "description" : "Custom Dataset for AICity Challenge 2025 Track 4 Fisheye8K",
        "contributor" : "Long H. Pham",
        "url"         : "",
        "date_created": "2025-05-08"
    }
    licenses    = []
    categories  = [
        {"id": 0, "name": "Bus"},
        {"id": 1, "name": "Bike"},
        {"id": 2, "name": "Car"},
        {"id": 3, "name": "Pedestrian"},
        {"id": 4, "name": "Truck"},
    ]
    images      = []
    annotations = []
    ann_id      = 0
    
    with mon.create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence    = enumerate(image_files),
            total       = len(image_files),
            description = f"[bright_yellow] Processing"
        ):
            # Append image
            h, w, c  = mon.read_image_shape(image_file)
            image_id = i
            images.append({"id": image_id, "file_name": image_file.name, "height": h, "width": w})
            
            # Append annotations
            label_file = None
            for j in range(0, 4):
                file = image_file.parents[j] / "label" / f"{image_file.stem}.txt"
                if file.is_txt_file():
                    label_file = file
                    break
            if label_file is None or not label_file.is_txt_file(exist=True):
                continue
            
            # Read the yolo label file and convert bbox format
            with open(label_file, "r") as f:
                l = f.readlines()
            l = [x.strip().split(" ") for x in l]
            l = [x for x in l if len(x) >= 5]
            if len(l) == 0:
                continue
            b = np.array([list(map(float, x[1:5])) for x in l])
            b = mon.convert_bbox(bbox=b, code=code, height=h, width=w)
            assert len(b) == len(l)
            
            for l_, b_ in zip(l, b):
                annotations.append({
                    "id"         : ann_id,
                    "image_id"   : image_id,
                    "category_id": int(l_[0]),
                    "bbox"       : [int(b_[0]), int(b_[1]), int(b_[2]), int(b_[3])],
                    "area"       : int(b_[2] * b_[3]),
                    "iscrowd"    : 0,
                })
                ann_id += 1
            
    # Write to JSON file
    json_data = {
        "info"       : info,
        "licenses"   : licenses,
        "categories" : categories,
        "images"     : images,
        "annotations": annotations
    }
    with open(str(json_file), "w") as f:
        json.dump(json_data, f, indent=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", required=True)
    args = parser.parse_args()
    
    convert_label_to_coco(args.split)
