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
from datetime import datetime

import numpy as np

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


def convert_label_to_coco(split: str):
    image_dir = current_dir / "data" / "fisheye8k" / split / "image"
    label_dir = current_dir / "data" / "fisheye8k" / split / "label"
    json_file = current_dir / "data" / "fisheye8k" / split / f"{split}.json"
    
    assert mon.Path(image_dir).is_dir()
    assert mon.Path(label_dir).is_dir()

    # COCO JSON Format
    info        = {
        "year"        : f"{datetime.now().year}",
        "version"     : "1",
        "description" : "Custom Dataset for AICity Challenge 2025 Track 4 Fisheye8K",
        "contributor" : "Long H. Pham",
        "url"         : "",
        "date_created": f"{datetime.now()}"
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

    image_files = sorted([f for f in list(image_dir.rglob("*")) if f.is_image_file()])
    with mon.create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence    = enumerate(image_files),
            total       = len(image_files),
            description = f"[bright_yellow] Processing"
        ):
            # Append image
            h, w, _  = mon.read_image_shape(image_file)
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
            
            # Read the YOLO label file and convert bbox format
            """
            with open(label_file, "r") as f:
                ls = f.readlines()
            ls = [l.strip().split(" ") for l in ls]
            ls = [l for l in ls if len(l) >= 5]
            if len(ls) == 0:
                continue
            bs = np.array([list(map(float, l[1:5])) for l in ls])
            bs = mon.convert_bbox(bbox=bs, code=mon.BBoxFormat.YOLO2COCO, height=h, width=w)
            assert len(bs) == len(ls)
            """
            # bs = mon.load_bbox(path=label_file, format=mon.BBoxFormat.YOLO)
            # bs = mon.convert_bbox(bbox=bs, code=mon.BBoxFormat.YOLO2COCO, height=h, width=w)
            bs = mon.load_bbox(path=label_file, fmt=mon.BBoxFormat.YOLO2COCO, height=h, width=w)
            if len(bs) == 0:
                continue

            # Append annotations
            for b in bs:
                annotations.append({
                    "id"         : ann_id,
                    "image_id"   : image_id,
                    "category_id": int(b[4]),
                    "bbox"       : [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
                    "area"       : int(b[2] * b[3]),
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
    parser.add_argument("--split", type=str, default="val", required=True)
    args = parser.parse_args()
    
    convert_label_to_coco(args.split)
