#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script visualizes bounding boxes on images."""

from __future__ import annotations

import cv2
import numpy as np

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


def visualize_bbox(data: str, fill: bool = False):
	image_dir = current_dir / "data" / data / "images"
	label_dir = current_dir / "data" / data / "labels"
	
	assert mon.Path(image_dir).is_dir()
	assert mon.Path(label_dir).is_dir()
	
	code   = mon.ShapeCode.from_value(value=f"yolo_to_voc")
	colors = [
		[  0,   0, 255],
		[255,   0,   0],
		[  0, 255,   0],
		[  0, 255, 255],
		[255,   0, 255],
	]
	
	image_files = list(image_dir.rglob("*"))
	image_files = sorted([f for f in image_files if f.is_image_file()])
	with mon.create_progress_bar() as pbar:
		for image_file in pbar.track(
			sequence    = image_files,
			total       = len(image_files),
			description = f"[bright_yellow] Processing"
		):
			image      = cv2.imread(str(image_file))
			h, w, c    = image.shape
			
			label_file = label_dir / f"{image_file.stem}.txt"
			if not label_file.is_txt_file(exist=True):
				continue
			
			with open(label_file, "r") as f:
				bboxes = f.readlines()
			
			bboxes = [b.strip().split(" ") for b in bboxes]
			b = np.array([list(map(float, b[1:])) for b in bboxes])
			if len(b) == 0:
				continue
			b = mon.convert_bbox(bbox=b, code=code, height=h, width=w)
			
			for j, x in enumerate(b):
				image = mon.draw_bbox(
					image     = image,
					bbox      = x,
					label     = None,  # l[j] if show_label else None,
					color     = colors[int(bboxes[j][0])],
					thickness = 1,
					fill      = fill,
				)
				
			'''
			image = cv2.putText(
				img       = image,
				text      = f"{image_file.stem}",
				org       = [50, 50],
				fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
				fontScale = 1,
				color     = [255, 255, 255],
				thickness = 3,
				lineType  = cv2.LINE_AA,
			)
			'''
			output_file = image_file.replace("images", "visualize")
			output_file = output_file.parent / f"{image_file.stem}.jpg"
			output_file.parent.mkdir(parents=True, exist_ok=True)
			cv2.imwrite(str(output_file), image)


if __name__ == "__main__":
	visualize_bbox(data="visdrone/train")
	visualize_bbox(data="visdrone/val")
	visualize_bbox(data="visdrone/test_dev")
