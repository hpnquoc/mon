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
	image_dir = current_dir / "data" / "fisheye8k" / data / "image"
	label_dir = current_dir / "data" / "fisheye8k" / data / "label"
	
	assert mon.Path(image_dir).is_dir()
	assert mon.Path(label_dir).is_dir()
	
	code   = mon.ShapeCode.from_value(value=f"yolo_to_voc")
	colors = [
		[  0,   0, 255],  # 0: bus        - red
		[255,   0,   0],  # 1: bike       - blue
		[  0, 255,   0],  # 2: car        - green
		[  0, 255, 255],  # 3: truck      - yellow
		[255,   0, 255],  # 4: pedestrian - magenta
	]
	
	image_files = list(image_dir.rglob("*"))
	image_files = sorted([f for f in image_files if f.is_image_file()])
	with mon.create_progress_bar() as pbar:
		for image_file in pbar.track(
			sequence    = image_files,
			total       = len(image_files),
			description = f"[bright_yellow] Processing"
		):
			image   = cv2.imread(str(image_file))
			h, w, _ = image.shape
			
			label_file = label_dir / f"{image_file.stem}.txt"
			if not label_file.is_txt_file(exist=True):
				continue
			
			with open(label_file, "r") as f:
				bs = f.readlines()
			
			bs = [b.strip().split(" ") for b in bs]
			bs = np.array([list(map(float, b[1:])) for b in bs])
			if len(bs) == 0:
				continue
			bs = mon.convert_bbox(bbox=bs, code=code, height=h, width=w)
			
			for j, x in enumerate(bs):
				image = mon.draw_bbox(
					image     = image,
					bbox      = x,
					label     = None,  # l[j] if show_label else None,
					color     = colors[int(bs[j][0])],
					thickness = 1,
					fill      = fill,
				)
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
			output_file = image_file.replace("image", "visualize")
			output_file = output_file.parent / f"{image_file.stem}.jpg"
			output_file.parent.mkdir(parents=True, exist_ok=True)
			cv2.imwrite(str(output_file), image)


if __name__ == "__main__":
	'''
	visualize_bbox(data="train/camera3_a")
	visualize_bbox(data="train/camera3_n")
	visualize_bbox(data="train/camera5_a")
	visualize_bbox(data="train/camera6_a")
	visualize_bbox(data="train/camera8_a")
	visualize_bbox(data="train/camera9_a")
	visualize_bbox(data="train/camera10_a")
	visualize_bbox(data="train/camera11_m")
	visualize_bbox(data="train/camera12_a")
	visualize_bbox(data="train/camera13_a_500")
	visualize_bbox(data="train/camera13_a_779")
	visualize_bbox(data="train/camera14_a")
	visualize_bbox(data="train/camera15_a")
	visualize_bbox(data="train/camera16_a")
	visualize_bbox(data="train/camera17_a")
	visualize_bbox(data="train/camera18_a")
	'''
	'''
	visualize_bbox(data="train_syn/camera3_e")
	visualize_bbox(data="train_syn/camera5_e")
	visualize_bbox(data="train_syn/camera6_e")
	visualize_bbox(data="train_syn/camera8_e")
	visualize_bbox(data="train_syn/camera9_e")
	visualize_bbox(data="train_syn/camera10_e")
	visualize_bbox(data="train_syn/camera12_e")
	visualize_bbox(data="train_syn/camera13_e")
	visualize_bbox(data="train_syn/camera14_e")
	visualize_bbox(data="train_syn/camera15_e")
	visualize_bbox(data="train_syn/camera16_e")
	visualize_bbox(data="train_syn/camera17_e")
	visualize_bbox(data="train_syn/camera18_e")
	'''
	'''
	visualize_bbox(data="val/camera1_a_test")
	visualize_bbox(data="val/camera2_a_test")
	visualize_bbox(data="val/camera4_a_e_m_n_test")
	visualize_bbox(data="val/camera7_a_test")
	'''
	'''
	visualize_bbox(data="val_syn/camera1_e")
	visualize_bbox(data="val_syn/camera2_e")
	visualize_bbox(data="val_syn/camera7_e")
	'''
	
	visualize_bbox(data="test/camera19_a")
	visualize_bbox(data="test/camera20_a")
	visualize_bbox(data="test/camera21_a")
	visualize_bbox(data="test/camera22_a")
	visualize_bbox(data="test/camera23_a")
	visualize_bbox(data="test/camera24_a")
	visualize_bbox(data="test/camera25_a")
	visualize_bbox(data="test/camera26_a")
	visualize_bbox(data="test/camera27_a")
	visualize_bbox(data="test/camera28_a")
	visualize_bbox(data="test/camera29_a_n")
	
	# visualize_bbox(data="visdrone/train")
	# visualize_bbox(data="visdrone/val")
	# visualize_bbox(data="visdrone/test_dev")
	# visualize_bbox(data="hcmaicity/train")
