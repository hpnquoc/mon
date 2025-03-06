# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

import mon

data_name   = "lol_v1"
split       = "test"
ref_ext     = ".png
input_dir   = mon.DATA_DIR / "enhance" / data_name / split / "image"
ref_dir     = mon.DATA_DIR / "enhance" / data_name / split / "ref"
output_dir  = mon.Path(f"run/predict/zero_linr/zero_linr_ref_v/{data_name}")

# List image files
image_files = list(input_dir.rglob("*"))
image_files = [f for f in image_files if f.is_image_file()]
image_files = sorted(image_files)
num_items   = len(image_files)

with mon.get_progress_bar() as pbar:
    for image_file in pbar.track(
        sequence    = image_files,
        total       = len(image_files),
        description = "Processing"
    ):
        # Image
        image     = mon.read_image(path=image_file, to_tensor=True, normalize=True)
        h0, w0    = mon.get_image_size(image)
        # Ref
        ref_file  = ref_dir / f"{image_file.stem}{ref_ext}"
        ref       = mon.read_image(path=ref_file, to_tensor=True, normalize=True)
        # HSV
        image_hsv = mon.rgb_to_hsv(image)
        ref_hsv   = mon.rgb_to_hsv(ref)
        print(image_file, image_hsv.shape, ref_hsv.shape)
        if image_hsv.shape != ref_hsv.shape:
            ref_hsv = mon.resize(image_hsv, (h0, w0))
        image_hsv[:, -1, :, :] = ref_hsv[:, -1, :, :]
        output    = mon.hsv_to_rgb(image_hsv)
        # output    = kornia.filters.bilateral_blur(output, (3, 3), 0.5, (1.5, 1.5))
        # Output
        output_path = output_dir / f"{image_file.stem}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mon.write_image(output_path, output)
