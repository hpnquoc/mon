# !/usr/bin/env python
# -*- coding: utf-8 -*-
import kornia

import mon

data_name   = "lol_v1"
input_dir   = mon.DATA_DIR / "enhance" / data_name / "test" / "image"
ref_dir     = mon.DATA_DIR / "enhance" / data_name / "test" / "ref"
output_dir  = mon.Path(f"run/predict/zero_linr/zero_linr_rebuttal/{data_name}")

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
        ref_file  = ref_dir / f"{image_file.stem}.png"
        ref       = mon.read_image(path=ref_file, to_tensor=True, normalize=True)
        # HSV
        image_hsv = mon.rgb_to_hsv(image)
        ref_v     = mon.rgb_to_v(ref)
        image_hsv[:, -1, :, :] = ref_v
        output    = mon.hsv_to_rgb(image_hsv)
        # output    = kornia.filters.bilateral_blur(output, (3, 3), 0.5, (1.5, 1.5))
        # Output
        output_path = output_dir / f"{image_file.stem}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mon.write_image(output_path, output)
