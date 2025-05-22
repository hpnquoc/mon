#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Transform normal images with bboxes to fisheye images."""

from __future__ import annotations

import cv2
import numpy as np

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


'''
def get_fish_xn_yn(source_x, source_y, radius, distortion, max_radius):
    if radius == 0:
        return source_x, source_y
    factor = (1 + distortion * (radius ** 2)) / (1 + distortion * (max_radius ** 2))
    return source_x * factor, source_y * factor


def transform_image(image, distortion, padding_factor=1.1):
    w, h = image.shape[0], image.shape[1]
    if len(image.shape) == 2:
        bw_channel = np.copy(image)
        image      = np.dstack((image, bw_channel, bw_channel))
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = np.dstack((image, np.full((w, h), 255)))

    # Calculate padded dimensions
    pad_w, pad_h = int(w * padding_factor), int(h * padding_factor)
    dstimg       = np.zeros((pad_w, pad_h, 4), dtype=np.uint8)
    max_radius   = np.sqrt(1 + (w / h) ** 2)  # Based on original image dimensions

    # Center offsets for padded image
    offset_x = (pad_w - w) / 2
    offset_y = (pad_h - h) / 2

    for x in range(pad_w):
        for y in range(pad_h):
            # Normalize coordinates relative to padded image, adjusted to original image center
            xnd = float((2 * (x - offset_x) - w) / w)
            ynd = float((2 * (y - offset_y) - h) / h)
            rd  = np.sqrt(xnd ** 2 + ynd ** 2)
            if rd > max_radius:
                continue  # Outside circular boundary
            xdu, ydu = get_fish_xn_yn(xnd, ynd, rd, distortion, max_radius)
            xu, yu   = int(((xdu + 1) * w) / 2), int(((ydu + 1) * h) / 2)
            if 0 <= xu < w and 0 <= yu < h:
                dstimg[x][y] = image[xu][yu]
            else:
                dstimg[x][y][3] = 0  # Transparent for out-of-bounds

    # Apply circular mask in padded image
    for x in range(pad_w):
        for y in range(pad_h):
            xnd = float((2 * (x - offset_x) - w) / w)
            ynd = float((2 * (y - offset_y) - h) / h)
            if np.sqrt(xnd ** 2 + ynd ** 2) > max_radius:
                dstimg[x][y][3] = 0  # Transparent outside circle

    return dstimg.astype(np.uint8)


def transform_bbox(bboxes, image_width, image_height, distortion, padding_factor=1.1):
    transformed_bboxes = []
    w, h         = float(image_width), float(image_height)
    pad_w, pad_h = w * padding_factor, h * padding_factor
    max_radius   = np.sqrt(1 + (h / w) ** 2)
    offset_x     = (pad_w - w) / 2
    offset_y     = (pad_h - h) / 2

    for bbox in bboxes:
        class_id, x_center_norm, y_center_norm, width_norm, height_norm = bbox
        # Convert normalized to pixel coordinates
        x_center = x_center_norm * w
        y_center = y_center_norm * h
        width    = width_norm    * w
        height   = height_norm   * h

        # Define the four corners of the bounding box
        corners = [
            (x_center - width / 2, y_center - height / 2),  # Top-left
            (x_center + width / 2, y_center - height / 2),  # Top-right
            (x_center - width / 2, y_center + height / 2),  # Bottom-left
            (x_center + width / 2, y_center + height / 2)   # Bottom-right
        ]

        # Transform each corner to distorted coordinates
        distorted_corners = []
        for x_s, y_s in corners:
            x_sn = (2 * x_s / w) - 1
            y_sn = (2 * y_s / h) - 1
            r_sn = np.sqrt(x_sn ** 2 + y_sn ** 2)
            if r_sn == 0:
                x_d, y_d = x_s, y_s
            else:
                factor = (1 + distortion * (r_sn ** 2)) / (1 + distortion * (max_radius ** 2))
                x_dn = x_sn / factor
                y_dn = y_sn / factor
                # Map to padded image coordinates
                x_d  = ((x_dn + 1) * w) / 2 + offset_x
                y_d  = ((y_dn + 1) * h) / 2 + offset_y
            distorted_corners.append((x_d, y_d))

        if not distorted_corners:
            continue  # Skip if no valid corners

        # Find enclosing rectangle
        x_coords, y_coords = zip(*distorted_corners)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        new_width    = x_max - x_min
        new_height   = y_max - y_min
        new_x_center = (x_min + x_max) / 2
        new_y_center = (y_min + y_max) / 2

        # Normalize to padded image dimensions
        new_x_center_norm = new_x_center / pad_w
        new_y_center_norm = new_y_center / pad_h
        new_width_norm    = new_width    / pad_w
        new_height_norm   = new_height   / pad_h

        # Ensure coordinates are within [0, 1]
        if (0 <= new_x_center_norm <= 1 and 0 <= new_y_center_norm <= 1 and
                new_width_norm > 0 and new_height_norm > 0):
            transformed_bboxes.append([
                class_id, new_x_center_norm, new_y_center_norm, new_width_norm, new_height_norm
            ])

    return transformed_bboxes
'''


def convert_fisheye(
    data        : str,
    distortion  : float = 1.0,
    area_thres  : int   = 32,
    aspect_thres: float = 0.1
):
    image_dir         = current_dir / "data" / "fisheye8k" / "extra" / data / "image"
    label_dir         = current_dir / "data" / "fisheye8k" / "extra" / data / "label"
    fisheye_image_dir = current_dir / "data" / "fisheye8k" / "extra" / data / "image_fisheye"
    fisheye_label_dir = current_dir / "data" / "fisheye8k" / "extra" / data / "label_fisheye"
    
    assert mon.Path(image_dir).is_dir()
    assert mon.Path(label_dir).is_dir()

    transform = mon.FisheyeTransform(distortion, area_thres, aspect_thres, p=1)

    image_files = sorted([f for f in list(image_dir.rglob("*")) if f.is_image_file()])
    with mon.create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence    = enumerate(image_files),
            total       = len(image_files),
            description = f"[bright_yellow] Processing"
        ):
            # Input
            image = cv2.imread(str(image_file))
            image = mon.resize(image, 1920, side="long", interpolation="bicubic")

            label_file = label_dir / f"{image_file.stem}.txt"
            if not label_file.is_txt_file(exist=True):
                continue

            with open(label_file, "r") as f:
                l = f.readlines()
            l = [l_.strip().split(" ") for l_ in l]
            l = [l_ for l_ in l if len(l_) >= 5]
            if len(l) == 0:
                continue
            t = np.array([list(map(float, l_[0:5])) for l_ in l])
            b = t.copy()
            b[:, 0:4] = t[:, 1:5]
            b[:, 4]   = t[:, 0]
            assert len(l) == len(b)

            # Split image and bounding boxes
            sis, sbs = mon.split_image_and_bboxes(image, b, 2)

            # Transform each sub-image and bounding box
            for j, (si, sb) in enumerate(zip(sis, sbs)):
                transformed = transform(image=si, bboxes=sb)
                si_t = transformed["image"]
                sb_t = transformed["bboxes"]

                si_t_file = fisheye_image_dir / f"{image_file.stem}_{j}.jpg"
                si_t_file.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(si_t_file), si_t)

                sb_t_file = fisheye_label_dir / f"{image_file.stem}_{j}.txt"
                sb_t_file.parent.mkdir(parents=True, exist_ok=True)
                with open(sb_t_file, "w") as f:
                    for b_ in sb_t:
                        f.write("{} {} {} {} {}\n".format(int(b_[4]), b_[0], b_[1], b_[2], b_[3]))


if __name__ == "__main__":
    convert_fisheye("visdrone", distortion=0.5, area_thres=10, aspect_thres=0.3)
