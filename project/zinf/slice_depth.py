#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
data_dir     = current_dir  # current_file.parents[1]
if data_dir.has_subdir("data"):
    data_dir = data_dir / "data"


# ----- Inputs -----
filename   = "a0952-kme_172"
#filename   = "a1179-IMG_0025"
#filename   = "sice_13"
#filename   = "778"
image_file = data_dir    / "inr" / f"{filename}_image.jpg"
ref_file   = data_dir    / "inr" / f"{filename}_ref.jpg"
pred_file  = data_dir    / "inr" / f"{filename}_pred.jpg"
depth_file = data_dir    / "inr" / f"{filename}_depth.jpg"
output_dir = current_dir / "run" / "illu"
num_bins   = 5
background = 200


# ----- Read image and depth -----
image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)     if image_file.exists() else None
ref   = cv2.imread(str(ref_file),   cv2.IMREAD_COLOR)     if ref_file.exists()   else None
pred  = cv2.imread(str(pred_file),  cv2.IMREAD_COLOR)     if pred_file.exists()  else None
depth = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE) if depth_file.exists() else None
output_dir.mkdir(parents=True, exist_ok=True)


# ----- Utils -----
def preprocess_depth(depth, preserve_zeros=True):
    # Create a copy of the depth map
    equalized_depth = depth.copy()

    if preserve_zeros:
        # Mask for zero values
        zero_mask = (depth == 0)
        # Equalize non-zero values
        non_zero_depth = depth[~zero_mask]
        if non_zero_depth.size > 0:
            equalized_non_zero = cv2.equalizeHist(non_zero_depth.reshape(-1, 1))
            equalized_depth[~zero_mask] = equalized_non_zero.ravel()
    else:
        # Apply histogram equalization to entire depth map
        equalized_depth = cv2.equalizeHist(depth)

    return equalized_depth


def categorize_depth(depth, num_bins=10):
    min_, max_   = 0, 255
    bins         = np.linspace(min_, max_, num_bins + 1)
    depth_slices = np.digitize(depth, bins, right=True).astype(np.uint8)
    depth_slices = np.clip(depth_slices, 0, num_bins - 1)
    return depth_slices, bins


def create_image_slices(image, slices, num_bins=10):
    image_slices = []
    for i in range(num_bins):
        # Create mask for the current bin
        mask = (slices == i).astype(np.uint8)
        # Initialize output image (black background)
        # bin_img = np.zeros_like(image)
        bin_img = np.full_like(image, background)
        # Apply mask to each channel
        for c in range(3):  # BGR channels
            bin_img[:, :, c] = np.where(mask, image[:, :, c], bin_img[:, :, c])
        image_slices.append(bin_img)
    return image_slices


def visualize_bin_grid(image_slices, bins, num_bins=10):
    rows, cols = (num_bins + 4) // 5, min(num_bins, 5)
    fig, axes  = plt.subplots(rows, cols, figsize=(15, 6))
    axes       = axes.flatten() if num_bins > 1 else [axes]

    for i in range(num_bins):
        # Convert BGR to RGB for Matplotlib
        bin_img_rgb = cv2.cvtColor(image_slices[i], cv2.COLOR_BGR2RGB)
        axes[i].imshow(bin_img_rgb)
        axes[i].set_title(f'Bin {i} ({bins[i]:.1f}-{bins[i+1]:.1f})')
        axes[i].axis('off')

    # Hide unused subplots if any
    for i in range(num_bins, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(str(output_dir / "depth_bin_grid.jpg"))
    plt.show()


# ----- Main -----
depth              = preprocess_depth(depth)
depth_slices, bins = categorize_depth(depth, num_bins)

image_slices = create_image_slices(image, depth_slices, num_bins)
for i, img in enumerate(image_slices):
    cv2.imwrite(str(output_dir / f"{filename}_image_{i}.jpg"), img)

if ref is not None:
    ref_slices = create_image_slices(ref,   depth_slices, num_bins)
    for i, img in enumerate(ref_slices):
        cv2.imwrite(str(output_dir / f"{filename}_ref_{i}.jpg"), img)

if pred is not None:
    pred_slices = create_image_slices(pred, depth_slices, num_bins)
    for i, img in enumerate(pred_slices):
        cv2.imwrite(str(output_dir / f"{filename}_pred_{i}.jpg"), img)
