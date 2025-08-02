#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
data_dir     = current_dir  # current_file.parents[1]
if data_dir.has_subdir("data"):
    data_dir = data_dir / "data"


# ----- Inputs -----
filename   = "a0952-kme_172"
image_fle  = data_dir / "illu" / f"{filename}_image.jpg"
ref_file   = data_dir / "illu" / f"{filename}_ref.jpg"
depth_file = data_dir / "illu" / f"{filename}_depth.jpg"
illu_file  = data_dir / "illu" / f"{filename}_illu_res.jpg"
illu2_file = data_dir / "illu" / f"{filename}_illu_res2.jpg"
output_dir = data_dir / "illu"
num_bins   = 5
background = 0
# cmap       = matplotlib.colormaps.get_cmap("bone")
cmap       = matplotlib.colormaps.get_cmap("magma")


# ----- Read image and depth -----
image = cv2.imread(str(image_fle),  cv2.IMREAD_COLOR)     if image_fle.exists()  else None
ref   = cv2.imread(str(ref_file),   cv2.IMREAD_COLOR)     if ref_file.exists()   else None
depth = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE) if depth_file.exists() else None
illu  = cv2.imread(str(illu_file),  cv2.IMREAD_GRAYSCALE) if illu_file.exists()  else None
illu2 = cv2.imread(str(illu2_file), cv2.IMREAD_GRAYSCALE) if illu2_file.exists() else None
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
        bin_img[:, :] = np.where(mask, image[:, :], bin_img[:, :])
        image_slices.append(bin_img)
    return image_slices


def preprocess_illu(illu, min_value=0, max_value=100):
    low, high = np.percentile(illu, (min_value, max_value))
    if high > low:
        normalized = np.clip((illu - low) / (high - low), 0.0, 1.0)
    else:
        normalized = illu / 255.0
    return normalized


def plot_illu(image, save_path):
    plt.figure(figsize=(4, 3), facecolor="none")
    plt.imshow(image[:, :, ::-1])  # Convert BGR to RGB
    plt.axis("off")
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=plt.gca())
    cbar.set_label("Adjustment Strength")
    plt.savefig(str(save_path), bbox_inches="tight")


def plot_3d_gaussian_map(image, save_path, sigma=5.0, stride=24):
    # image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)

    # Convert to float and determine dynamic range
    image = image.astype(float)
    min_val, max_val = image.min(), image.max()
    if min_val == max_val:
        raise ValueError("Input image has constant values; min and max are equal.")

    # Scale to preserve the original min-max range (no additional normalization)
    image_scaled = image  # Use original values directly

    # Apply Gaussian filter
    smoothed_image = gaussian_filter(image_scaled, sigma=sigma)

    # Create grid for 3D plot
    x    = np.arange(0, smoothed_image.shape[1], stride)
    y    = np.arange(0, smoothed_image.shape[0], stride)
    X, Y = np.meshgrid(x, y)
    Z    = smoothed_image[np.ix_(y, x)]  # Sampled Z values based on stride

    # Create 3D plot
    fig  = plt.figure(figsize=(3, 3), facecolor="none")
    ax   = fig.add_subplot(111, projection="3d")
    norm = Normalize(vmin=0, vmax=1)  # Normalize color bar to 0-1
    surf = ax.plot_surface(
        X, Y, Z,
        cmap        = "jet",
        facecolors  = plt.cm.jet(norm((Z - min_val) / (max_val - min_val))),
        rstride     = 1,
        cstride     = 1,
        linewidth   = 0.5,
        edgecolor   = "k",
        antialiased = True,
        shade       = True
    )
    # plt.colorbar(surf, norm=norm)  # Color bar with 0-1 range

    # ax.view_init(elev=30, azim=45)  # Adjust view angle for better perspective
    # ax.view_init(elev=90, azim=90)   # Top-down view
    ax.view_init(elev=20, azim=30)
    fig.tight_layout()

    # Set labels and limits
    ax.patch.set_facecolor("none")
    # ax.patch.set_alpha(0)
    # ax.set_xlabel("X")
    #ax.set_ylabel("Y")
    #ax.set_zlabel("Z")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.set_zlim(0, 1)

    plt.savefig(str(save_path), bbox_inches="tight")


# ----- Main -----
image_hsv    = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image_v      = image_hsv[:, :, 2]
ref_hsv      = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
ref_v        = ref_hsv[:, :, 2]
illu3_g      = np.abs(ref_v - image_v)
illu3_g      = preprocess_illu(illu3_g, 20, 95)

depth        = preprocess_depth(depth)
slices, bins = categorize_depth(depth, num_bins)

illu_g  = preprocess_illu(illu, 0, 80)
illu_c  = (cmap(illu_g)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
illu2_g = preprocess_illu(illu2, 15, 100)
illu2_c = (cmap(illu2_g)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

#
illu2_slices   = create_image_slices(illu2, slices, num_bins)
illu2_c_slices = []
for i, img in enumerate(illu2_slices):
    img = preprocess_illu(img, 0, 99.5)
    img = (cmap(img)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    illu2_c_slices.append(img)
    cv2.imwrite(str(output_dir / f"{filename}_illu_res2_{i}.png"), img)

# Visualize merged image with colorbar
h, w     = slices.shape
merged_c = np.full((h, w, 3), background, dtype=np.uint8)
for i in range(min(num_bins, len(slices))):
    mask     = (slices == i).astype(np.uint8)[:, :, None]
    merged_c = np.where(mask, illu2_c_slices[i], merged_c)
merged_g   = cv2.cvtColor(merged_c, cv2.COLOR_RGB2GRAY)
merged_g   = preprocess_illu(merged_g, 0, 100)


# ----- Save image -----
plot_illu(illu_c,   str(output_dir / f"{filename}_illu_res_color.png"))
plot_illu(merged_c, str(output_dir / f"{filename}_illu_res2_color.png"))

plot_3d_gaussian_map(illu_g,   str(output_dir / f"{filename}_illu_res_gauss_color.png"),  sigma=3.0)
plot_3d_gaussian_map(merged_g, str(output_dir / f"{filename}_illu_res2_gauss_color.png"), sigma=3.0)
plot_3d_gaussian_map(illu3_g,  str(output_dir / f"{filename}_illu_res3_gauss_color.png"), sigma=3.0)
# plt.show()
