#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import matplotlib
import numpy as np

from mon import Path

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]
data_dir     = current_dir  # current_file.parents[1]
if data_dir.has_subdir("data"):
    data_dir = data_dir / "data"


# ----- Inputs -----
filename   = "51a"
image_file = data_dir / "depth" / f"{filename}_image_depth.jpg"
ref_file   = data_dir / "depth" / f"{filename}_ref_depth.jpg"


# ----- Read image and depth -----
image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE) if image_file.exists() else None
ref   = cv2.imread(str(ref_file),   cv2.IMREAD_GRAYSCALE) if ref_file.exists()   else None


# ----- Utils -----
def compute_depth_metrics(
    pred      : np.ndarray,
    target    : np.ndarray,
    valid_mask: np.ndarray = None,
    normalize : bool       = False
) -> dict:
    # Input validation
    if pred.shape != target.shape:
        raise ValueError("Predicted and target depth maps must have the same shape.")

    pred = pred.astype(np.float32)
    target = target.astype(np.float32)

    # Create valid mask if not provided
    if valid_mask is None:
        valid_mask = (target > 0) & (~np.isnan(pred)) & (~np.isnan(target))

    if not np.any(valid_mask):
        raise ValueError("No valid pixels found in the depth maps.")

    # Normalize predicted depth map to target's range
    if normalize:
        valid_target = target[valid_mask]
        target_min, target_max = np.min(valid_target), np.max(valid_target)

        # Avoid division by zero in normalization
        if target_max == target_min:
            raise ValueError("Target depth map has no range (min equals max).")

        # Min-max normalization of predicted depths
        valid_pred = pred[valid_mask]
        pred_min, pred_max = np.min(valid_pred), np.max(valid_pred)
        if pred_max != pred_min:  # Only normalize if predicted map has a range
            pred = target_min + (target_max - target_min) * (pred - pred_min) / (pred_max - pred_min)
        else:
            # If predicted map is constant, scale to target's mean or min
            pred = np.full_like(pred, target_min)

        # Update min/max of normalized predicted map
        valid_pred = pred[valid_mask]
        pred_min, pred_max = np.min(valid_pred), np.max(valid_pred)

    # Flatten arrays and apply mask
    pred_flat = pred[valid_mask]
    target_flat = target[valid_mask]
    n = pred_flat.size

    # Compute differences
    diff     = pred_flat - target_flat
    abs_diff = np.abs(diff)

    # Absolute Relative Error
    abs_rel = np.mean(abs_diff / target_flat)

    # Squared Relative Error
    sq_rel  = np.mean((diff ** 2) / target_flat)

    # RMSE
    rmse = np.sqrt(np.mean(diff ** 2))

    # RMSE log
    log_pred   = np.log(np.clip(pred_flat, 1e-10, None))  # Avoid log(0)
    log_target = np.log(np.clip(target_flat, 1e-10, None))
    rmse_log   = np.sqrt(np.mean((log_pred - log_target) ** 2))

    # MAE
    mae = np.mean(abs_diff)

    # Threshold accuracies
    thresh = np.maximum(pred_flat / target_flat, target_flat / pred_flat)
    delta1 = np.mean(thresh < 1.25)
    delta2 = np.mean(thresh < 1.25 ** 2)
    delta3 = np.mean(thresh < 1.25 ** 3)

    return {
        "abs_rel" : abs_rel,
        "sq_rel"  : sq_rel,
        "rmse"    : rmse,
        "rmse_log": rmse_log,
        "mae"     : mae,
        "delta1"  : delta1,
        "delta2"  : delta2,
        "delta3"  : delta3
    }


def analyze_depth_map_range(depth_map: np.ndarray, valid_mask: np.ndarray = None) -> dict:
    if depth_map.size == 0:
        raise ValueError("Depth map is empty.")

    # Create valid mask if not provided
    if valid_mask is None:
        valid_mask = (depth_map > 0) & (~np.isnan(depth_map))

    # Extract valid depths
    valid_depths = depth_map[valid_mask]

    if valid_depths.size == 0:
        raise ValueError("No valid pixels found in the depth map.")

    # Compute statistics
    return {
        'min'           : np.min(valid_depths),
        'max'           : np.max(valid_depths),
        'mean'          : np.mean(valid_depths),
        'std'           : np.std(valid_depths),
        'valid_pixels'  : np.sum(valid_mask),
        'invalid_pixels': np.sum(~valid_mask)
    }


# ----- Main -----
cmap    = matplotlib.colormaps.get_cmap("Spectral_r")
#image   = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
#ref     = cv2.resize(ref,   (256, 256), interpolation=cv2.INTER_NEAREST)
# image   = (cmap(image)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
# ref     = (cmap(ref)[:, :, :3]   * 255)[:, :, ::-1].astype(np.uint8)
# image   = torch.tensor(image, dtype=torch.float32) / 255.0
# ref     = torch.tensor(ref,   dtype=torch.float32) / 255.0
# metrics = eval_depth(image, ref)
metrics   = compute_depth_metrics(image, ref)


# ----- Print results -----
print(f"Metrics for {filename}:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
