#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Drawing.

This module implements drawing functionalities for images.
"""

from __future__ import annotations

__all__ = [
    "draw_bbox",
    "draw_heatmap",
    "draw_semantic",
    "draw_trajectory",
]

import cv2
import numpy as np

from mon.vision.dtype import image as I


def draw_bbox(
    image     : np.ndarray,
    bbox      : np.ndarray | list,
    label     : int | str    = None,
    color     : list[int]    = [255, 255, 255],
    thickness : int          = 1,
    line_type : int          = cv2.LINE_8,
    shift     : int          = 0,
    font_face : int          = cv2.FONT_HERSHEY_DUPLEX,
    font_scale: float        = 0.8,
    fill      : bool | float = False
) -> np.ndarray:
    """Draws a bounding box on an image.

    Args:
        image: Image as numpy.ndarray in [H, W, C] format, range [0, 255].
        bbox: Bounding box in XYXY format.
        label: Label for the bounding box. Default is ``None``.
        color: Color of the bounding box. Default is [255, 255, 255].
        thickness: Thickness of the rectangle borderline in px. Default is ``1``.
        line_type: Type of line (e.g., ``cv2.LINE_8``). Default is ``cv2.LINE_8``.
        shift: Fractional bits in point coordinates. Default is ``0``.
        font_face: Font of label text. Default is ``cv2.FONT_HERSHEY_DUPLEX``.
        font_scale: Scale of label text. Default is ``0.8``.
        fill: Fill inside with transparency (0.0-1.0, True=0.5). Default is ``False``.
        
    Returns:
        Image with drawn bounding box.
    """
    drawing = image.copy()
    color   = color or [255, 255, 255]
    white   = [255, 255, 255]
    pt1     = (int(bbox[0]), int(bbox[1]))
    pt2     = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(drawing, pt1, pt2, color, thickness, line_type, shift)

    if label not in [None, "None", ""]:
        label  = f"{label}"
        offset = int(thickness / 2)
        text_size, baseline = cv2.getTextSize(label, font_face, font_scale, 1)
        cv2.rectangle(
            img       = drawing,  # Changed from 'image' to 'drawing' for consistency
            pt1       = (pt1[0] - offset, pt1[1] - text_size[1] - offset),
            pt2       = (pt1[0] + text_size[0], pt1[1]),
            color     = color,
            thickness = cv2.FILLED
        )
        text_org = (pt1[0] - offset, pt1[1] - offset)
        cv2.putText(drawing, label, text_org, font_face, font_scale, white, 1)

    if fill is True or fill > 0.0:
        alpha   = 0.5 if fill is True else fill
        overlay = drawing.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, alpha, drawing, 1 - alpha, 0, drawing)

    return drawing


def draw_heatmap(
    image     : np.ndarray,
    heatmap   : np.ndarray,
    color_map : int   = cv2.COLORMAP_JET,
    alpha     : float = 0.5,
    use_rgb   : bool  = False
) -> np.ndarray:
    """Overlays a heatmap on an image.

    Args:
        image: RGB/BGR image as numpy.ndarray in [H, W, C], range [0.0, 1.0].
        heatmap: Heatmap mask to overlay.
        color_map: Color map for heatmap. Default is ``cv2.COLORMAP_JET``.
        alpha: Transparency ratio (0.0-1.0) for blending. Default is ``0.5``.
        use_rgb: Convert heatmap to RGB if True. Default is ``False``.
    
    Returns:
        Image with heatmap overlay.
    
    Raises:
        ValueError: If image exceeds range [0.0, 1.0] or alpha is invalid.
    """
    
    if np.max(image) > 1:
        raise ValueError(f"[image] should be np.float32 in range [0.0, 1.0], but got {np.max(image)}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"[alpha] should be in range [0.0, 1.0], but got {alpha}")

    heatmap = I.convert_depth_to_color(heatmap, color_map, use_rgb)
    heatmap = np.float32(heatmap) / 255
    drawing = I.blend_images(image, heatmap, alpha)
    drawing = drawing / np.max(drawing)
    drawing = np.uint8(255 * drawing)
    return drawing


def draw_semantic(
    image      : np.ndarray,
    semantic   : np.ndarray,
    classlabels: "ClassLabels",
    alpha      : float = 0.5
) -> np.ndarray:
    """Overlays a semantic mask on an image.

    Args:
        image: RGB image as numpy.ndarray in [H, W, C], range [0, 255].
        semantic: Semantic mask as numpy.ndarray in [H, W, 1].
        classlabels: List of class labels.
        alpha: Transparency ratio (0.0-1.0) for blending. Default is ``0.5``.
    
    Returns:
        Image with semantic mask overlay.
    """
    color_map = I.convert_label_map_id_to_color(semantic, classlabels)
    drawing   = I.blend_images(image, color_map, alpha)
    drawing   = drawing.astype(np.uint8)
    return drawing
    

def draw_trajectory(
    image     : np.ndarray,
    trajectory: np.ndarray | list,
    color     : list[int] = [255, 255, 255],
    thickness : int       = 1,
    line_type : int       = cv2.LINE_8,
    point     : bool      = False,
    radius    : int       = 3
) -> np.ndarray:
    """Draws a trajectory path on an image.

    Args:
        image: RGB image as numpy.ndarray in [H, W, C], range [0, 255].
        trajectory: 2D points as array or list in [(x1, y1), ...] format.
        color: Color of the trajectory. Default is [255, 255, 255].
        thickness: Thickness of the path in px. Default is ``1``.
        line_type: Type of line (e.g., ``cv2.LINE_8``). Default is ``cv2.LINE_8``.
        point: Draw points along the path if True. Default is ``False``.
        radius: Radius of points in px. Default is ``3``.
    
    Returns:
        Image with drawn trajectory.
    
    Raises:
        TypeError: If trajectory format is invalid.
    """
    drawing = image.copy()

    if isinstance(trajectory, list):
        if not all(len(t) == 2 for t in trajectory):
            raise TypeError("[trajectory] must be a list of points in [(x1, y1), ...] format")
        trajectory = np.array(trajectory)
    trajectory = np.array(trajectory).reshape((-1, 1, 2)).astype(int)
    color      = color or [255, 255, 255]
    cv2.polylines(drawing, [trajectory], False, color, thickness, line_type)
    if point:
        for p in trajectory:
            cv2.circle(drawing, tuple(p[0]), radius, color, -1)  # Fixed syntax and type

    return drawing
