#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Data Type.

"""

from __future__ import annotations

__all__ = [
    "BoundaryAwarePrior",
    "BrightnessAttentionMap",
    "ImageLocalMean",
    "ImageLocalStdDev",
    "ImageLocalVariance",
    "add_noise",
    "add_weighted",
    "adjust_gamma",
    "atmospheric_prior",
    "blend_images",
    "blur_spot_prior",
    "boundary_aware_prior",
    "bright_channel_prior",
    "bright_spot_prior",
    "brightness_attention_map",
    "dark_channel_prior",
    "dark_channel_prior_02",
    "denormalize_image",
    "denormalize_image_mean_std",
    "depth_map_to_color",
    "get_image_center",
    "get_image_center4",
    "get_image_channel",
    "get_image_num_channels",
    "get_image_shape",
    "get_image_size",
    "image_local_mean",
    "image_local_stddev",
    "image_local_variance",
    "is_channel_first_image",
    "is_channel_last_image",
    "is_color_image",
    "is_gray_image",
    "is_image",
    "is_integer_image",
    "is_normalized_image",
    "label_map_color_to_id",
    "label_map_id_to_color",
    "label_map_id_to_one_hot",
    "label_map_id_to_train_id",
    "label_map_one_hot_to_id",
    "normalize_image",
    "normalize_image_by_range",
    "normalize_image_mean_std",
    "read_image",
    "read_image_shape",
    "to_2d_image",
    "to_3d_image",
    "to_4d_image",
    "to_channel_first_image",
    "to_channel_last_image",
    "to_image_nparray",
    "to_image_tensor",
    "to_list_of_3d_image",
    "write_image",
    "write_image_cv",
    "write_image_torch",
    "write_images_cv",
    "write_images_torch",
]

import copy
import functools
import math
import multiprocessing
from typing import Any, Literal, Sequence

import cv2
import joblib
import kornia
import numpy as np
import rawpy
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torchvision.transforms import functional as TF

from mon import core


# region Adjustment

def adjust_gamma(
    image: torch.Tensor | np.ndarray,
    gamma: float = 1.0,
    gain : float = 1.0
) -> torch.Tensor | np.ndarray:
    """Adjust gamma value in the image. Also known as Power Law Transform.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        gamma: Non-negative real number, same as `gamma` in the equation.
            - :obj:`gamma` larger than ``1`` makes the shadows darker, while
            - :obj:`gamma` smaller than ``1`` makes dark regions lighter.
        gain: The constant multiplier.
        
    Returns:
        A gamma-corrected image.
    """
    if isinstance(image, torch.Tensor):
        return TF.adjust_gamma(img=image, gamma=gamma, gain=gain)
    elif isinstance(image, np.ndarray):
        # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values.
        inv_gamma = 1.0 / gamma
        table     = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
        table.astype("uint8")
        # Apply gamma correction using the lookup table
        return cv2.LUT(image, table)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    
    
def add_noise(
    image      : torch.Tensor,
    noise_level: int = 25,
    noise_type : Literal["gaussian", "poisson"] = "gaussian"
) -> torch.Tensor:
    """Add noise to an image.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        noise_level: The noise level.
        noise_type: The type of noise to add. One of:
            - ``'gaussian'``
            - ``'poisson'``
            Default: ``"gaussian"``.
        
    Returns:
        A noisy image.
    """
    if noise_type == "gaussian":
        noisy = image + torch.normal(0, noise_level / 255, image.shape)
        noisy = torch.clamp(noisy, 0, 1)
    elif noise_type == "poisson":
        noisy = torch.poisson(noise_level * image) / noise_level
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    return noisy

# endregion


# region Assertion

def is_channel_first_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is in channel-first format. We assume
    that if the first dimension is the smallest.
    """
    if not 3 <= image.ndim <= 5:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``3`` and ``5``, but got {image.ndim}.")
    if image.ndim == 5:
        _, _, s2, s3, s4 = list(image.shape)
        if (s2 < s3) and (s2 < s4):
            return True
        elif (s4 < s2) and (s4 < s3):
            return False
    elif image.ndim == 4:
        _, s1, s2, s3 = list(image.shape)
        if (s1 < s2) and (s1 < s3):
            return True
        elif (s3 < s1) and (s3 < s2):
            return False
    elif image.ndim == 3:
        s0, s1, s2 = list(image.shape)
        if (s0 < s1) and (s0 < s2):
            return True
        elif (s2 < s0) and (s2 < s1):
            return False
    return False


def is_channel_last_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is in channel-first format."""
    return not is_channel_first_image(image)


def is_color_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is a color image. It is assumed that the
    image has ``3`` or ``4`` channels.
    """
    if get_image_num_channels(image) in [3, 4]:
        return True
    return False


def is_gray_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is a gray image. It is assumed that the
    image has one channel.
    """
    if get_image_num_channels(image) in [1] or len(image.shape) == 2:
        return True
    return False


def is_color_or_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is a color or gray image."""
    return is_color_image(image) or is_gray_image(image)


def is_image(image: torch.Tensor, bits: int = 8) -> bool:
    """Check whether an image tensor is ranged properly ``[0.0, 1.0]`` for
    :obj:`float` or ``[0, 2 ** bits]`` for :obj:`int`.

    Args:
        image: Image tensor to evaluate.
        bits: The image bits. The default checks if given :obj:`int` input
            image is an 8-bit image `[0-255]` or not.

    Raises:
        TypeException: if all the input tensor has not
        1) a shape `[3, H, W]`,
        2) ``[0.0, 1.0]`` for :obj:`float` or ``[0, 255]`` for :obj:`int`,
        3) and raises is ``True``.
    
    Example:
        >>> img = torch.rand(2, 3, 4, 4)
        >>> is_image(img)
        True
    """
    if not isinstance(image, torch.Tensor | np.ndarray):
        return False
    res = is_color_or_image(image)
    if not res:
        return False
    '''
    if (
        input.dtype in [torch.float16, torch.float32, torch.float64]
        and (input.min() < 0.0 or input.max() > 1.0)
    ):
        return False
    elif input.min() < 0 or input.max() > 2 ** bits - 1:
        return False
    '''
    return True


def is_integer_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is integer-encoded."""
    c = get_image_num_channels(image)
    if c == 1:
        return True
    return False


def is_normalized_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is normalized."""
    if isinstance(image, torch.Tensor):
        return abs(torch.max(image)) <= 1.0
    elif isinstance(image, np.ndarray):
        return abs(np.amax(image)) <= 1.0
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")

# endregion


# region Accessing

def get_image_channel(
    image   : torch.Tensor | np.ndarray,
    index   : int | Sequence[int],
    keep_dim: bool = True,
) -> torch.Tensor | np.ndarray:
    """Return a channel of an image.

    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        index: The channel's index.
        keep_dim: If ``True``, keep the dimensions of the return output.
            Default: ``True``.
    """
    if isinstance(index, int):
        i1 = index
        i2 = None if i1 < 0 else i1 + 1
    elif isinstance(index, (list, tuple)):
        i1 = index[0]
        i2 = index[1]
    else:
        raise TypeError
    
    if is_channel_first_image(image):
        if image.ndim == 5:
            if keep_dim:
                return image[:, :, i1:i2, :, :] if i2 else image[:, :, i1:, :, :]
            else:
                return image[:, :, i1, :, :] if i2 else image[:, :, i1, :, :]
        elif image.ndim == 4:
            if keep_dim:
                return image[:, i1:i2, :, :] if i2 else image[:, i1:, :, :]
            else:
                return image[:, i1, :, :] if i2  else image[:, i1, :, :]
        elif image.ndim == 3:
            if keep_dim:
                return image[i1:i2, :, :] if i2 else image[i1:, :, :]
            else:
                return image[i1, :, :] if i2 else image[i1, :, :]
        else:
            raise ValueError
    else:
        if image.ndim == 5:
            if keep_dim:
                return image[:, :, :, :, i1:i2] if i2 else image[:, :, :, :, i1:]
            else:
                return image[:, :, :, :, i1] if i2 else image[:, :, :, :, i1]
        elif image.ndim == 4:
            if keep_dim:
                return image[:, :, :, i1:i2] if i2 else image[:, :, :, i1:]
            else:
                return image[:, :, :, i1] if i2 else image[:, :, :, i1]
        elif image.ndim == 3:
            if keep_dim:
                return image[:, :, i1:i2] if i2 else image[:, :, i1:]
            else:
                return image[:, :, i1] if i2 else image[:, :, i1]
        else:
            raise ValueError
    

def get_image_num_channels(image: torch.Tensor | np.ndarray) -> int:
    """Return the number of channels of an image.

    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
    """
    if image.ndim == 4:
        if is_channel_first_image(image):
            _, c, h, w = list(image.shape)
        else:
            _, h, w, c = list(image.shape)
    elif image.ndim == 3:
        if is_channel_first_image(image):
            c, h, w = list(image.shape)
        else:
            h, w, c = list(image.shape)
    elif image.ndim == 2:
        c = 1
    else:
        # error_console.log(
        #     f":obj:`image`'s number of dimensions must be between ``2`` and ``4``, "
        #     f"but got {input.ndim}."
        # )
        c = 0
    return c


def get_image_center(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the center of a given image specified as `(x=h/2, y=w/2)`.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
    """
    h, w = get_image_size(image)
    if isinstance(image, torch.Tensor):
        return torch.Tensor([h / 2, w / 2])
    elif isinstance(image, np.ndarray):
        return np.array([h / 2, w / 2])
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def get_image_center4(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the center of a given image specified as
    `(x=h/2, y=w/2, x=h/2, y=w/2)`.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
    """
    h, w = get_image_size(image)
    if isinstance(image, torch.Tensor):
        return torch.Tensor([h / 2, w / 2, h / 2, w / 2])
    elif isinstance(image, np.ndarray):
        return np.array([h / 2, w / 2, h / 2, w / 2])
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def get_image_shape(image: torch.Tensor | np.ndarray) -> list[int]:
    """Return height, width, and channel value of an image.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
    """
    if is_channel_first_image(image):
        return [image.shape[-2], image.shape[-1], image.shape[-3]]
    else:
        return [image.shape[-3], image.shape[-2], image.shape[-1]]


def get_image_size(
    input  : torch.Tensor | np.ndarray | int | Sequence[int] | str | core.Path,
    divisor: int = None,
) -> tuple[int, int]:
    """Return height and width value of an image in the ``[H, W]`` format.
    
    Args:
        input: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
            - A size of an image, windows, or kernels, etc.
        divisor: The divisor. Default: ``None``.
        
    Returns:
        A size in ``[H, W]`` format.
    """
    # Get raw size
    if isinstance(input, list | tuple):
        if len(input) == 3:
            if input[0] >= input[2]:
                size = input[0:2]
            else:
                size = input[1:3]
        elif len(input) == 2:
            size = input
        elif len(input) == 1:
            size = (input[0], input[0])
        else:
            raise ValueError(f"`input` must be a `list` of length in range "
                             f"``[1, 3]``, but got {input}.")
    elif isinstance(input, int | float):
        size = (input, input)
    elif isinstance(input, (torch.Tensor, np.ndarray)):
        if is_channel_first_image(input):
            size = (input.shape[-2], input.shape[-1])
        else:
            size = (input.shape[-3], input.shape[-2])
    elif isinstance(input, str | core.Path):
        size = read_image_shape(input)[0:2]
    else:
        raise TypeError(f"`input` must be a `torch.Tensor`, `numpy.ndarray`, "
                        f"or a `list` of `int`, but got {type(input)}.")
    
    # Divisible
    if divisor is not None:
        h, w  = size
        new_h = int(math.ceil(h / divisor) * divisor)
        new_w = int(math.ceil(w / divisor) * divisor)
        size  = (new_h, new_w)
    return size

# endregion


# region Combination

def add_weighted(
    image1: torch.Tensor | np.ndarray,
    image2: torch.Tensor | np.ndarray,
    alpha : float,
    beta  : float,
    gamma : float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Calculate the weighted sum of two image tensors as follows:
        output = image1 * alpha + image2 * beta + gamma

    Args:
        image1: The first image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        image2: The same as :obj:`image1`.
        alpha: The weight of the :obj:`image1` elements.
        beta: The weight of the :obj:`image2` elements.
        gamma: A scalar added to each sum. Default: ``0.0``.

    Returns:
        A weighted image.
    """
    if image1.shape != image2.shape:
        raise ValueError(f"`image1` and `image2` must have the same shape, "
                         f"but got {image1.shape} != {image2.shape}.")
    if type(image1) is not type(image2):
        raise ValueError(f"`image1` and `image2` must have the same type, "
                         f"but got {type(image1)} != {type(image2)}.")
    
    output = image1 * alpha + image2 * beta + gamma
    bound  = 1.0 if is_normalized_image(image1) else 255.0
    
    if isinstance(output, torch.Tensor):
        output = output.clamp(0, bound).to(image1.dtype)
    elif isinstance(output, np.ndarray):
        output = np.clip(output, 0, bound).astype(image1.dtype)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(input)}.")
    return output


def blend_images(
    image1: torch.Tensor | np.ndarray,
    image2: torch.Tensor | np.ndarray,
    alpha : float,
    gamma : float = 0.0
) -> torch.Tensor | np.ndarray:
    """Blend 2 images together using the formula:
        output = :obj:`image1` * alpha + :obj:`image2` * beta + gamma

    Args:
        image1: A source image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        image2: An overlay image that we want to blend on top of :obj:`image1`.
        alpha: An alpha transparency of the overlay.
        gamma: A scalar added to each sum. Default: ``0.0``.
    
    Returns:
        A blended image.
    """
    return add_weighted(
        image1 = image2,
        image2 = image1,
        alpha  = alpha,
        beta   = 1.0 - alpha,
        gamma  = gamma,
    )

# endregion


# region Conversion

def depth_map_to_color(
    depth_map: np.ndarray,
    color_map: int = cv2.COLORMAP_JET,
    use_rgb  : bool = False,
) -> np.ndarray:
    """Convert depth map to color-coded images.
    
    Args:
        depth_map: A depth map of type :obj:`numpy.ndarray` in ``[H, W, 1]``
            format.
        color_map: A color map for the depth map. Default: ``cv2.COLORMAP_JET``.
        use_rgb: If ``True``, convert the heatmap to RGB format.
            Default: ``False``.
    """
    if is_normalized_image(depth_map):
        depth_map = np.uint8(255 * depth_map)
    depth_map = cv2.applyColorMap(np.uint8(255 * depth_map), color_map)
    if use_rgb:
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)
    return depth_map
    

def label_map_id_to_train_id(
    label_map  : np.ndarray,
    classlabels: "ClassLabels",
) -> np.ndarray:
    """Convert label map from IDs to train IDs.
    
    Args:
        label_map: An IDs label map of type :obj:`numpy.ndarray` in
            ``[H, W, C]`` format.
        classlabels: A list of class-labels.
    """
    id2train_id = classlabels.id2train_id
    h, w        = get_image_size(label_map)
    label_ids   = np.zeros((h, w), dtype=np.uint8)
    label_map   = to_2d_image(label_map)
    for id, train_id in id2train_id.items():
        label_ids[label_map == id] = train_id
    label_ids   = np.expand_dims(label_ids, axis=-1)
    return label_ids
 

def label_map_id_to_color(
    label_map  : np.ndarray,
    classlabels: "ClassLabels",
) -> np.ndarray:
    """Convert label map from label IDs to color-coded.
    
    Args:
        label_map: An IDs label map of type :obj:`numpy.ndarray` in
            ``[H, W, C]`` format.
        classlabels: A list of class-labels, each has predefined color.
    """
    id2color  = classlabels.id2color
    h, w      = get_image_size(label_map)
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    label_map = to_2d_image(label_map)
    for id, color in id2color.items():
        color_map[label_map == id] = color
    return color_map


def label_map_color_to_id(
    label_map  : np.ndarray,
    classlabels: "ClassLabels",
) -> np.ndarray:
    """Convert label map from color-coded to label IDS.

    Args:
        label_map: A color-coded label map of type :obj:`numpy.ndarray` in
            ``[H, W, C]`` format.
        classlabels: A list of class-labels, each has predefined color.
    """
    id2color  = classlabels.id2color
    h, w      = get_image_size(label_map)
    label_ids = np.zeros((h, w), dtype=np.uint8)
    for id, color in id2color.items():
        label_ids[np.all(label_map == color, axis=-1)] = id
    label_ids = np.expand_dims(label_ids, axis=-1)
    return label_ids


def label_map_id_to_one_hot(
    label_map  : torch.Tensor | np.ndarray,
    num_classes: int           = None,
    classlabels: "ClassLabels" = None,
) ->torch.Tensor | np.ndarray:
    """Convert label map from label IDs to one-hot encoded.
    
    Args:
        label_map: An IDs label map of type:
            - :obj:`torch.Tensor` in ``[B, 1, H, W]`` format.
            - :obj:`numpy.ndarray` in ``[H, W, 1]`` format.
        num_classes: The number of classes in the label map.
        classlabels: A list of class-labels.
    """
    if num_classes is None and classlabels is None:
        raise ValueError("Either `num_classes` or `classlabels` must be "
                         "provided.")
    
    num_classes = num_classes or classlabels.num_trainable_classes
    if isinstance(label_map, torch.Tensor):
        label_map = to_3d_image(label_map).long()
        one_hot   = F.one_hot(label_map, num_classes)
        one_hot   = to_channel_first_image(one_hot).contiguous()
    elif isinstance(label_map, np.ndarray):
        label_map = to_2d_image(label_map)
        one_hot   = np.eye(num_classes)[label_map]
    else:
        raise TypeError(f"`label_map` must be a `numpy.ndarray` or "
                        f"`torch.Tensor`, but got {type(label_map)}.")
    return one_hot


def label_map_one_hot_to_id(
    label_map: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """Convert label map from one-hot encoded to label IDs.
    
    Args:
        label_map: A one-hot encoded label map of type:
            - :obj:`torch.Tensor` in ``[B, num_classes, H, W]`` format.
            - :obj:`numpy.ndarray` in ``[H, W, num_classes]`` format.
    """
    if isinstance(label_map, torch.Tensor):
        label_map = torch.argmax(label_map, dim=-1, keepdim=True)
    elif isinstance(label_map, np.ndarray):
        label_map = np.argmax(label_map, axis=-1, keepdims=True)
    else:
        raise TypeError(f"`label_map` must be a `numpy.ndarray` or "
                        f"`torch.Tensor`, but got {type(label_map)}.")
    return label_map


def to_2d_image(image: Any) -> torch.Tensor | np.ndarray:
    """Convert a 3D or 4D image to a 2D."""
    if not 3 <= image.ndim <= 4:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``3`` and ``4``, but got {image.ndim}.")
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:  # 1HW -> HW
            image = image.squeeze(dim=0)
        elif image.ndim == 4 and image.shape[0] == 1 and image.shape[1] == 1:  # 11HW -> HW
            image = image.squeeze(dim=0)
            image = image.squeeze(dim=0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:  # HW1 -> HW
            image = np.squeeze(image, axis=-1)
        elif image.ndim == 4 and image.shape[0] == 1 and image.shape[3] == 1:  # 1HW1 -> HW
            image = np.squeeze(image, axis=0)
            image = np.squeeze(image, axis=-1)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    return image


def to_3d_image(image: Any) -> torch.Tensor | np.ndarray:
    """Convert a 2D or 4D image to a 3D."""
    if not 2 <= image.ndim <= 4:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``2`` and ``4``, but got {image.ndim}.")
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:  # HW -> 1HW
            image = image.unsqueeze(dim=0)
        elif image.ndim == 4 and image.shape[1] == 1:  # B1HW -> BHW
            image = image.squeeze(dim=1)
        elif image.ndim == 4 and image.shape[0] == 1:  # 1CHW -> CHW
            image = image.squeeze(dim=0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:  # HW -> HW1
            image = np.expand_dims(image, axis=-1)
        elif image.ndim == 4 and image.shape[0] == 1:  # 1HWC -> HWC
            image = np.squeeze(image, axis=0)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    return image


def to_list_of_3d_image(image: Any) -> list[torch.Tensor | np.ndarray]:
    """Convert arbitrary input to a :obj:`list` of 3D images."""
    if isinstance(image, (torch.Tensor, np.ndarray)):
        if image.ndim == 3:
            image = [image]
        elif image.ndim == 4:
            image = list(image)
        else:
            raise ValueError
    elif isinstance(image, list | tuple):
        if not all(isinstance(i, (torch.Tensor, np.ndarray)) for i in image):
            raise ValueError
    return image


def to_4d_image(image: Any) -> torch.Tensor | np.ndarray:
    """Convert a 2D, 3D, 5D, list of 3D, and list of 4D images to 4D."""
    if isinstance(image, (torch.Tensor, np.ndarray)) and not 2 <= image.ndim <= 5:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``2`` and ``5``, but got {image.ndim}.")
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:  # HW -> 11HW
            image = image.unsqueeze(dim=0)
            image = image.unsqueeze(dim=0)
        elif image.ndim == 3:  # CHW -> 1CHW
            image = image.unsqueeze(dim=0)
        elif image.ndim == 5 and image.shape[0] == 1:  # 1BCHW -> BCHW
            image = image.squeeze(dim=0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:  # HW -> 1HW1
            image = np.expand_dims(image, axis=-1)
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3:  # HWC -> 1HWC
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 5 and image.shape[0] == 1:  # 1BHWC -> BHWC
            image = np.squeeze(image, axis=0)
    elif isinstance(image, list | tuple):
        if all(isinstance(i, torch.Tensor)   and i.ndim == 3 for i in image):
            image = torch.stack(image, dim=0)
        elif all(isinstance(i, torch.Tensor) and i.ndim == 4 for i in image):
            image = torch.cat(image, dim=0)
        elif all(isinstance(i, np.ndarray)   and i.ndim == 3 for i in image):
            image = np.array(image)
        elif all(isinstance(i, np.ndarray)   and i.ndim == 4 for i in image):
            image = np.concatenate(image, axis=0)
        # else:
        #     error_console.log(f"input's number of dimensions must be between ``3`` and ``4``.")
        #     image = None
    else:
        raise TypeError(f"`image` must be a `numpy.ndarray`, `torch.Tensor`, "
                        f"or a `list` of either of them, but got {type(image)}.")
    return image


def to_channel_first_image(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an image to the channel-first format."""
    if is_channel_first_image(image):
        return image
    if not 3 <= image.ndim <= 5:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``3`` and ``5``, but got {image.ndim}.")
    if isinstance(image, torch.Tensor):
        image = image.clone()
        if image.ndim == 3:
            image = image.permute(2, 0, 1)
        elif image.ndim == 4:
            image = image.permute(0, 3, 1, 2)
        elif image.ndim == 5:
            image = image.permute(0, 1, 4, 2, 3)
    elif isinstance(image, np.ndarray):
        image = copy.deepcopy(image)
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))
        elif image.ndim == 4:
            image = np.transpose(image, (0, 3, 1, 2))
        elif image.ndim == 5:
            image = np.transpose(image, (0, 1, 4, 2, 3))
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    return image


def to_channel_last_image(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an image to the channel-last format."""
    if is_channel_last_image(image):
        return image
    if not 3 <= image.ndim <= 5:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``3`` and ``5``, but got {image.ndim}.")
    if isinstance(image, torch.Tensor):
        image = image.clone()
        if image.ndim == 3:
            image = image.permute(1, 2, 0)
        elif image.ndim == 4:
            image = image.permute(0, 2, 3, 1)
        elif image.ndim == 5:
            image = image.permute(0, 1, 3, 4, 2)
    elif isinstance(image, np.ndarray):
        image = copy.deepcopy(image)
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
        elif image.ndim == 4:
            image = np.transpose(image, (0, 2, 3, 1))
        elif image.ndim == 5:
            image = np.transpose(image, (0, 1, 3, 4, 2))
    else:
        raise TypeError(
            f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
            f"but got {type(image)}."
        )
    return image


def to_image_nparray(
    image      : torch.Tensor | np.ndarray,
    keepdim    : bool = False,
    denormalize: bool = False,
) -> np.ndarray:
    """Convert an image to :obj:`numpy.ndarray`.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        keepdim: If `True`, keep the original shape. If ``False``, convert it to
            a 3D shape. Default: ``True``.
        denormalize: If ``True``, convert image to ``[0, 255]``. Default: ``True``.

    Returns:
        An image of type :obj:`numpy.ndarray`.
    """
    if not 3 <= image.ndim <= 5:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``3`` and ``5``, but got {image.ndim}.")
    if isinstance(image, torch.Tensor):
        image = image.detach()
        image = image.cpu().numpy()
    image = denormalize_image(image).astype(np.uint8) if denormalize else image
    image = to_channel_last_image(image)
    if not keepdim:
        image = to_3d_image(image)
    return image


def to_image_tensor(
    image    : torch.Tensor | np.ndarray,
    keepdim  : bool = False,
    normalize: bool = False,
    device   : Any  = None,
) -> torch.Tensor:
    """Convert an image from :obj:`PIL.Image` or :obj:`numpy.ndarray` to
    :obj:`torch.Tensor`. Optionally, convert :obj:`image` to channel-first
    format and normalize it.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        keepdim: If ``True``, keep the original shape. If ``False``, convert it
            to a 4D shape. Default: ``True``.
        normalize: If ``True``, normalize the image to ``[0.0, 1.0]``.
            Default: ``False``.
        device: The device to run the model on. If ``None``, the default
            ``'cpu'`` device is used.
        
    Returns:
        A image of type :obj:`torch.Tensor`.
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).contiguous()
    elif isinstance(image, torch.Tensor):
        image = image.clone()
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    image = to_channel_first_image(image)
    if not keepdim:
        image = to_4d_image(image)
    image = normalize_image(image) if normalize else image
    # Place in memory
    image = image.contiguous()
    if device:
        image = image.to(device)
    return image

# endregion


# region I/O

def read_image(
    path     : core.Path,
    flags    : int  = cv2.IMREAD_COLOR,
    to_tensor: bool = False,
    normalize: bool = False,
) -> torch.Tensor | np.ndarray:
    """Read an image from a file path using :obj:`cv2`. Optionally, convert it
    to RGB format, and :obj:`torch.Tensor` type of shape ``[1, C, H, W]``.

    Args:
        path: An image's file path.
        flags: A flag to read the image. One of:
            - cv2.IMREAD_UNCHANGED           = -1,
            - cv2.IMREAD_GRAYSCALE           = 0,
            - cv2.IMREAD_COLOR               = 1,
            - cv2.IMREAD_ANYDEPTH            = 2,
            - cv2.IMREAD_ANYCOLOR            = 4,
            - cv2.IMREAD_LOAD_GDAL           = 8,
            - cv2.IMREAD_REDUCED_GRAYSCALE_2 = 16,
            - cv2.IMREAD_REDUCED_COLOR_2     = 17,
            - cv2.IMREAD_REDUCED_GRAYSCALE_4 = 32,
            - cv2.IMREAD_REDUCED_COLOR_4     = 33,
            - cv2.IMREAD_REDUCED_GRAYSCALE_8 = 64,
            - cv2.IMREAD_REDUCED_COLOR_8     = 65,
            - cv2.IMREAD_IGNORE_ORIENTATION  = 128
            Default: ``cv2.IMREAD_COLOR``.
        to_tensor: If ``True``, convert the image from :obj:`numpy.ndarray`
            to :obj:`torch.Tensor`. Default: ``False``.
        normalize: If ``True``, normalize the image to ``[0.0, 1.0]``.
            Default: ``False``.
        
    Return:
        An RGB or grayscale image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
    """
    path = core.Path(path)
    # Read raw image
    if path.is_raw_image_file():
        image = rawpy.imread(str(path))
        image = image.postprocess()
    # Read other types of image
    else:
        image = cv2.imread(str(path), flags)  # BGR
        if image.ndim == 2:  # HW -> HW1 (OpenCV read grayscale image)
            image = np.expand_dims(image, axis=-1)
        if is_color_image(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to tensor
    if to_tensor:
        image = to_image_tensor(image, False, normalize)
    return image


def read_image_shape(path: core.Path) -> tuple[int, int, int]:
    """Read an image from a file path using :obj:`PIL` and get its shape in
    ``[H, W, C]`` format. Using :obj:`PIL` is faster than using OpenCV.
    
    Args:
        path: An image file path.
    """
    # Read raw image
    if path.is_raw_image_file():
        image = rawpy.imread('path_to_your_image.dng')
        image = image.raw_image_visible
        h, w  = image.shape
        c     = 3
    # Read other types of image
    else:
        with Image.open(str(path)) as image:
            w, h = image.size
            mode = image.mode  # This tells the color depth (e.g., "RGB", "L", "RGBA")
            # Determine the number of channels (depth) based on the mode
            if mode == "RGB":
                c = 3
            elif mode == "RGBA":
                c = 4
            elif mode == "L":  # Grayscale
                c = 1
            else:
                raise ValueError(f"Unsupported image mode: {mode}.")
    return h, w, c


def write_image(path: core.Path, image: torch.Tensor | np.ndarray):
    """Write an image to a file path.
    
    Args:
        path: A path to write the image to.
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
    """
    path = core.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, torch.Tensor):
        torchvision.utils.save_image(image, str(path))
    elif isinstance(image, np.ndarray):
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        raise ValueError(f"`image` must be `torch.Tensor` or `numpy.ndarray`, "
                         f"but got {type(image)}.")
    

def write_image_cv(
    image      : torch.Tensor | np.ndarray,
    dir_path   : core.Path,
    name       : str,
    prefix     : str  = "",
    extension  : str  = ".jpg",
    denormalize: bool = False
):
    """Write an image to a directory using :obj:`cv2`.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        dir_path: A directory to write the image to.
        name: An image's name.
        prefix: A prefix to add to the :obj:`name`. Default: ``''``.
        extension: An extension of the image file. Default: ``'.png'``.
        denormalize: If ``True``, convert the image to ``[0, 255]``.
            Default: ``False``.
    """
    # Convert image
    if isinstance(image, torch.Tensor):
        image = to_image_nparray(image, True, denormalize)
    image = to_channel_last_image(image)
    if 2 <= image.ndim <= 3:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``2`` and ``3``, but got {image.ndim}.")
    # Write image
    dir_path  = core.Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    name      = core.Path(name)
    stem      = name.stem
    extension = extension  # name.suffix
    extension = f"{name.suffix}" if extension == "" else extension
    extension = f".{extension}"  if "." not in extension else extension
    stem      = f"{prefix}_{stem}" if prefix != "" else stem
    name      = f"{stem}{extension}"
    file_path = dir_path / name
    cv2.imwrite(str(file_path), image)


def write_image_torch(
    image      : torch.Tensor | np.ndarray,
    dir_path   : core.Path,
    name       : str,
    prefix     : str  = "",
    extension  : str  = ".jpg",
    denormalize: bool = False
):
    """Write an image to a directory.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        dir_path: A directory to write the image to.
        name: An image's name.
        prefix: A prefix to add to the :obj:`name`. Default: ``''``.
        extension: An extension of the image file. Default: ``'.png'``.
        denormalize: If ``True``, convert the image to ``[0, 255]``.
            Default: ``False``.
    """
    # Convert image
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        image = to_channel_first_image(image)
    image = denormalize_image(image) if denormalize else image
    image = image.to(torch.uint8)
    image = image.cpu()
    if 2 <= image.ndim <= 3:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``2`` and ``3``, but got {image.ndim}.")
    # Write image
    dir_path  = core.Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    name      = core.Path(name)
    stem      = name.stem
    extension = extension  # name.suffix
    extension = f"{name.suffix}" if extension == "" else extension
    extension = f".{extension}" if "." not in extension else extension
    stem      = f"{prefix}_{stem}" if prefix != "" else stem
    name      = f"{stem}{extension}"
    file_path = dir_path / name
    if extension in [".jpg", ".jpeg"]:
        torchvision.io.image.write_jpeg(input=image, filename=str(file_path))
    elif extension in [".jpg"]:
        torchvision.io.image.write_png(input=image, filename=str(file_path))


def write_images_cv(
    images     : list[torch.Tensor | np.ndarray],
    dir_path   : core.Path,
    names      : list[str],
    prefixes   : list[str] = "",
    extension  : str       = ".jpg",
    denormalize: bool      = False
):
    """Write a :obj:`list` of images to a directory using :obj:`cv2`.
   
    Args:
        images: A :obj:`list` of images.
        dir_path: A directory to write the images to.
        names: A :obj:`list` of images' names.
        prefixes: A prefix to add to the :obj:`names`. Default: ``''``.
        extension: An extension of image files. Default: ``'.png'``.
        denormalize: If ``True``, convert image to ``[0, 255]``.
            Default: ``False``.
    """
    if isinstance(names, str):
        names = [names for _ in range(len(images))]
    if isinstance(prefixes, str):
        prefixes = [prefixes for _ in range(len(prefixes))]
    if not len(images) == len(names):
        raise ValueError(f"`images` and `names` must have the same length, "
                         f"but got {len(images)} and {len(names)}.")
    if not len(images) == len(prefixes):
        raise ValueError(f"`images` and `prefixes` must have the same length, "
                         f"but got {len(images)} and {len(prefixes)}.")
    num_jobs = multiprocessing.cpu_count()
    joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(write_image_cv)(
            image, dir_path, names[i], prefixes[i], extension, denormalize
        )
        for i, image in enumerate(images)
    )


def write_images_torch(
    images     : list[torch.Tensor | np.ndarray],
    dir_path   : core.Path,
    names      : list[str],
    prefixes   : list[str] = "",
    extension  : str       = ".jpg",
    denormalize: bool      = False
):
    """Write a :obj:`list` of images to a directory using :obj:`torchvision`.
   
    Args:
        images: A :obj:`list` of images.
        dir_path: A directory to write the images to.
        names: A :obj:`list` of images' names.
        prefixes: A prefix to add to the :obj:`names`. Default: ``''``.
        extension: An extension of image files. Default: ``'.png'``.
        denormalize: If ``True``, convert image to ``[0, 255]``.
            Default: ``False``.
    """
    if isinstance(names, str):
        names = [names for _ in range(len(images))]
    if isinstance(prefixes, str):
        prefixes = [prefixes for _ in range(len(prefixes))]
    if not len(images) == len(names):
        raise ValueError(f"`images` and `names` must have the same length, "
                         f"but got {len(images)} and {len(names)}.")
    if not len(images) == len(prefixes):
        raise ValueError(f"`images` and `prefixes` must have the same length, "
                         f"but got {len(images)} and {len(prefixes)}.")
    num_jobs = multiprocessing.cpu_count()
    joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(write_image_torch)(
            image, dir_path, names[i], prefixes[i], extension, denormalize
        )
        for i, image in enumerate(images)
    )

# endregion


# region Normalize

def denormalize_image_mean_std(
    image: torch.Tensor | np.ndarray,
    mean : float | list[float] = [0.485, 0.456, 0.406],
    std  : float | list[float] = [0.229, 0.224, 0.225],
    eps  : float               = 1e-6,
) -> torch.Tensor | np.ndarray:
    """Denormalize an image with mean and standard deviation.
    
    image[channel] = (image[channel] * std[channel]) + mean[channel]
    where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n` channels.

    Args:
        image: An image of type :obj:`torch.Tensor` in ``[B, C, H, W]`` format
            with data in the range ``[0.0, 1.0]``.
        mean: A sequence of means for each channel.
            Default: ``[0.485, 0.456, 0.406]``.
        std: A sequence of standard deviations for each channel.
            Default: ``[0.229, 0.224, 0.225]``.
        eps: Epsilon value to avoid division by zero. Defaults: ``1e-6``.
        
    Returns:
        A denormalized image.
    """
    if not image.ndim >= 3:
        raise ValueError(f"`image`'s number of dimensions must be >= ``3``, "
                         f"but got {image.ndim}.")
    if isinstance(image, torch.Tensor):
        image = image.clone()
        image = image.to(dtype=torch.get_default_dtype()) \
            if not image.is_floating_point() else image
        shape  = image.shape
        device = image.device
        dtype  = image.dtype
        if isinstance(mean, float):
            mean = torch.tensor([mean] * shape[1], dtype=dtype, device=device)
        elif isinstance(mean, (list, tuple)):
            mean = torch.as_tensor(mean, dtype=dtype, device=image.device)
        elif isinstance(mean, torch.Tensor):
            mean = mean.to(dtype=dtype, device=image.device)
        
        if isinstance(std, float):
            std = torch.tensor([std] * shape[1], dtype=dtype, device=device)
        elif isinstance(std, (list, tuple)):
            std = torch.as_tensor(std, dtype=dtype, device=image.device)
        elif isinstance(std, torch.Tensor):
            std = std.to(dtype=dtype, device=image.device)
        
        std_inv  = 1.0 / (std + eps)
        mean_inv = -mean * std_inv
        std_inv  = std_inv.view(-1, 1, 1)  if std_inv.ndim  == 1 else std_inv
        mean_inv = mean_inv.view(-1, 1, 1) if mean_inv.ndim == 1 else mean_inv
        image.sub_(mean_inv).div_(std_inv)
    elif isinstance(image, np.ndarray):
        raise NotImplementedError(f"This function has not been implemented.")
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    return image


def normalize_image_mean_std(
    image: torch.Tensor | np.ndarray,
    mean : float | list[float] = [0.485, 0.456, 0.406],
    std  : float | list[float] = [0.229, 0.224, 0.225],
    eps  : float               = 1e-6,
) -> torch.Tensor | np.ndarray:
    """Normalize an image with mean and standard deviation.
    
    image[channel] = (image[channel] * std[channel]) + mean[channel]
    where :obj:`mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for ``n``
    channels.

    Args:
        image: An image of type :obj:`torch.Tensor` in ``[B, C, H, W]`` format
            with data in the range ``[0, 255]``.
        mean: A sequence of means for each channel.
            Default: ``[0.485, 0.456, 0.406]``.
        std: A sequence of standard deviations for each channel.
            Default: ``[0.229, 0.224, 0.225]``.
        eps: Epsilon value to avoid division by zero. Defaults: ``1e-6``.
        
    Returns:
        A normalized image.
    """
    if not image.ndim >= 3:
        raise ValueError(f"`image`'s number of dimensions must be >= ``3``, "
                         f"but got {image.ndim}.")
    if isinstance(image, torch.Tensor):
        image = image.clone()
        image = image.to(dtype=torch.get_default_dtype()) \
            if not image.is_floating_point() else image
        shape  = image.shape
        device = image.device
        dtype  = image.dtype
        if isinstance(mean, float):
            mean = torch.tensor([mean] * shape[1], device=device, dtype=dtype)
        elif isinstance(mean, (list, tuple)):
            mean = torch.as_tensor(mean, dtype=dtype, device=image.device)
        elif isinstance(mean, torch.Tensor):
            mean = mean.to(dtype=dtype, device=image.device)
        
        if isinstance(std, float):
            std = torch.tensor([std] * shape[1], device=device, dtype=dtype)
        elif isinstance(std, (list, tuple)):
            std = torch.as_tensor(std, dtype=dtype, device=image.device)
        elif isinstance(std, torch.Tensor):
            std = std.to(dtype=dtype, device=image.device)
        std += eps
        
        mean = mean.view(-1, 1, 1) if mean.ndim == 1 else mean
        std  = std.view(-1, 1, 1)  if std.ndim  == 1 else std
        image.sub_(mean).div_(std)
    elif isinstance(image, np.ndarray):
        raise NotImplementedError(f"This function has not been implemented.")
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    return image


def normalize_image_by_range(
    image  : torch.Tensor | np.ndarray,
    min    : float = 0.0,
    max    : float = 255.0,
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> torch.Tensor | np.ndarray:
    """Normalize an image from the range ``[:obj:`min`, :obj:`max`]`` to the
    ``[:obj:`new_min`, :obj:`new_max`]``.
    
    Args:
        image: An image of type :obj:`torch.Tensor` in ``[B, C, H, W]`` format
            with data in the range ``[0, 255]``.
        min: The current minimum pixel value of the image. Default: ``0.0``.
        max: The current maximum pixel value of the image. Default: ``255.0``.
        new_min: A new minimum pixel value of the image. Default: ``0.0``.
        new_max: A new minimum pixel value of the image. Default: ``1.0``.
        
    Returns:
        A normalized image.
    """
    if not image.ndim >= 3:
        raise ValueError(f"`image`'s number of dimensions must be >= ``3``, "
                         f"but got {image.ndim}.")
    # if is_normalized_image(image=image):
    #     return image
    if isinstance(image, torch.Tensor):
        image = image.clone()
        # input = input.to(dtype=torch.get_default_dtype()) if not input.is_floating_point() else input
        image = image.to(dtype=torch.get_default_dtype())
        ratio = (new_max - new_min) / (max - min)
        image = (image - min) * ratio + new_min
        # image = torch.clamp(image, new_min, new_max)
    elif isinstance(image, np.ndarray):
        image = copy.deepcopy(image)
        image = image.astype(np.float32)
        ratio = (new_max - new_min) / (max - min)
        image = (image - min) * ratio + new_min
        # image = np.clip(image, new_min, new_max)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    return image


denormalize_image = functools.partial(
    normalize_image_by_range,
    min     = 0.0,
    max     = 1.0,
    new_min = 0.0,
    new_max = 255.0
)
normalize_image = functools.partial(
    normalize_image_by_range,
    min     = 0.0,
    max     = 255.0,
    new_min = 0.0,
    new_max = 1.0
)

# endregion


# region Prior

def atmospheric_prior(
    image      : np.ndarray,
    kernel_size: _size_2_t = 15,
    p          : float     = 0.0001
) -> np.ndarray:
    """Get the atmosphere light in RGB image.

    Args:
        image: An RGB image of type :obj:`numpy.ndarray` in ``[H, W, C]``
            format with data in the range ``[0, 255]``.
        kernel_size: Window for the dark channel. Default: ``15``.
        p: Percentage of pixels for estimating the atmosphere light.
            Default: ``0.0001``.
    
    Returns:
        A 3-element array containing atmosphere light ``([0, L-1])`` for each
        channel.
    """
    image      = image.transpose(1, 2, 0)
    # Reference CVPR09, 4.4
    dark       = dark_channel_prior_02(image=image, kernel_size=kernel_size)
    m, n       = dark.shape
    flat_i     = image.reshape(m * n, 3)
    flat_dark  = dark.ravel()
    search_idx = (-flat_dark).argsort()[:int(m * n * p)]  # find top M * N * p indexes
    # Return the highest intensity for each channel
    return np.max(flat_i.take(search_idx, axis=0), axis=0)


def blur_spot_prior(image: np.ndarray, threshold: int = 250) -> bool:
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding for bright spot detection
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Apply Laplacian filter for edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Calculate maximum intensity and variance
    laplacian_var = laplacian.var()
    # Check blur condition based on variance of Laplacian image
    is_blur = True if laplacian_var < threshold else False
    return is_blur


def boundary_aware_prior(
    image      : torch.Tensor | np.ndarray,
    eps        : float = 0.05,
    as_gradient: bool  = False,
    normalized : bool  = False,
) -> torch.Tensor | np.ndarray:
    """Get the boundary prior from an RGB or grayscale image.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        eps: Threshold to remove weak edges. Default: ``0.05``.
        as_gradient: If ``True``, return the gradient image. Default: ``False``.
        normalized: If ``True``, L1 norm of the kernel is set to ``1``.
            Default: ``False``.
        
    Returns:
        A boundary aware prior as a binary image.
    """
    if isinstance(image, torch.Tensor):
        gradient = kornia.filters.sobel(image, normalized=normalized, eps=1e-6)
        g_max    = torch.max(gradient)
        gradient = gradient / g_max
        boundary = (gradient > eps).float()
    elif isinstance(image, np.ndarray):
        if is_color_image(image):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        from mon.vision.filtering import sobel_filter
        gradient = sobel_filter(image, kernel_size=3)
        g_max    = np.max(gradient)
        gradient = gradient / g_max
        boundary = (gradient > eps).float()
        return boundary
    else:
        raise ValueError(f"Unsupported input type: {type(image)}.")
    
    # return boundary, gradient
    if as_gradient:
        return gradient
    else:
        return boundary


def bright_spot_prior(image: np.ndarray) -> bool:
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding for bright spot detection
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Calculate maximum intensity and variance
    binary_var = binary.var()
    # Check bright spot condition based on variance of binary image
    is_bright = True if 5000 < binary_var < 8500 else False
    return is_bright


def bright_channel_prior(
    image      : torch.Tensor | np.ndarray,
    kernel_size: _size_2_t
) -> torch.Tensor | np.ndarray:
    """Get the bright channel prior from an RGB image.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        kernel_size: Window size.

    Returns:
        A bright channel prior.
    """
    kernel_size = core.to_2tuple(kernel_size)
    if isinstance(image, torch.Tensor):
        bright_channel = torch.max(image, dim=1)[0]
        kernel         = torch.ones(kernel_size[0], kernel_size[0])
        bcp            = kornia.morphology.erosion(bright_channel, kernel)
    elif isinstance(image, np.ndarray):
        bright_channel = np.max(image, axis=2)
        kernel_size    = core.to_2tuple(kernel_size)
        kernel         = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        bcp            = cv2.erode(bright_channel, kernel)
    else:
        raise ValueError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`,"
                         f" but got {type(image)}.")
    return bcp


def brightness_attention_map(
    image        : torch.Tensor | np.ndarray,
    gamma        : float     = 2.5,
    denoise_ksize: _size_2_t = None,
) -> torch.Tensor:
    """Get the Brightness Attention Map (BAM) prior from an RGB image.
    
    This is a self-attention map extracted from the V-channel of a low-light
    image. This map is multiplied to convolutional activations of all layers in
    the enhancement network. Brighter regions are given lower weights to avoid
    over-saturation, while preserving image details and enhancing the contrast
    in the dark regions effectively.
    
    Equation: `I_{attn} = (1 - I_{V})^{\gamma}`, where `\gamma \geq 1`.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        gamma: A parameter controls the curvature of the map.
        denoise_ksize: Window size for de-noising operation. Default: ``None``.
        
    Returns:
        An :obj:`numpy.ndarray` brightness enhancement map as prior.
    """
    if isinstance(image, torch.Tensor):
        if denoise_ksize:
            image = kornia.filters.median_blur(image, denoise_ksize)
            # image = kornia.filters.bilateral_blur(image, denoise_ksize, 0.1, (1.5, 1.5))
        hsv = kornia.color.rgb_to_hsv(image)
        v   = get_image_channel(image=hsv, index=(2, 3), keep_dim=True)  # hsv[:, 2:3, :, :]
        bam = torch.pow((1 - v), gamma)
    elif isinstance(image, np.ndarray):
        if denoise_ksize:
            image = cv2.medianBlur(image, denoise_ksize)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if hsv.dtype != np.float64:
            hsv  = hsv.astype("float64")
            hsv /= 255.0
        v   = get_image_channel(image=hsv, index=(2, 3), keep_dim=True)  # hsv[:, :, 2:3]
        bam = np.power((1 - v), gamma)
    else:
        raise ValueError(f"Unsupported input type: {type(image)}.")
    return bam


def dark_channel_prior(
    image      : torch.Tensor | np.ndarray,
    kernel_size: int
) -> torch.Tensor | np.ndarray:
    """Get the dark channel prior from an RGB image.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        kernel_size: Window size.
        
    Returns:
        A dark channel prior.
    """
    kernel_size = core.to_2tuple(kernel_size)
    if isinstance(image, torch.Tensor):
        dark_channel = torch.min(image, dim=1)[0]
        kernel       = torch.ones(kernel_size[0], kernel_size[1])
        dcp          = kornia.morphology.erosion(dark_channel, kernel)
    elif isinstance(image, np.ndarray):
        dark_channel = np.min(image, axis=2)
        kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        dcp          = cv2.erode(dark_channel, kernel)
    else:
        raise ValueError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`,"
                         f" but got {type(image)}.")
    return dcp


def dark_channel_prior_02(
    image      : torch.Tensor | np.ndarray,
    kernel_size: _size_2_t
) -> torch.Tensor | np.ndarray:
    """Get the dark channel prior from an RGB image.

    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        kernel_size: Window size.

    Returns:
        A dark channel prior.
    """
    m, n, _ = image.shape
    w       = kernel_size
    padded  = np.pad(image, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), "edge")
    dcp     = np.zeros((m, n))
    for i, j in np.ndindex(dcp.shape):
        dcp[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return dcp


def image_local_mean(image: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
    """Calculate the local mean of an image using a sliding window.
    
    Args:
        image: The input image tensor of shape ``[B, C, H, W]``.
        patch_size: The size of the sliding window. Default: ``5``.
    """
    padding = patch_size // 2
    image   = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    return patches.mean(dim=(4, 5))


def image_local_variance(image: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
    """Calculate the local variance of an image using a sliding window.
    
    Args:
        image: The input image tensor of shape ``[B, C, H, W]``.
        patch_size: The size of the sliding window. Default: ``5``.
    """
    padding = patch_size // 2
    image   = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    mean    = patches.mean(dim=(4, 5))
    return ((patches - mean.unsqueeze(4).unsqueeze(5)) ** 2).mean(dim=(4, 5))


def image_local_stddev(
    image     : torch.Tensor,
    patch_size: int   = 5,
    eps       : float = 1e-9
) -> torch.Tensor:
    """Calculate the local standard deviation of an image using a sliding window.
    
    Args:
        image: The input image tensor of shape ``[B, C, H, W]``.
        patch_size: The size of the sliding window. Default: ``5``.
        eps: A small value to avoid sqrt by zero. Default: ``1e-9``.
    """
    padding        = patch_size // 2
    image          = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches        = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    mean           = patches.mean(dim=(4, 5), keepdim=True)
    squared_diff   = (patches - mean) ** 2
    local_variance = squared_diff.mean(dim=(4, 5))
    local_stddev   = torch.sqrt(local_variance + eps)
    return local_stddev


class BoundaryAwarePrior(nn.Module):
    """Get the boundary prior from an RGB or grayscale image.
    
    Args:
        eps: Threshold weak edges. Default: ``0.05``.
        normalized: If ``True``, L1 norm of the kernel is set to ``1``.
            Default: ``True``.
    """
    
    def __init__(self, eps: float = 0.05, normalized: bool = False):
        super().__init__()
        self.eps        = eps
        self.normalized = normalized
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return boundary_aware_prior(image, self.eps, self.normalized)


class BrightnessAttentionMap(nn.Module):
    """Get the Brightness Attention Map (BAM) prior from an RGB image.
    
    This is a self-attention map extracted from the V-channel of a low-light
    image. This map is multiplied to convolutional activations of all layers in
    the enhancement network. Brighter regions are given lower weights to avoid
    over-saturation, while preserving image details and enhancing the contrast
    in the dark regions effectively.
    
    Equation: `I_{attn} = (1 - I_{V})^{\gamma}`, where `\gamma \geq 1`.
    
    Args:
        gamma: A parameter controls the curvature of the map.
        denoise_ksize: Window size for de-noising operation. Default: ``None``.
    """
    
    def __init__(
        self,
        gamma        : float     = 2.5,
        denoise_ksize: _size_2_t = None
    ):
        super().__init__()
        self.gamma         = gamma
        self.denoise_ksize = denoise_ksize
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return brightness_attention_map(image, self.gamma, self.denoise_ksize)


class ImageLocalMean(nn.Module):
    """Calculate the local mean of an image using a sliding window.
    
    Args:
        patch_size: The size of the sliding window. Default: ``5``.
    """
    
    def __init__(self, patch_size: int = 5):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, image):
        return image_local_mean(image, self.patch_size)


class ImageLocalVariance(nn.Module):
    """Calculate the local variance of an image using a sliding window.
    
    Args:
        patch_size: The size of the sliding window. Default: ``5``.
    """
    
    def __init__(self, patch_size: int = 5):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, image):
        return image_local_variance(image, self.patch_size)


class ImageLocalStdDev(nn.Module):
    """Calculate the local standard deviation of an image using a sliding window.
    
    Args:
        patch_size: The size of the sliding window. Default: ``5``.
        eps: A small value to avoid sqrt by zero. Default: ``1e-9``.
    """
    
    def __init__(self, patch_size: int = 5, eps: float = 1e-9):
        super().__init__()
        self.patch_size = patch_size
        self.eps        = eps
    
    def forward(self, image):
        return image_local_stddev(image, self.patch_size, self.eps)

# endregion
