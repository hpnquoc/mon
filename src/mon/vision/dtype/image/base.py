#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements basic functionalities for image data."""

from __future__ import annotations

__all__ = [
    "add_images_weighted",
    "blend_images",
    "convert_image_to_2d",
    "convert_image_to_3d",
    "convert_image_to_4d",
    "convert_image_to_array",
    "convert_image_to_channel_first",
    "convert_image_to_channel_last",
    "convert_image_to_tensor",
    "denormalize_image",
    "get_image_center",
    "get_image_center4",
    "get_image_channel",
    "get_image_num_channels",
    "get_image_shape",
    "get_image_size",
    "is_image",
    "is_image_channel_first",
    "is_image_channel_last",
    "is_image_colored",
    "is_image_grayscale",
    "is_image_normalized",
    "normalize_image",
    "normalize_image_by_range",
    "read_image",
    "read_image_shape",
    "write_image",
]

import functools
import math
from typing import Any, Sequence

import cv2
import numpy as np
import rawpy
import torch
import torchvision
from PIL import Image

from mon import core


# region Assertion

def is_image(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an input is an image tensor or array.

    Args:
        image: Input to evaluate as ``torch.Tensor`` or ``np.ndarray``.

    Returns:
        ``True`` if input is a tensor or array and a color or grayscale image,
        ``False`` otherwise.
    """
    return (isinstance(image, (torch.Tensor, np.ndarray)) and
            (is_image_colored(image) or is_image_grayscale(image)))


def is_image_channel_first(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an image is in channel-first format.

    Args:
        image: Image as ``torch.Tensor`` or ``np.ndarray`` in [C, H, W] or
            [B, C, H, W] format.

    Returns:
        ``True`` if channel-first (e.g., [C, H, W]),
        ``False`` if channel-last (e.g., [H, W, C]).

    Raises:
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``np.ndarray``.
        ValueError: If ``image`` dimensions are invalid or channel format is ambiguous.

    Notes:
        Assumes the smallest dimension is the channel dimension.
    """
    # Determine tensor type and get shape
    if isinstance(image, torch.Tensor):
        shape = image.size()
    elif isinstance(image, np.ndarray):
        shape = image.shape
    else:
        raise TypeError(f"[image] must be a numpy.ndarray or torch.Tensor, got {type(image)}.")
    
    # Check if tensor has at least 3 dimensions (batch, height/width, channels)
    if not 3 <= len(shape) <= 4:
        raise ValueError("[image] must have at least 3 dimensions (batch, channels, height/width).")
    
    # Extract dimensions
    if len(shape) == 3:
        s0, s1, s2 = shape
    else:
        _, s0, s1, s2 = shape
    
    # Heuristic: Channels are typically smaller than spatial dimensions
    if (s0 < s1) and (s0 < s2):
        return True
    elif (s2 < s0) and (s2 < s1):
        return False
    else:
        raise ValueError(f"Cannot determine channel format for shape [{shape}].")


def is_image_channel_last(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an image is in channel-last format.

    Args:
        image: Image as ``torch.Tensor`` or ``np.ndarray`` in [H, W, C] or
            [B, H, W, C] format.

    Returns:
        ``True`` if channel-last (e.g., [H, W, C]),
        ``False`` if channel-first (e.g., [C, H, W]).
    """
    return not is_image_channel_first(image)


def is_image_colored(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an image is a color image.

    Args:
        image: Image as ``torch.Tensor`` or ``np.ndarray``.

    Returns:
        ``True`` if the image has 3 or 4 channels, ``False`` otherwise.

    Notes:
        Assumes a color image has 3 or 4 channels (e.g., RGB or RGBA).
    """
    return get_image_num_channels(image) in [3, 4]


def is_image_grayscale(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an image is grayscale.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray``.
   
    Returns:
        ``True`` if the image has 1 channel or 2 dimensions, ``False`` otherwise.
    
    Notes:
        Assumes a grayscale image has 1 channel (e.g., [H, W] or [B, 1, H, W]).
    """
    return get_image_num_channels(image) == 1 or len(image.shape) == 2


def is_image_normalized(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an image is normalized to range [-1.0, 1.0] or [0.0, 1.0].

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray``.
    
    Returns:
        ``True`` if absolute max value is <= 1.0, ``False`` otherwise.
    
    Raises:
        TypeError: If image is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if isinstance(image, torch.Tensor):
        return abs(torch.max(image)) <= 1.0
    elif isinstance(image, np.ndarray):
        return abs(np.amax(image)) <= 1.0
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")

# endregion


# region Accessing

def get_image_center(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Returns the center of an image as (x=h/2, y=w/2).

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray``.
    
    Returns:
        Center coordinates as ``torch.Tensor`` or ``numpy.ndarray`` with shape [2].
    """
    h, w   = get_image_size(image)
    center = [h / 2, w / 2]
    return torch.tensor(center) if isinstance(image, torch.Tensor) else np.array(center)


def get_image_center4(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Returns the center of an image as (x=h/2, y=w/2, x=h/2, y=w/2).

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray``.
    
    Returns:
        Center coordinates as ``torch.Tensor`` or ``numpy.ndarray`` with shape [4].
    """
    h, w   = get_image_size(image)
    center = [h / 2, w / 2, h / 2, w / 2]
    return torch.tensor(center) if isinstance(image, torch.Tensor) else np.array(center)


def get_image_channel(
    image   : torch.Tensor | np.ndarray,
    index   : int | Sequence[int],
    keep_dim: bool = True
) -> torch.Tensor | np.ndarray:
    """Extracts a channel or channels from an image.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray``.
        index: Channel index (int) or range (Sequence[int]) to extract.
        keep_dim: Keep singleton dimension if ``True``. Default is ``True``.
    
    Returns:
        Extracted channel(s) as ``torch.Tensor`` or ``numpy.ndarray``.
   
    Raises:
        ValueError: If image dimensions are invalid for channel extraction.
    """
    i1, i2 = (index, index + 1) if isinstance(index, int) else (index[0], index[1])
    
    if is_image_channel_first(image):
        if image.ndim == 4:
            return image[:, i1:i2, :, :] if keep_dim else image[:, i1, :, :]
        elif image.ndim == 3:
            return image[i1:i2, :, :]    if keep_dim else image[i1, :, :]
    else:
        if image.ndim == 4:
            return image[:, :, :, i1:i2] if keep_dim else image[:, :, :, i1]
        elif image.ndim == 3:
            return image[:, :, i1:i2]    if keep_dim else image[:, :, i1]
    raise ValueError(f"Invalid image dimensions for channel extraction {image.ndim}.")
    

def get_image_num_channels(image: torch.Tensor | np.ndarray) -> int:
    """Returns the number of channels in an image.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray``.
   
    Returns:
        Number of channels (e.g., 1 for grayscale, 3 for RGB).
    """
    if image.ndim == 4:
        c = image.shape[1] if is_image_channel_first(image) else image.shape[3]
    elif image.ndim == 3:
        c = image.shape[0] if is_image_channel_first(image) else image.shape[2]
    elif image.ndim == 2:
        c = 1
    else:
        c = 0
    return c


def get_image_shape(image: torch.Tensor | np.ndarray) -> list[int]:
    """Returns height, width, and channels of an image.

    Args:
        image: RGB image as ``torch.Tensor`` with shape [B, C, H, W] or
            ``np.ndarray`` with shape [H, W, C].

    Returns:
        List of [height, width, channels] as ``list[int]``.
    """
    h, w, c = (
        (image.shape[-2], image.shape[-1], image.shape[-3])
        if is_image_channel_first(image)
        else (image.shape[-3], image.shape[-2], image.shape[-1])
    )
    return [h, w, c]


def get_image_size(
    input  : torch.Tensor | np.ndarray | int | Sequence[int] | str | core.Path,
    divisor: int = None,
) -> tuple[int, int]:
    """Returns height and width of an image in [H, W] format.

    Args:
        input: RGB image, tensor, array, size, or path as ``torch.Tensor``,
            ``np.ndarray``, ``int``, ``Sequence[int]``, ``str``, or ``core.Path``.
        divisor: Divisor to adjust size as ``int`` or ``None``. Default is ``None``.

    Returns:
        Tuple of (height, width) in pixels as ``tuple[int, int]``.

    Raises:
        TypeError: If ``input`` type is not supported.
    """
    if isinstance(input, (list, tuple)):
        size = input[:2] if len(input) == 3 and input[0] >= input[2] else input[-2:]
    elif isinstance(input, (int, float)):
        size = (input, input)
    elif isinstance(input, (torch.Tensor, np.ndarray)):
        size = (
            (input.shape[-2], input.shape[-1])
            if is_image_channel_first(input)
            else (input.shape[-3], input.shape[-2])
        )
    elif isinstance(input, (str, core.Path)):
        size = read_image_shape(input)[:2]
    else:
        raise TypeError(f"[input] must be a torch.Tensor, numpy.ndarray, int, "
                        f"Sequence[int], str, or core.Path, got {type(input)}.")

    if divisor is not None:
        size = tuple(int(math.ceil(dim / divisor) * divisor) for dim in size)
    return size

# endregion


# region Conversion

def convert_image_to_2d(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Converts a 3D or 4D image to 2D.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray`` in 3D or 4D format.
    
    Returns:
        2D image as ``torch.Tensor`` [H, W] or ``numpy.ndarray`` [H, W].
    
    Raises:
        ValueError: If ``image`` dimensions are not 3 or 4.
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if not 3 <= image.ndim <= 4:
        raise ValueError(f"[image]'s number of dimensions must be between 3 and 4, "
                         f"got {image.ndim}.")
    
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.squeeze(0)
        elif image.ndim == 4 and image.shape[:2] == (1, 1):
            image = image.squeeze(0).squeeze(0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            image = np.squeeze(image, -1)
        elif image.ndim == 4 and image.shape[0] == 1 and image.shape[3] == 1:
            image = np.squeeze(image, (0, -1))
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
    
    return image


def convert_image_to_3d(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Converts a 2D or 4D image to 3D.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray`` in 2D or 4D format.
    
    Returns:
        3D image as ``torch.Tensor`` [C, H, W] or ``numpy.ndarray`` [H, W, C].
    
    Raises:
        ValueError: If ``image`` dimensions are not 2, 3, or 4.
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if not 2 <= image.ndim <= 4:
        raise ValueError(f"[image]'s number of dimensions must be between 2 and 4, "
                         f"got {image.ndim}.")

    if isinstance(image, torch.Tensor):
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 4:
            image = image.squeeze(1) if image.shape[1] == 1 else image.squeeze(0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        elif image.ndim == 4 and image.shape[0] == 1:
            image = np.squeeze(image, 0)
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
    
    return image


def convert_image_to_4d(
    image: torch.Tensor | np.ndarray | list[torch.Tensor] | list[np.ndarray] | tuple[torch.Tensor, ...] | tuple[np.ndarray, ...]
) -> torch.Tensor | np.ndarray:
    """Converts a 2D or 3D image to 4D.

    Args:
        image: Image as ``torch.Tensor``, ``numpy.ndarray``, or list/tuple of 3D/4D images.
    
    Returns:
        4D image as ``torch.Tensor`` [B, C, H, W] or ``numpy.ndarray`` [B, H, W, C].
    
    Raises:
        ValueError: If ``image`` dimensions are not 2, 3, or 4.
        TypeError: If ``image`` type is not supported.
    """
    if not 2 <= image.ndim <= 4:
        raise ValueError(f"[image]'s number of dimensions must be between 2 and 4, "
                         f"got {image.ndim}.")

    if isinstance(image, torch.Tensor):
        if image.ndim == 2:  # [H, W] -> [1, 1, H, W]
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.ndim == 3:  # [C, H, W] -> [1, C, H, W]
            image = image.unsqueeze(0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:  # [H, W] -> [1, H, W, 1]
            image = np.expand_dims(image, axis=(0, -1))
        elif image.ndim == 3:  # [H, W, C] -> [1, H, W, C]
            image = np.expand_dims(image, axis=0)
    elif isinstance(image, (list, tuple)):
        if all(isinstance(i, torch.Tensor) and i.ndim == 3 for i in image):
            image = torch.stack(image, dim=0)  # Stack 3D tensors to [B, C, H, W]
        elif all(isinstance(i, torch.Tensor) and i.ndim == 4 for i in image):
            image = torch.cat(image, dim=0)  # Concatenate 4D tensors along batch
        elif all(isinstance(i, np.ndarray) and i.ndim == 3 for i in image):
            image = np.array(image)  # Convert list of 3D arrays to [B, H, W, C]
        elif all(isinstance(i, np.ndarray) and i.ndim == 4 for i in image):
            image = np.concatenate(image, axis=0)  # Concatenate 4D arrays along batch
        else:
            raise TypeError(f"[image] list/tuple must contain consistent 3D or 4D "
                            f"torch.Tensor or numpy.ndarray, got mixed types or "
                            f"dimensions.")
    else:
        raise TypeError(f"[image] must be a torch.Tensor, numpy.ndarray, or "
                        f"list/tuple of either, got {type(image)}.")
    
    return image


def convert_image_to_channel_first(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Converts an image to channel-first format.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray`` in 3D or 4D format.
    
    Returns:
        Channel-first image as ``torch.Tensor`` [C, H, W] or [B, C, H, W], or
            ``numpy.ndarray`` [C, H, W] or [B, C, H, W].
    
    Raises:
        ValueError: If ``image`` dimensions are not 3 or 4.
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if is_image_channel_first(image):
        return image
    if not 3 <= image.ndim <= 4:
        raise ValueError(f"[image]'s number of dimensions must be between 3 and 4, "
                         f"got {image.ndim}.")
    
    if isinstance(image, torch.Tensor):
        image = image.clone()
        if image.ndim == 3:
            image = image.permute(2, 0, 1)     # [H, W, C] -> [C, H, W]
        elif image.ndim == 4:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    elif isinstance(image, np.ndarray):
        image = np.copy(image)  # Changed from copy.deepcopy for efficiency
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))     # [H, W, C] -> [C, H, W]
        elif image.ndim == 4:
            image = np.transpose(image, (0, 3, 1, 2))  # [B, H, W, C] -> [B, C, H, W]
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
    
    return image


def convert_image_to_channel_last(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Converts an image to channel-last format.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray`` in 3D or 4D format.
    
    Returns:
        Channel-last image as ``torch.Tensor`` [H, W, C] or [B, H, W, C], or
            ``numpy.ndarray`` [H, W, C] or [B, H, W, C].
    
    Raises:
        ValueError: If ``image`` dimensions are not 3 or 4.
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if is_image_channel_last(image):
        return image
    if not 3 <= image.ndim <= 4:
        raise ValueError(f"[image]'s number of dimensions must be between 3 and 4, got {image.ndim}.")
    
    if isinstance(image, torch.Tensor):
        image = image.clone()
        if image.ndim == 3:
            image = image.permute(1, 2, 0)     # [C, H, W] -> [H, W, C]
        elif image.ndim == 4:
            image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    elif isinstance(image, np.ndarray):
        image = np.copy(image)  # Changed from copy.deepcopy for efficiency
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))     # [C, H, W] -> [H, W, C]
        elif image.ndim == 4:
            image = np.transpose(image, (0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
    
    return image


def convert_image_to_array(image: torch.Tensor | np.ndarray, denormalize: bool = False) -> np.ndarray:
    """Converts an image to a ``numpy.ndarray``.
    
    Args:
        image: RGB image as ``torch.Tensor`` [B, C, H, W] or ``numpy.ndarray`` [H, W, C].
        denormalize: Convert to [0, 255] range if ``True``. Default is ``True``.
    
    Returns:
        Image as ``numpy.ndarray`` in [H, W, C] or original shape if ``keepdim`` is ``True``.
    
    Raises:
        ValueError: If ``image`` dimensions are not 3, or 4.
        
    Recommend order:
        image = (tensor.squeeze().detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    """
    # Check shape
    if not 3 <= image.ndim <= 4:
        raise ValueError(f"[image]'s number of dimensions must be between 3 and 4, got {image.ndim}.")
    
    # Remove batch dimension
    image = convert_image_to_3d(image)
    # Detach
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
    # Clamp
    if isinstance(image, torch.Tensor):
        image = image.clamp(0, 1)
    else:
        image = np.clip(image, 0, 1)
    # Rearrange
    image = convert_image_to_channel_last(image)
    # Convert to numpy
    image = image.numpy() if isinstance(image, torch.Tensor) else image
    # Denormalize
    if denormalize:
        image = denormalize_image(image).round().astype(np.uint8)
    
    return image


def convert_image_to_tensor(
    image    : torch.Tensor | np.ndarray,
    normalize: bool = False,
    device   : Any  = None
) -> torch.Tensor:
    """Converts an image to a ``torch.Tensor`` with optional normalization.

    Args:
        image: RGB image as ``torch.Tensor`` [B, C, H, W] or ``numpy.ndarray`` [H, W, C].
        normalize: Normalize to [0.0, 1.0] if ``True``. Default is ``False``.
        device: Device to place tensor on, e.g., ``'cuda'`` or ``None`` for CPU.
            Default is ``None``.
    
    Returns:
        Image as ``torch.Tensor`` in [B, C, H, W] format.
    
    Raises:
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
        
    Recommend order:
        image = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    """
    # Convert to tensor
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).contiguous()
    elif isinstance(image, torch.Tensor):
        image = image.clone()
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
        
    # Rearrange before sending to GPU for better memory layout.
    image = convert_image_to_channel_first(image)
    # Ensure float32 for model input.
    image = image.float()
    # Normalize image
    image = normalize_image(image) if normalize else image
    # Add batch dimension
    image = convert_image_to_4d(image)
    # Place on device
    if device:
        image = image.to(device)
    image = image.contiguous()
    
    return image

# endregion


# region Fusion

def add_images_weighted(
    image1: torch.Tensor | np.ndarray,
    image2: torch.Tensor | np.ndarray,
    alpha : float,
    beta  : float,
    gamma : float = 0.0
) -> torch.Tensor | np.ndarray:
    """Calculates the weighted sum of two image tensors.

    Args:
        image1: First image as ``torch.Tensor`` or ``numpy.ndarray``.
        image2: Second image as ``torch.Tensor`` or ``numpy.ndarray``.
        alpha: Weight for ``image1``.
        beta: Weight for ``image2``.
        gamma: Scalar offset added to the sum. Default is ``0.0``.
    
    Returns:
        Weighted sum as ``torch.Tensor`` or ``numpy.ndarray``.
    
    Raises:
        ValueError: If ``image1`` and ``image2`` differ in shape or type.
        TypeError: If output type is not ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if image1.shape != image2.shape or type(image1) is not type(image2):
        raise ValueError(f"[image1] and [image2] must have the same shape and type, "
                         f"got {type(image1).__name__} and {type(image2).__name__}.")
    
    output = image1 * alpha + image2 * beta + gamma
    bound  = 1.0 if is_image_normalized(image1) else 255.0
    
    if isinstance(output, torch.Tensor):
        output = output.clamp(0, bound).to(image1.dtype)
    elif isinstance(output, np.ndarray):
        output = np.clip(output, 0, bound).astype(image1.dtype)
    else:
        raise TypeError(f"[output] must be a torch.Tensor or numpy.ndarray, got {type(output)}.")
    return output


def blend_images(
    image1: torch.Tensor | np.ndarray,
    image2: torch.Tensor | np.ndarray,
    alpha : float,
    gamma : float = 0.0
) -> torch.Tensor | np.ndarray:
    """Blends two images using a weighted sum.

    Args:
        image1: First image as ``torch.Tensor`` or ``numpy.ndarray``.
        image2: Second image as ``torch.Tensor`` or ``numpy.ndarray``.
        alpha: Weight for ``image1``, with ``image2`` weighted as (1 - ``alpha``).
        gamma: Scalar offset added to the sum. Default is ``0.0``.
    
    Returns:
        Blended image as ``torch.Tensor`` or ``numpy.ndarray``.
    """
    return add_images_weighted(image1=image1, image2=image2, alpha=alpha, beta=1.0 - alpha, gamma=gamma)

# endregion


# region I/O

def read_image(
    path     : core.Path,
    flags    : int = cv2.IMREAD_COLOR,
    to_tensor: bool = False,
    normalize: bool = False,
    device   : Any = None
) -> torch.Tensor | np.ndarray:
    """Reads an image from a file path using OpenCV.

    Args:
        path: Image file path as ``core.Path`` or ``str``.
        flags: OpenCV flag for reading the image. Default is ``cv2.IMREAD_COLOR``.
        to_tensor: Convert to ``torch.Tensor`` if ``True``. Default is ``False``.
        normalize: Normalize to [0.0, 1.0] if ``True``. Default is ``False``.
        device: Device to place tensor on, e.g., ``'cuda'`` or ``None`` for CPU.
            Default is ``None``.
    
    Returns:
        RGB or grayscale image as ``torch.Tensor`` [B, C, H, W] or
        ``numpy.ndarray`` [H, W, C].
    """
    path = core.Path(path)
    if path.is_raw_image_file():  # Read raw image
        image = rawpy.imread(str(path))
        image = image.postprocess()
    else:  # Read other types of image
        image = cv2.imread(str(path), flags)  # BGR
        if image.ndim == 2:  # [H, W] -> [H, W, 1] for grayscale
            image = np.expand_dims(image, axis=-1)
        if is_image_colored(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if to_tensor:
        image = convert_image_to_tensor(image, normalize=normalize, device=device)
    
    return image


def read_image_shape(path: core.Path) -> tuple[int, int, int]:
    """Reads an image shape from a file path using PIL or rawpy.

    Args:
        path: Image file path as ``core.Path`` or ``str``.
    
    Returns:
        Tuple of (height, width, channels) in [H, W, C] format.
    
    Raises:
        ValueError: If image mode is unsupported for non-RAW images.
    """
    path = core.Path(path)
    if path.is_raw_image_file():
        image = rawpy.imread(str(path)).raw_image_visible
        h, w  = image.shape
        c     = 3
    else:
        with Image.open(str(path)) as image:
            w, h = image.size
            mode = image.mode
            c    = {"RGB": 3, "RGBA": 4, "L": 1}.get(mode, None)
            if c is None:
                raise ValueError(f"Unsupported image mode {mode}.")
    
    return h, w, c


def write_image(path: core.Path, image: torch.Tensor | np.ndarray):
    """Writes an image to a file path.

    Args:
        path: Output file path as ``core.Path`` or ``str``.
        image: Image as ``torch.Tensor`` [B, C, H, W] or ``numpy.ndarray`` [H, W, C].
    
    Raises:
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    path = core.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, torch.Tensor):
        torchvision.utils.save_image(image, str(path))
    elif isinstance(image, np.ndarray):
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")

# endregion


# region Normalize

def normalize_image_by_range(
    image  : torch.Tensor | np.ndarray,
    min    : float = 0.0,
    max    : float = 255.0,
    new_min: float = 0.0,
    new_max: float = 1.0
) -> torch.Tensor | np.ndarray:
    """Normalizes an image from range [min, max] to [new_min, new_max].

    Args:
        image: Image as ``torch.Tensor`` [B, C, H, W] or ``numpy.ndarray`` [H, W, C].
        min: Current minimum pixel value. Default is ``0.0``.
        max: Current maximum pixel value. Default is ``255.0``.
        new_min: New minimum pixel value. Default is ``0.0``.
        new_max: New maximum pixel value. Default is ``1.0``.
    
    Returns:
        Normalized image as ``torch.Tensor`` or ``numpy.ndarray``.
    
    Raises:
        ValueError: If ``image`` dimensions are less than 3.
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if not image.ndim >= 3:
        raise ValueError(f"[image]'s number of dimensions must be >= 3, got {image.ndim}.")
    
    ratio = (new_max - new_min) / (max - min)
    if isinstance(image, torch.Tensor):
        image = image.clone().to(dtype=torch.get_default_dtype())
    elif isinstance(image, np.ndarray):
        image = np.copy(image).astype(np.float32)
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
    image = (image - min) * ratio + new_min
    
    return image


normalize_image = functools.partial(
    normalize_image_by_range,
    min     = 0.0,
    max     = 255.0,
    new_min = 0.0,
    new_max = 1.0
)
denormalize_image = functools.partial(
    normalize_image_by_range,
    min     = 0.0,
    max     = 1.0,
    new_min = 0.0,
    new_max = 255.0
)

# endregion
