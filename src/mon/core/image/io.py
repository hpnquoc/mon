#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image I/O.

This module implements the basic I/O functionalities of images.
"""

from __future__ import annotations

__all__ = [
    "read_image",
    "read_image_shape",
    "write_image",
    "write_image_cv",
    "write_image_torch",
    "write_images_cv",
    "write_images_torch",
]

import multiprocessing

import cv2
import joblib
import numpy as np
import rawpy
import torch
import torchvision
from PIL import Image

from mon.core import pathlib
from mon.core.image import photometry, utils


# region Read

def read_image(
    path     : pathlib.Path,
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
    path = pathlib.Path(path)
    # Read raw image
    if path.is_raw_image_file():
        image = rawpy.imread(str(path))
        image = image.postprocess()
    # Read other types of image
    else:
        image = cv2.imread(str(path), flags)  # BGR
        if image.ndim == 2:  # HW -> HW1 (OpenCV read grayscale image)
            image = np.expand_dims(image, axis=-1)
        if utils.is_color_image(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to tensor
    if to_tensor:
        image = utils.to_image_tensor(image, False, normalize)
    return image


def read_image_shape(path: pathlib.Path) -> tuple[int, int, int]:
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

# endregion


# region Write

def write_image(path: pathlib.Path, image: torch.Tensor | np.ndarray):
    """Write an image to a file path.
    
    Args:
        path: A path to write the image to.
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
    """
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
    dir_path   : pathlib.Path,
    name       : str,
    prefix     : str  = "",
    extension  : str  = ".png",
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
        image = utils.to_image_nparray(image, True, denormalize)
    image = utils.to_channel_last_image(image)
    if 2 <= image.ndim <= 3:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``2`` and ``3``, but got {image.ndim}.")
    # Write image
    dir_path  = pathlib.Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    name      = pathlib.Path(name)
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
    dir_path   : pathlib.Path,
    name       : str,
    prefix     : str  = "",
    extension  : str  = ".png",
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
        image = utils.to_channel_first_image(image)
    image = photometry.denormalize_image(image) if denormalize else image
    image = image.to(torch.uint8)
    image = image.cpu()
    if 2 <= image.ndim <= 3:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``2`` and ``3``, but got {image.ndim}.")
    # Write image
    dir_path  = pathlib.Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    name      = pathlib.Path(name)
    stem      = name.stem
    extension = extension  # name.suffix
    extension = f"{name.suffix}" if extension == "" else extension
    extension = f".{extension}" if "." not in extension else extension
    stem      = f"{prefix}_{stem}" if prefix != "" else stem
    name      = f"{stem}{extension}"
    file_path = dir_path / name
    if extension in [".jpg", ".jpeg"]:
        torchvision.io.image.write_jpeg(input=image, filename=str(file_path))
    elif extension in [".png"]:
        torchvision.io.image.write_png(input=image, filename=str(file_path))


def write_images_cv(
    images     : list[torch.Tensor | np.ndarray],
    dir_path   : pathlib.Path,
    names      : list[str],
    prefixes   : list[str] = "",
    extension  : str       = ".png",
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
    dir_path   : pathlib.Path,
    names      : list[str],
    prefixes   : list[str] = "",
    extension  : str       = ".png",
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
