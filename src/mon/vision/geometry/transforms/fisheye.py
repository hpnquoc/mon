#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements fisheye transformation function for image and segmentation polygon --> bbox.

References:
    - https://github.com/Zane-Gu/AirEyeSeg
"""

__all__ = [
    "FishEyeGenerator",
    "FisheyeTransform"
]

import math
import random
from typing import Any

import cv2
import numpy as np

from mon import core
from mon.constants import TRANSFORMS
from mon.nn import _size_2_t, _size_3_t
from mon.vision import types
from mon.vision.geometry.transforms.albumentation import DualTransform, Targets, BaseModel, Field


# ----- Utils -----
def double_resolution(path: core.Path) -> np.ndarray | None:
    """Double Resolution (Only for Visdrone, before fisheye conversion)"""
    image = cv2.imread(str(path))
    if image is None:
        core.console.log(f"Invalid image path: {path}.")
        return None

    # Get the original image dimensions
    h0, w0   = types.image_size(image)
    # Double each dimension
    new_size = (2 * w0, 2 * h0)
    # Resize the image
    image    = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

    return image


# ----- Transformation Functions -----
class FishEyeGenerator:
    """Generates fisheye transformation for images."""

    def __init__(
        self,
        focal_len: int,
        shape    : tuple[int, int, int],
        bg_color : tuple[int, int, int] = (0, 0, 0),
        bg_label : int = 20
    ):
        self.shape     = shape
        self.ratio     = None
        self.focal_len = focal_len
        self.bg_color  = bg_color
        self.bg_label  = bg_label

        mask     = np.ones([self.shape[0], self.shape[1]], dtype=np.uint8)
        square_r = (min(self.shape[0], self.shape[1]) / 2) ** 2
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if ((i - self.shape[0] / 2) ** 2 + (j - self.shape[1] / 2) ** 2) >= square_r:
                    mask[i, j] = 0
        mask = np.array(mask)
        mask = mask.reshape(-1)
        self.bad_index = (mask == 0)

        # Parameters
        self.param         = 500
        self.alpha_range   = [0, 0]
        self.beta_range    = [0, 0]
        self.theta_range   = [0, 0]
        self.x_trans_range = [-self.shape[1] / 2, self.shape[1] / 2]
        self.y_trans_range = [-self.shape[0] / 2, self.shape[0] / 2]
        self.z_trans_range = [-0.6 * self.param ,  0.6 * self.param]
        self.alpha         = 0
        self.beta          = 0
        self.theta         = 0
        self.x_trans       = 0
        self.y_trans       = 0
        self.z_trans       = 0

    @property
    def focal_len(self) -> int:
        return self._focal_len

    @focal_len.setter
    def focal_len(self, focal_len: int):
        self._focal_len = focal_len
        self.ratio      = min(self.shape[0], self.shape[1]) / (self.focal_len * math.pi)

    def random_focal_len(self, focal_len_range: list[int] = [200, 400]):
        temp           = random.random()
        self.focal_len = focal_len_range[0] * (1 - temp) + focal_len_range[1] * temp
        print("focal len", self.focal_len)

    def set_ext_params_range(self, ext_params_range: list[int]):
        self.alpha_range   = [-ext_params_range[0] * math.pi / 180, ext_params_range[0] * math.pi / 180]
        self.beta_range    = [-ext_params_range[1] * math.pi / 180, ext_params_range[1] * math.pi / 180]
        self.theta_range   = [-ext_params_range[2] * math.pi / 180, ext_params_range[2] * math.pi / 180]
        self.x_trans_range = [-self.shape[1] * ext_params_range[3], self.shape[1] * ext_params_range[3]]
        self.y_trans_range = [-self.shape[0] * ext_params_range[4], self.shape[0] * ext_params_range[4]]
        self.z_trans_range = [-ext_params_range[5] * self.param   , ext_params_range[5] * self.param]

    def set_ext_params(self, ext_params: list[int]):
        self.alpha   = ext_params[0] * math.pi / 180
        self.beta    = ext_params[1] * math.pi / 180
        self.theta   = ext_params[2] * math.pi / 180
        self.x_trans = ext_params[3] * self.shape[1]
        self.y_trans = ext_params[4] * self.shape[0]
        self.z_trans = ext_params[5] * self.param

    def random_ext_params(self):
        temp         = random.random()
        self.alpha   = self.alpha_range[0] * (1 - temp) + self.alpha_range[1] * temp
        temp         = random.random()
        self.beta    = self.beta_range[0] * (1 - temp) + self.beta_range[1] * temp
        temp         = random.random()
        self.theta   = self.theta_range[0] * (1 - temp) + self.theta_range[1] * temp

        temp         = random.random()
        self.x_trans = self.x_trans_range[0] * (1 - temp) + self.x_trans_range[1] * temp
        temp         = random.random()
        self.y_trans = self.y_trans_range[0] * (1 - temp) + self.y_trans_range[1] * temp
        temp         = random.random()
        self.z_trans = self.z_trans_range[0] * (1 - temp) + self.z_trans_range[1] * temp

    def print_ext_param(self):
        print("alpha:         ", self.alpha * 180 / math.pi)
        print("beta:          ",  self.beta * 180 / math.pi)
        print("theta:         ", self.theta * 180 / math.pi)
        print("X translation: ", self.x_trans)
        print("Y translation: ", self.y_trans)
        print("Z translation: ", self.z_trans)

    def _calculate_coord_map(self, image: np.ndarray):
        self._init_ext_matrix()
        self._init_pin_matrix(image.shape)

        src_rows = image.shape[0]
        src_cols = image.shape[1]
        dst_rows = self.shape[0]
        dst_cols = self.shape[1]

        # 生成坐标矩阵
        cord_x, cord_y = np.meshgrid(np.arange(dst_cols), np.arange(dst_rows))
        cord = np.dstack((cord_x, cord_y)).astype(np.float32) - np.array([dst_cols / 2, dst_rows / 2])
        cord = cord.reshape(-1, 2)

        # shape=(dst_rows * dst_cols, 2)
        cord = np.array(cord) / self.ratio

        radius_array = np.sqrt(np.square(cord[:, 0]) + np.square(cord[:, 1]))
        theta_array = radius_array / self.focal_len

        new_x_array = np.tan(theta_array) * cord[:, 0] / radius_array * self.focal_len
        new_y_array = np.tan(theta_array) * cord[:, 1] / radius_array * self.focal_len

        temp_index1 = radius_array == 0
        temp_index2 = cord[:, 0] == 0
        temp_index3 = cord[:, 1] == 0
        bad_x_index = temp_index1 | (temp_index2 & temp_index1)
        bad_y_index = temp_index1 | (temp_index3 & temp_index1)

        new_x_array[bad_x_index] = 0
        new_y_array[bad_y_index] = 0

        new_x_array = new_x_array.reshape((-1, 1))
        new_y_array = new_y_array.reshape((-1, 1))

        new_cord = np.hstack((new_x_array, new_y_array))
        new_cord = np.hstack((new_cord, np.ones((dst_rows * dst_cols, 1)) * self.param))
        new_cord = np.hstack((new_cord, np.ones((dst_rows * dst_cols, 1))))

        # shape=(pix_num, 4)
        pin_camera_array = np.matmul(self.rotate_trans_matrix, new_cord.T).T

        # shape=(pix_num, 3)
        pin_image_cords = np.matmul(self.pin_matrix, pin_camera_array.T).T

        self.map_cols = pin_image_cords[:, 0] / pin_image_cords[:, 2]
        self.map_rows = pin_image_cords[:, 1] / pin_image_cords[:, 2]

        self.map_cols = self.map_cols.round().astype(int)
        self.map_rows = self.map_rows.round().astype(int)

        index1 = self.map_rows < 0
        index2 = self.map_rows >= src_rows
        index3 = self.map_cols < 0
        index4 = self.map_cols >= src_cols
        index5 = pin_image_cords[:, 2] <= 0

        bad_index = index1 | index2 | index3 | index4 | index5
        bad_index = bad_index | self.bad_index
        self.map_cols[bad_index] = image.shape[1]
        self.map_rows[bad_index] = 0

    def _init_ext_matrix(self):
        self.rotate_trans_matrix = \
            np.array([
                [
                    math.cos(self.beta) * math.cos(self.theta),
                    math.cos(self.beta) * math.sin(self.theta),
                    -math.sin(self.beta),
                     self.x_trans
                ],
                [
                    -math.cos(self.alpha) * math.sin(self.theta) + math.sin(self.alpha) * math.sin(self.beta) * math.cos(self.theta),
                    math.cos(self.alpha) * math.cos(self.theta) + math.sin(self.alpha) * math.sin(self.beta) * math.sin(self.theta),
                    math.sin(self.alpha) * math.cos(self.beta),
                     self.y_trans
                ],
                [
                    math.sin(self.alpha) * math.sin(self.theta) + math.cos(self.alpha) * math.sin(self.beta) * math.cos(self.theta),
                    -math.sin(self.alpha) * math.cos(self.theta) + math.cos(self.alpha) * math.sin(self.beta) * math.sin(self.theta),
                    math.cos(self.alpha) * math.cos(self.beta),
                     self.z_trans
                ],
                [0, 0, 0, 1]
            ])

    def _init_pin_matrix(self, image_shape: tuple[int, int, int]):
        rows = image_shape[0]
        cols = image_shape[1]
        self.pin_matrix = \
            np.array([
                [self.param, 0,          cols / 2, 0],
                [0,          self.param, rows / 2, 0],
                [0,          0,          1,        0]
            ])

    def transform_color_image(self, image: np.ndarray, reuse: bool = False) -> np.ndarray:
        if not reuse:
            self._calculate_coord_map(image)

        image = self._extend_image_color(image)
        image = np.array(image[(self.map_rows, self.map_cols)])
        image = image.reshape(self.shape[0], self.shape[1], 3)
        return image

    def transform_gray_image(self, image: np.ndarray, reuse: bool = False) -> np.ndarray:
        if not reuse:
            self._calculate_coord_map(image)

        image = self._extend_image_gray(image)
        image = np.array(image[(self.map_rows, self.map_cols)])
        image = image.reshape(self.shape[0], self.shape[1])
        return image

    def _extend_image_color(self, image: np.ndarray) -> np.ndarray:
        image = np.hstack((image, np.zeros((image.shape[0], 1, 3), dtype=np.uint8)))
        image[0, image.shape[1] - 1] = self.bg_color
        return image

    def _extend_image_gray(self, image: np.ndarray) -> np.ndarray:
        image = np.hstack((image, np.zeros((image.shape[0], 1), dtype=np.uint8)))
        image[0, image.shape[1] - 1] = self.bg_label
        return image


# ----- Augmentation -----
@TRANSFORMS.register(name="fisheye_transform")
class FisheyeTransform(DualTransform):
    """Apply fisheye transformation to inputs.

    Args:
        focal_len: Focal length of the fisheye lens. Default is ``150``.
        imgsz: Size of the output image. Default is ``(1280, 1280)``.
        bg_color: Background color for the fisheye image. Default is ``(0, 0, 0)``.
        bg_label: Background label for the fisheye image. Default is ``20``.
        reuse: Whether to reuse the transformation parameters. Default is ``False``.
        p: Probability of applying the transformation. Default is ``1.0``.
    """

    _targets = (Targets.IMAGE, Targets.MASK)

    class InitSchema(BaseModel):
        focal_len: int       = Field(ge=0, description="Focal length.")
        imgsz    : _size_2_t = (1280, 1280)
        bg_color : tuple     = (0, 0, 0)
        bg_label : int       = 20
        reuse    : bool      = False
        p        : float     = 1.0

    def __init__(
        self,
        focal_len: int       = 150,
        imgsz    : _size_2_t = (1280, 1280),
        bg_color : _size_3_t = (0, 0, 0),
        bg_label : int       = 20,
        reuse    : bool      = False,
        p        : float     = 1.0,
    ):
        super().__init__(p=p)
        self.focal_len = focal_len
        self.imgsz     = types.image_size(imgsz)
        self.ratio     = min(self.imgsz[0], self.imgsz[1]) / (self.focal_len * math.pi)
        self.bg_color  = bg_color
        self.bg_label  = bg_label
        self.reuse     = reuse

        # Mask for bad pixels
        mask     = np.ones([self.imgsz[0], self.imgsz[1]], dtype=np.uint8)
        square_r = (min(self.imgsz[0], self.imgsz[1]) / 2) ** 2
        for i in range(self.imgsz[0]):
            for j in range(self.imgsz[1]):
                if ((i - self.imgsz[0] / 2) ** 2 + (j - self.imgsz[1] / 2) ** 2) >= square_r:
                    mask[i, j] = 0
        mask = np.array(mask)
        mask = mask.reshape(-1)
        self.bad_index = (mask == 0)

        # Parameters
        self.param         = 500
        self.alpha_range   = [0, 0]
        self.beta_range    = [0, 0]
        self.theta_range   = [0, 0]
        self.x_trans_range = [-self.imgsz[1] / 2, self.imgsz[1] / 2]
        self.y_trans_range = [-self.imgsz[0] / 2, self.imgsz[0] / 2]
        self.z_trans_range = [-0.6 * self.param ,  0.6 * self.param]
        self.alpha         = 0
        self.beta          = 0
        self.theta         = 0
        self.x_trans       = 0
        self.y_trans       = 0
        self.z_trans       = 0

    # ----- Init -----
    def set_ext_params_range(self, ext_params_range: list[int]):
        """Set the range for external parameters."""
        self.alpha_range   = [-ext_params_range[0] * math.pi / 180, ext_params_range[0] * math.pi / 180]
        self.beta_range    = [-ext_params_range[1] * math.pi / 180, ext_params_range[1] * math.pi / 180]
        self.theta_range   = [-ext_params_range[2] * math.pi / 180, ext_params_range[2] * math.pi / 180]
        self.x_trans_range = [-self.imgsz[1] * ext_params_range[3], self.imgsz[1] * ext_params_range[3]]
        self.y_trans_range = [-self.imgsz[0] * ext_params_range[4], self.imgsz[0] * ext_params_range[4]]
        self.z_trans_range = [-ext_params_range[5] * self.param   , ext_params_range[5] * self.param]

    def set_ext_params(self, ext_params: list[int]):
        """Set the external parameters for fisheye transformation."""
        self.alpha   = ext_params[0] * math.pi / 180
        self.beta    = ext_params[1] * math.pi / 180
        self.theta   = ext_params[2] * math.pi / 180
        self.x_trans = ext_params[3] * self.imgsz[1]
        self.y_trans = ext_params[4] * self.imgsz[0]
        self.z_trans = ext_params[5] * self.param

    def random_focal_len(self, focal_len_range: tuple[int, int] = [200, 400]):
        """Randomly set the focal length within a specified range."""
        temp           = random.random()
        self.focal_len = focal_len_range[0] * (1 - temp) + focal_len_range[1] * temp
        # print("focal len", self.focal_len)

    def random_ext_params(self):
        """Randomly set the external parameters."""
        temp         = random.random()
        self.alpha   = self.alpha_range[0] * (1 - temp) + self.alpha_range[1] * temp
        temp         = random.random()
        self.beta    = self.beta_range[0] * (1 - temp) + self.beta_range[1] * temp
        temp         = random.random()
        self.theta   = self.theta_range[0] * (1 - temp) + self.theta_range[1] * temp

        temp         = random.random()
        self.x_trans = self.x_trans_range[0] * (1 - temp) + self.x_trans_range[1] * temp
        temp         = random.random()
        self.y_trans = self.y_trans_range[0] * (1 - temp) + self.y_trans_range[1] * temp
        temp         = random.random()
        self.z_trans = self.z_trans_range[0] * (1 - temp) + self.z_trans_range[1] * temp

    def print_ext_param(self):
        core.console.log(f"alpha:         {self.alpha * 180 / math.pi}.")
        core.console.log(f"beta:          { self.beta * 180 / math.pi}.")
        core.console.log(f"theta:         {self.theta * 180 / math.pi}.")
        core.console.log(f"X translation: {self.x_trans}.")
        core.console.log(f"Y translation: {self.y_trans}.")
        core.console.log(f"Z translation: {self.z_trans}.")

    # ----- Calculation -----
    def _calculate_coord_map(self, image: np.ndarray):
        self._init_ext_matrix()
        self._init_pin_matrix(image.shape)

        src_rows = image.shape[0]
        src_cols = image.shape[1]
        dst_rows = self.imgsz[0]
        dst_cols = self.imgsz[1]

        # 生成坐标矩阵
        cord_x, cord_y = np.meshgrid(np.arange(dst_cols), np.arange(dst_rows))
        cord = np.dstack((cord_x, cord_y)).astype(np.float32) - np.array([dst_cols / 2, dst_rows / 2])
        cord = cord.reshape(-1, 2)

        # shape=(dst_rows * dst_cols, 2)
        cord = np.array(cord) / self.ratio

        radius_array = np.sqrt(np.square(cord[:, 0]) + np.square(cord[:, 1]))
        theta_array  = radius_array / self.focal_len

        new_x_array  = np.tan(theta_array) * cord[:, 0] / radius_array * self.focal_len
        new_y_array  = np.tan(theta_array) * cord[:, 1] / radius_array * self.focal_len
                     
        temp_index1  = radius_array == 0
        temp_index2  = cord[:, 0] == 0
        temp_index3  = cord[:, 1] == 0
        bad_x_index  = temp_index1 | (temp_index2 & temp_index1)
        bad_y_index  = temp_index1 | (temp_index3 & temp_index1)

        new_x_array[bad_x_index] = 0
        new_y_array[bad_y_index] = 0

        new_x_array = new_x_array.reshape((-1, 1))
        new_y_array = new_y_array.reshape((-1, 1))

        new_cord = np.hstack((new_x_array, new_y_array))
        new_cord = np.hstack((new_cord, np.ones((dst_rows * dst_cols, 1)) * self.param))
        new_cord = np.hstack((new_cord, np.ones((dst_rows * dst_cols, 1))))

        # shape=(pix_num, 4)
        pin_camera_array = np.matmul(self.rotate_trans_matrix, new_cord.T).T

        # shape=(pix_num, 3)
        pin_image_cords  = np.matmul(self.pin_matrix, pin_camera_array.T).T

        self.map_cols = pin_image_cords[:, 0] / pin_image_cords[:, 2]
        self.map_rows = pin_image_cords[:, 1] / pin_image_cords[:, 2]

        self.map_cols = self.map_cols.round().astype(int)
        self.map_rows = self.map_rows.round().astype(int)

        index1 = self.map_rows < 0
        index2 = self.map_rows >= src_rows
        index3 = self.map_cols < 0
        index4 = self.map_cols >= src_cols
        index5 = pin_image_cords[:, 2] <= 0

        bad_index = index1 | index2 | index3 | index4 | index5
        bad_index = bad_index | self.bad_index
        self.map_cols[bad_index] = image.shape[1]
        self.map_rows[bad_index] = 0

    def _init_ext_matrix(self):
        self.rotate_trans_matrix = \
            np.array([
                [
                    math.cos(self.beta) * math.cos(self.theta),
                    math.cos(self.beta) * math.sin(self.theta),
                    -math.sin(self.beta),
                     self.x_trans
                ],
                [
                    -math.cos(self.alpha) * math.sin(self.theta) + math.sin(self.alpha) * math.sin(self.beta) * math.cos(self.theta),
                    math.cos(self.alpha) * math.cos(self.theta) + math.sin(self.alpha) * math.sin(self.beta) * math.sin(self.theta),
                    math.sin(self.alpha) * math.cos(self.beta),
                     self.y_trans
                ],
                [
                    math.sin(self.alpha) * math.sin(self.theta) + math.cos(self.alpha) * math.sin(self.beta) * math.cos(self.theta),
                    -math.sin(self.alpha) * math.cos(self.theta) + math.cos(self.alpha) * math.sin(self.beta) * math.sin(self.theta),
                    math.cos(self.alpha) * math.cos(self.beta),
                     self.z_trans
                ],
                [0, 0, 0, 1]
            ])

    def _init_pin_matrix(self, shape: tuple[int, int, int]):
        rows = shape[0]
        cols = shape[1]
        self.pin_matrix = \
            np.array([
                [self.param, 0,          cols / 2, 0],
                [0,          self.param, rows / 2, 0],
                [0,          0,          1,        0]
            ])

    def _extend_image_color(self, image: np.ndarray) -> np.ndarray:
        image = np.hstack((image, np.zeros((image.shape[0], 1, 3), dtype=np.uint8)))
        image[0, image.shape[1] - 1] = self.bg_color
        return image

    def _extend_image_gray(self, image: np.ndarray) -> np.ndarray:
        image = np.hstack((image, np.zeros((image.shape[0], 1), dtype=np.uint8)))
        image[0, image.shape[1] - 1] = self.bg_label
        return image

    def _transform_color_image(self, image: np.ndarray) -> np.ndarray:
        if not self.reuse:
            self._calculate_coord_map(image)

        fisheye_image = self._extend_image_color(image)
        fisheye_image = np.array(fisheye_image[(self.map_rows, self.map_cols)])
        fisheye_image = fisheye_image.reshape(self.imgsz[0], self.imgsz[1], 3)
        return fisheye_image

    def _transform_gray_image(self, image: np.ndarray) -> np.ndarray:
        if not self.reuse:
            self._calculate_coord_map(image)

        fisheye_image = self._extend_image_gray(image)
        fisheye_image = np.array(fisheye_image[(self.map_rows, self.map_cols)])
        fisheye_image = fisheye_image.reshape(self.imgsz[0], self.imgsz[1], 3)
        return fisheye_image

    # ----- Apply -----
    def apply(self, img: np.ndarray, old_size: _size_2_t, *args: Any, **params: Any) -> np.ndarray:
        return self._transform_color_image(img)

    def apply_to_mask(self, img: np.ndarray, old_size: _size_2_t, *args: Any, **params: Any) -> np.ndarray:
        return self._transform_gray_image(img)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Returns parameters dependent on input."""
        image  = data["image"]
        h0, w0 = types.image_size(image)
        return params | {
            "old_size": (w0, h0),
        }
