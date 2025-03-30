#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends ``enum.Enum`` class and defines custom enums."""

from __future__ import annotations

__all__ = [
    "AppleRGB",
    "BBoxFormat",
    "BasicRGB",
    "Enum",
    "LType",
    "MemoryUnit",
    "ShapeCode",
    "Split",
    "Task",
    "TrackState",
]

import enum
import random
from typing import Any


# region Enum

class Enum(enum.Enum):
    """Extension of Python ``enum.Enum`` with utility methods."""
    
    @classmethod
    def random(cls):
        """Returns a random enum member.

        Returns:
            Random member of the enum class.
        """
        return random.choice(list(cls))
    
    @classmethod
    def random_value(cls):
        """Returns a random enum value.

        Returns:
            Value of a random enum member.
        """
        return cls.random().value
    
    @classmethod
    def keys(cls) -> list['Enum']:
        """Returns all enum members.

        Returns:
            List of all enum members.
        """
        return list(cls)
    
    @classmethod
    def values(cls) -> list[Any]:
        """Returns all enum values.

        Returns:
            List of values from all enum members.
        """
        return [e.value for e in cls]

# endregion


# region Color

class RGB(Enum):
    """138 RGB colors."""
    
    MAROON                  = (128, 0  , 0)
    DARK_RED                = (139, 0  , 0)
    BROWN                   = (165, 42 , 42)
    FIREBRICK               = (178, 34 , 34)
    CRIMSON                 = (220, 20 , 60)
    RED                     = (255, 0  , 0)
    TOMATO                  = (255, 99 , 71)
    CORAL                   = (255, 127, 80)
    INDIAN_RED              = (205, 92 , 92)
    LIGHT_CORAL             = (240, 128, 128)
    DARK_SALMON             = (233, 150, 122)
    SALMON                  = (250, 128, 114)
    LIGHT_SALMON            = (255, 160, 122)
    ORANGE_RED              = (255, 69 , 0)
    DARK_ORANGE             = (255, 140, 0)
    ORANGE                  = (255, 165, 0)
    GOLD                    = (255, 215, 0)
    DARK_GOLDEN_ROD         = (184, 134, 11)
    GOLDEN_ROD              = (218, 165, 32)
    PALE_GOLDEN_ROD         = (238, 232, 170)
    DARK_KHAKI              = (189, 183, 107)
    KHAKI                   = (240, 230, 140)
    OLIVE                   = (128, 128, 0)
    YELLOW                  = (255, 255, 0)
    YELLOW_GREEN            = (154, 205, 50)
    DARK_OLIVE_GREEN        = (85 , 107, 47)
    OLIVE_DRAB              = (107, 142, 35)
    LAWN_GREEN              = (124, 252, 0)
    CHART_REUSE             = (127, 255, 0)
    GREEN_YELLOW            = (173, 255, 47)
    DARK_GREEN              = (0  , 100, 0)
    GREEN                   = (0  , 128, 0)
    FOREST_GREEN            = (34 , 139, 34)
    LIME                    = (0  , 255, 0)
    LIME_GREEN              = (50 , 205, 50)
    LIGHT_GREEN             = (144, 238, 144)
    PALE_GREEN              = (152, 251, 152)
    DARK_SEA_GREEN          = (143, 188, 143)
    MEDIUM_SPRING_GREEN     = (0  , 250, 154)
    SPRING_GREEN            = (0  , 255, 127)
    SEA_GREEN               = (46 , 139, 87)
    MEDIUM_AQUA_MARINE      = (102, 205, 170)
    MEDIUM_SEA_GREEN        = (60 , 179, 113)
    LIGHT_SEA_GREEN         = (32 , 178, 170)
    DARK_SLATE_GRAY         = (47 , 79 , 79)
    TEAL                    = (0  , 128, 128)
    DARK_CYAN               = (0  , 139, 139)
    AQUA                    = (0  , 255, 255)
    CYAN                    = (0  , 255, 255)
    LIGHT_CYAN              = (224, 255, 255)
    DARK_TURQUOISE          = (0  , 206, 209)
    TURQUOISE               = (64 , 224, 208)
    MEDIUM_TURQUOISE        = (72 , 209, 204)
    PALE_TURQUOISE          = (175, 238, 238)
    AQUA_MARINE             = (127, 255, 212)
    POWDER_BLUE             = (176, 224, 230)
    CADET_BLUE              = (95 , 158, 160)
    STEEL_BLUE              = (70 , 130, 180)
    CORN_FLOWER_BLUE        = (100, 149, 237)
    DEEP_SKY_BLUE           = (0  , 191, 255)
    DODGER_BLUE             = (30 , 144, 255)
    LIGHT_BLUE              = (173, 216, 230)
    SKY_BLUE                = (135, 206, 235)
    LIGHT_SKY_BLUE          = (135, 206, 250)
    MIDNIGHT_BLUE           = (25 , 25 , 112)
    NAVY                    = (0  , 0  , 128)
    DARK_BLUE               = (0  , 0  , 139)
    MEDIUM_BLUE             = (0  , 0  , 205)
    BLUE                    = (0  , 0  , 255)
    ROYAL_BLUE              = (65 , 105, 225)
    BLUE_VIOLET             = (138, 43 , 226)
    INDIGO                  = (75 , 0  , 130)
    DARK_SLATE_BLUE         = (72 , 61 , 139)
    SLATE_BLUE              = (106, 90 , 205)
    MEDIUM_SLATE_BLUE       = (123, 104, 238)
    MEDIUM_PURPLE           = (147, 112, 219)
    DARK_MAGENTA            = (139, 0  , 139)
    DARK_VIOLET             = (148, 0  , 211)
    DARK_ORCHID             = (153, 50 , 204)
    MEDIUM_ORCHID           = (186, 85 , 211)
    PURPLE                  = (128, 0  , 128)
    THISTLE                 = (216, 191, 216)
    PLUM                    = (221, 160, 221)
    VIOLET                  = (238, 130, 238)
    MAGENTA                 = (255, 0  , 255)
    ORCHID                  = (218, 112, 214)
    MEDIUM_VIOLET_RED       = (199, 21 , 133)
    PALE_VIOLET_RED         = (219, 112, 147)
    DEEP_PINK               = (255, 20 , 147)
    HOT_PINK                = (255, 105, 180)
    LIGHT_PINK              = (255, 182, 193)
    PINK                    = (255, 192, 203)
    ANTIQUE_WHITE           = (250, 235, 215)
    BEIGE                   = (245, 245, 220)
    BISQUE                  = (255, 228, 196)
    BLANCHED_ALMOND         = (255, 235, 205)
    WHEAT                   = (245, 222, 179)
    CORN_SILK               = (255, 248, 220)
    LEMON_CHIFFON           = (255, 250, 205)
    LIGHT_GOLDEN_ROD_YELLOW = (250, 250, 210)
    LIGHT_YELLOW            = (255, 255, 224)
    SADDLE_BROWN            = (139, 69 , 19)
    SIENNA                  = (160, 82 , 45)
    CHOCOLATE               = (210, 105, 30)
    PERU                    = (205, 133, 63)
    SANDY_BROWN             = (244, 164, 96)
    BURLY_WOOD              = (222, 184, 135)
    TAN                     = (210, 180, 140)
    ROSY_BROWN              = (188, 143, 143)
    MOCCASIN                = (255, 228, 181)
    NAVAJO_WHITE            = (255, 222, 173)
    PEACH_PUFF              = (255, 218, 185)
    MISTY_ROSE              = (255, 228, 225)
    LAVENDER_BLUSH          = (255, 240, 245)
    LINEN                   = (250, 240, 230)
    OLD_LACE                = (253, 245, 230)
    PAPAYA_WHIP             = (255, 239, 213)
    SEA_SHELL               = (255, 245, 238)
    MINT_CREAM              = (245, 255, 250)
    SLATE_GRAY              = (112, 128, 144)
    LIGHT_SLATE_GRAY        = (119, 136, 153)
    LIGHT_STEEL_BLUE        = (176, 196, 222)
    LAVENDER                = (230, 230, 250)
    FLORAL_WHITE            = (255, 250, 240)
    ALICE_BLUE              = (240, 248, 255)
    GHOST_WHITE             = (248, 248, 255)
    HONEYDEW                = (240, 255, 240)
    IVORY                   = (255, 255, 240)
    AZURE                   = (240, 255, 255)
    SNOW                    = (255, 250, 250)
    BLACK                   = (0  , 0  , 0)
    DIM_GRAY                = (105, 105, 105)
    GRAY                    = (128, 128, 128)
    DARK_GRAY               = (169, 169, 169)
    SILVER                  = (192, 192, 192)
    LIGHT_GRAY              = (211, 211, 211)
    GAINSBORO               = (220, 220, 220)
    WHITE_SMOKE             = (245, 245, 245)
    WHITE                   = (255, 255, 255)


class AppleRGB(Enum):
    """Apple's RGB colors."""
    
    BLACK       = (  0,   0,   0)
    BLUE        = (  0, 122, 255)
    BROWN       = (162, 132,  94)
    CYAN        = ( 50, 173, 230)
    GRAY        = (128, 128, 128)
    GRAY2       = (174, 174, 178)
    GRAY3       = (199, 199, 204)
    GRAY4       = (209, 209, 214)
    GRAY5       = (229, 229, 234)
    GRAY6       = (242, 242, 247)
    GREEN       = ( 52, 199,  89)
    INDIGO      = ( 85, 190, 240)
    MINT        = (  0, 199,  89)
    ORANGE      = (255, 149,   5)
    PINK        = (255,  45,  85)
    PURPLE      = ( 88,  86, 214)
    RED         = (255,  59,  48)
    TEAL        = ( 90, 200, 250)
    WHITE       = (255, 255, 255)
    YELLOW      = (255, 204,   0)
    DARK_BLUE   = (  0,  64, 221)
    DARK_BROWN  = (127, 101,  69)
    DARK_CYAN   = (  0, 113, 164)
    DARK_GRAY2  = ( 99,  99, 102)
    DARK_GRAY3  = ( 72,  72,  74)
    DARK_GRAY4  = ( 58,  58,  60)
    DARK_GRAY5  = ( 44,  44,  46)
    DARK_GRAY6  = ( 28,  28,  30)
    DARK_GREEN  = ( 36, 138,  61)
    DARK_INDIGO = ( 54,  52, 163)
    DARK_MINT   = ( 12, 129, 123)
    DARK_ORANGE = (201,  52,   0)
    DARK_PINK   = (211,  15,  69)
    DARK_PURPLE = (137,  68, 171)
    DARK_RED    = (255,  69,  58)
    DARK_TEAL   = (  0, 130, 153)
    DARK_YELLOW = (178,  80,   0)


class BasicRGB(Enum):
    """12 basic RGB colors."""
    
    BLACK   = (0  , 0  , 0)
    WHITE   = (255, 255, 255)
    RED     = (255, 0  , 0)
    LIME    = (0  , 255, 0)
    BLUE    = (0  , 0  , 255)
    YELLOW  = (255, 255, 0)
    CYAN    = (0  , 255, 255)
    MAGENTA = (255, 0  , 255)
    SILVER  = (192, 192, 192)
    GRAY    = (128, 128, 128)
    MAROON  = (128, 0  , 0)
    OLIVE   = (128, 128, 0)
    GREEN   = (0  , 128, 0)
    PURPLE  = (128, 0  , 128)
    TEAL    = (0  , 128, 128)
    NAVY    = (0  , 0  , 128)

# endregion


# region ML/DL

class LType(Enum):
    """Learning types."""
    
    INFERENCE    = "inference"          # Inference Only: we don't have training code.
    TRADITIONAL  = "traditional"        # Traditional Method (non-learning)
    SUPERVISED   = "supervised"         # Supervised Learning.
    UNSUPERVISED = "unsupervised"       # Unsupervised Learning.
    ZERO_SHOT    = "zero_shot"          # Zero-Shot Learning.
    
    @classmethod
    def trainable(cls) -> list[LType]:
        """Return a list of trainable learning types."""
        return [cls.SUPERVISED, cls.UNSUPERVISED]


class RunMode(Enum):
    """Run modes."""
    
    TRAIN   = "train"
    PREDICT = "predict"
    METRIC  = "metric"


class Split(Enum):
    """Dataset split types."""
    
    TRAIN   = "train"
    VAL     = "val"
    TEST    = "test"
    PREDICT = "predict"

# endregion


# region Vision

class BBoxFormat(Enum):
    """Bounding box formats.
    
    CX, CY: refers to a center of bounding box.
    W, H: refers to the width and height of bounding box.
    N: refers to the normalized value in the range [0.0, 1.0]:
        x_norm = absolute_x / image_width
        height_norm = absolute_height / image_height
    """
    
    XYXY    = "pascal_voc"
    XYWH    = "coco"
    CXCYWHN = "yolo"
    XYXYN   = "albumentations"
    VOC     = "pascal_voc"
    COCO    = "coco"
    YOLO    = "yolo"
    
    @classmethod
    def str_mapping(cls) -> dict[str, BBoxFormat]:
        """Returns a dictionary mapping string keys to ``BBoxFormat`` enum values.
    
        This method provides a mapping from string representations of bounding
        box formats to their corresponding ``BBoxFormat`` enum values. This is useful
        for converting string inputs to enum values in a consistent manner.
    
        Returns:
            A dictionary where the keys are string representations of bounding
            box formats and the values are the corresponding ``BBoxFormat`` enum values.
        """
        return {
            "xyxy"          : cls.XYXY,
            "xywh"          : cls.XYWH,
            "cxcyn"         : cls.CXCYWHN,
            "albumentations": cls.XYXYN,
            "pascal_voc"    : cls.VOC,
            "coco"          : cls.COCO,
            "yolo"          : cls.YOLO,
        }
    
    @classmethod
    def int_mapping(cls) -> dict[int, BBoxFormat]:
        """Returns a dictionary mapping integer keys to ``BBoxFormat`` enum values.

        This method provides a mapping from integer representations of bounding
        box formats to their corresponding ``BBoxFormat`` enum values. This is
        useful for converting integer inputs to enum values in a consistent manner.
    
        Returns:
            A dictionary where the keys are integer representations of bounding
            box formats and the values are the corresponding ``BBoxFormat`` enum values.
        """
        return {
            0: cls.XYXY,
            1: cls.XYWH,
            2: cls.CXCYWHN,
            3: cls.XYXYN,
            4: cls.VOC,
            5: cls.COCO,
            6: cls.YOLO,
        }
    
    @classmethod
    def from_str(cls, value: str) -> BBoxFormat:
        """Converts a string to a ``BBoxFormat`` enum.
    
        This method takes a string representation of a bounding box format and
        converts it to the corresponding ``BBoxFormat`` enum value. If the string
        is not a valid key in the mapping, a ValueError is raised.
    
        Args:
            value: The string representation of the bounding box format.
    
        Returns:
            The corresponding ``BBoxFormat`` enum value.
    
        Raises:
            ValueError: If the string is not a valid enum key.
        """
        value_lower = value.lower()
        if value_lower not in cls.str_mapping():
            raise ValueError(f"`value` must be a valid enum key, got {value_lower}.")
        return cls.str_mapping()[value_lower]
    
    @classmethod
    def from_int(cls, value: int) -> BBoxFormat:
        """Convert an integer to a ``BBoxFormat`` enum.
    
        This method takes an integer representation of a bounding box format and
        converts it to the corresponding ``BBoxFormat`` enum value. If the integer
        is not a valid key in the mapping, a ValueError is raised.
    
        Args:
            value: The integer representation of the bounding box format.
    
        Returns:
            The corresponding ``BBoxFormat`` enum value.
    
        Raises:
            ValueError: If the integer is not a valid enum key.
        """
        if value not in cls.int_mapping():
            raise ValueError(f"`value` must be a valid enum key, got {value}.")
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: Any) -> BBoxFormat | None:
        """Convert an arbitrary value to a ``BBoxFormat`` enum.
    
        This method takes an arbitrary value and attempts to convert it to a
        ``BBoxFormat`` enum. It supports conversion from ``BBoxFormat``, ``str``,
        and ``int`` types. If the value is not of a supported type, ``None`` is returned.
    
        Args:
            value: The value to convert to a ``BBoxFormat`` enum.
    
        Returns:
            The corresponding ``BBoxFormat`` enum value, or ``None`` if the value
            is not of a supported type.
        """
        if isinstance(value, BBoxFormat):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None


class ShapeCode(Enum):
    """Shape conversion code."""
    
    # Bounding box
    SAME       = 0
    XYXY2XYWH  = 1
    XYXY2CXCYN = 2
    XYWH2XYXY  = 3
    XYWH2CXCYN = 4
    CXCYN2XYXY = 5
    CXCYN2XYWH = 6
    VOC2COCO   = 7
    VOC2YOLO   = 8
    COCO2VOC   = 9
    COCO2YOLO  = 10
    YOLO2VOC   = 11
    YOLO2COCO  = 12
    
    @classmethod
    def str_mapping(cls) -> dict[str, ShapeCode]:
        """Returns a dictionary mapping string keys to ``ShapeCode`` enum values.
    
        This method provides a mapping from string representations of shape codes
        to their corresponding ``ShapeCode`` enum values. This is useful for converting
        string inputs to enum values in a consistent manner.
    
        Returns:
            A dictionary where the keys are string representations of shape
            codes and the values are the corresponding ``ShapeCode`` enum values.
        """
        return {
            "same"         : cls.SAME,
            "xyxy_to_xywh" : cls.XYXY2XYWH,
            "xyxy_to_cxcyn": cls.XYXY2CXCYN,
            "xywh_to_xyxy" : cls.XYWH2XYXY,
            "xywh_to_cxcyn": cls.XYWH2CXCYN,
            "cxcyn_to_xyxy": cls.CXCYN2XYXY,
            "cxcyn_to_xywh": cls.CXCYN2XYWH,
            "voc_to_coco"  : cls.VOC2COCO,
            "voc_to_yolo"  : cls.VOC2YOLO,
            "coco_to_voc"  : cls.COCO2VOC,
            "coco_to_yolo" : cls.COCO2YOLO,
            "yolo_to_voc"  : cls.YOLO2VOC,
            "yolo_to_coco" : cls.YOLO2COCO,
        }

    @classmethod
    def int_mapping(cls) -> dict[int, ShapeCode]:
        """Returns a dictionary mapping integer keys to ``ShapeCode`` enum values.
    
        This method provides a mapping from integer representations of shape codes
        to their corresponding ``ShapeCode`` enum values. This is useful for converting
        integer inputs to enum values in a consistent manner.
    
        Returns:
            A dictionary where the keys are integer representations of shape
            codes and the values are the corresponding ``ShapeCode`` enum values.
        """
        return {
            0 : cls.SAME,
            1 : cls.XYXY2XYWH,
            2 : cls.XYXY2CXCYN,
            3 : cls.XYWH2XYXY,
            4 : cls.XYWH2CXCYN,
            5 : cls.CXCYN2XYXY,
            6 : cls.CXCYN2XYWH,
            7 : cls.VOC2COCO,
            8 : cls.VOC2YOLO,
            9 : cls.COCO2VOC,
            10: cls.COCO2YOLO,
            11: cls.YOLO2VOC,
            12: cls.YOLO2COCO,
        }
    
    @classmethod
    def from_str(cls, value: str) -> ShapeCode:
        """Converts a string to a ``ShapeCode`` enum.
    
        This method takes a string representation of a shape code and converts it
        to the corresponding ``ShapeCode`` enum value. If the string is not a valid key
        in the mapping, a ValueError is raised.
    
        Args:
            value: The string representation of the shape code.
    
        Returns:
            The corresponding ``ShapeCode`` enum value.
    
        Raises:
            ValueError: If the string is not a valid enum key.
        """
        value_lower = value.lower()
        if value_lower not in cls.str_mapping():
            parts = value.split("_to_")
            if parts[0] == parts[1]:
                return cls.SAME
            raise ValueError(f"`value` must be a valid enum key, got {value_lower}.")
        return cls.str_mapping()[value_lower]

    @classmethod
    def from_int(cls, value: int) -> ShapeCode:
        """Convert an integer to a ``ShapeCode`` enum.
    
        This method takes an integer representation of a shape code and converts it
        to the corresponding ``ShapeCode`` enum value. If the integer is not a valid key
        in the mapping, a ValueError is raised.
    
        Args:
            value: The integer representation of the shape code.
    
        Returns:
            The corresponding ``ShapeCode`` enum value.
    
        Raises:
            ValueError: If the integer is not a valid enum key.
        """
        if value not in cls.int_mapping():
            raise ValueError(f"`value` must be a valid enum key, got {value}.")
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: Any) -> ShapeCode | None:
        """Convert an arbitrary value to a ``ShapeCode`` enum.
    
        This method takes an arbitrary value and attempts to convert it to a
        ``ShapeCode`` enum. It supports conversion from ``ShapeCode``, ``str``,
        and ``int`` types. If the value is not of a supported type, ``None`` is returned.
    
        Args:
            value: The value to convert to a ``ShapeCode`` enum.
    
        Returns:
            The corresponding ``ShapeCode`` enum value, or ``None`` if the value is not
            of a supported type.
        """
        if isinstance(value, ShapeCode):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None


class TrackState(Enum):
    """Enumeration type for a single target track state.
    
    Newly created tracks are classified as ``NEW`` until enough evidence has been
    collected. Then, the track state is changed to ``TRACKED``. Tracks that are no
    longer alive are classified as ``REMOVED`` to mark them for removal from the set of
    active tracks.
    """
    
    NEW      = 0
    TRACKED  = 1
    LOST     = 2
    REMOVED  = 3
    REPLACED = 4
    COUNTED  = 5
    
    @classmethod
    def str_mapping(cls) -> dict:
        """Return a dictionary mapping strings to ``TrackState`` enums.
    
        This method provides a mapping from string representations of track states
        to their corresponding ``TrackState`` enum values. This is useful for converting
        string inputs to enum values in a consistent manner.
    
        Returns:
            A dictionary where the keys are string representations of track
            states and the values are the corresponding ``TrackState`` enum values.
        """
        return {
            "new"     : cls.NEW,
            "tracked" : cls.TRACKED,
            "lost"    : cls.LOST,
            "removed" : cls.REMOVED,
            "replaced": cls.REPLACED,
            "counted" : cls.COUNTED,
        }
    
    @classmethod
    def int_mapping(cls) -> dict:
        """Return a dictionary mapping integers to ``TrackState`` enums.
    
        This method provides a mapping from integer representations of track
        states to their corresponding ``TrackState`` enum values. This is useful
        for converting integer inputs to enum values in a consistent manner.
    
        Returns:
            A dictionary where the keys are integer representations of track
            states and the values are the corresponding ``TrackState`` enum values.
        """
        return {
            0: cls.NEW,
            1: cls.TRACKED,
            2: cls.LOST,
            3: cls.REMOVED,
            4: cls.REPLACED,
            5: cls.COUNTED,
        }
    
    @classmethod
    def from_str(cls, value: str) -> "TrackState":
        """Convert a string to a ``TrackState`` enum.
        
        This method takes a string representation of a track state and converts
        it to the corresponding ``TrackState`` enum value. If the string is not
        a valid key in the mapping, a ValueError is raised.
    
        Args:
            value: The string representation of the track state.
    
        Returns:
            The corresponding ``TrackState`` enum value.
    
        Raises:
            ValueError: If the string is not a valid enum key.
        """
        value_lower = value.lower()
        if value_lower not in cls.str_mapping():
            raise ValueError(f"`value` must be a valid enum key, got {value_lower}.")
        return cls.str_mapping()[value_lower]
    
    @classmethod
    def from_int(cls, value: int) -> "TrackState":
        """Convert an integer to a ``TrackState`` enum.
    
        This method takes an integer representation of a track state and converts
        it to the corresponding ``TrackState`` enum value. If the integer is not
        a valid key in the mapping, a ValueError is raised.
    
        Args:
            value: The integer representation of the track state.
    
        Returns:
            The corresponding ``TrackState`` enum value.
    
        Raises:
            ValueError: If the integer is not a valid enum key.
        """
        if value not in cls.int_mapping():
            raise ValueError(f"`value` must be a valid enum key, got {value}.")
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: "TrackState" | str | int) -> TrackState | None:
        """Convert an arbitrary value to a ``TrackState`` enum.
    
        This method takes an arbitrary value and attempts to convert it to a
        ``TrackState`` enum. It supports conversion from ``TrackState``,
        ``str``, and ``int`` types. If the value is not of a supported type,
        ``None`` is returned.
    
        Args:
            value: The value to convert to a ``TrackState`` enum.
    
        Returns:
            The corresponding ``TrackState`` enum value, or ``None`` if the value
            is not of a supported type.
        """
        if isinstance(value, TrackState):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None

# endregion


# region Device

class MemoryUnit(Enum):
    """Memory units."""
    
    B  = "B"
    KB = "KB"
    MB = "MB"
    GB = "GB"
    TB = "TB"
    PB = "PB"
    
    @classmethod
    def str_mapping(cls) -> dict[str, MemoryUnit]:
        """Return a dictionary mapping strings to ``MemoryUnit`` enums.
    
        This method provides a mapping from string representations of memory units
        to their corresponding ``MemoryUnit`` enum values. This is useful for
        converting string inputs to enum values in a consistent manner.
    
        Returns:
            A dictionary where the keys are string representations of memory
            units and the values are the corresponding ``MemoryUnit`` enum values.
        """
        return {
            "b" : cls.B,
            "kb": cls.KB,
            "mb": cls.MB,
            "gb": cls.GB,
            "tb": cls.TB,
            "pb": cls.PB,
        }
    
    @classmethod
    def int_mapping(cls) -> dict[int, MemoryUnit]:
        """Return a dictionary mapping integers to ``MemoryUnit`` enums.
    
        This method provides a mapping from integer representations of memory units
        to their corresponding ``MemoryUnit`` enum values. This is useful for converting
        integer inputs to enum values in a consistent manner.
    
        Returns:
            A dictionary where the keys are integer representations of memory
            units and the values are the corresponding MemoryUnit enum values.
        """
        return {
            0: cls.B,
            1: cls.KB,
            2: cls.MB,
            3: cls.GB,
            4: cls.TB,
            5: cls.PB,
        }
    
    @classmethod
    def byte_conversion_mapping(cls) -> dict[MemoryUnit, int]:
        """Return a dictionary mapping memory units to their corresponding
        number of bytes.
    
        This method provides a mapping from ``MemoryUnit`` enum values to their
        corresponding number of bytes. This is useful for converting memory units
        to their byte equivalents in a consistent manner.
    
        Returns:
            A dictionary where the keys are ``MemoryUnit`` enum values and the
            values are the corresponding number of bytes.
        """
        return {
            cls.B : 1024 ** 0,
            cls.KB: 1024 ** 1,
            cls.MB: 1024 ** 2,
            cls.GB: 1024 ** 3,
            cls.TB: 1024 ** 4,
            cls.PB: 1024 ** 5,
        }
    
    @classmethod
    def from_str(cls, value: str) -> MemoryUnit:
        """Convert a string to a ``MemoryUnit`` enum.
    
        This method takes a string representation of a memory unit and converts
        it to the corresponding ``MemoryUnit`` enum value. If the string is not
        a valid key in the mapping, a ValueError is raised.
    
        Args:
            value: The string representation of the memory unit.
    
        Returns:
            The corresponding ``MemoryUnit`` enum value.
    
        Raises:
            ValueError: If the string is not a valid enum key.
        """
        value_lower = value.lower()
        if value_lower not in cls.str_mapping():
            raise ValueError(f"`value` must be a valid enum key, got {value_lower}.")
        return cls.str_mapping()[value_lower]
    
    @classmethod
    def from_int(cls, value: int) -> MemoryUnit:
        """Convert an integer to a ``MemoryUnit`` enum.
    
        This method takes an integer representation of a memory unit and converts
        it to the corresponding ``MemoryUnit`` enum value. If the integer is not
        a valid key in the mapping, a ValueError is raised.
    
        Args:
            value: The integer representation of the memory unit.
    
        Returns:
            The corresponding ``MemoryUnit`` enum value.
    
        Raises:
            ValueError: If the integer is not a valid enum key.
        """
        if value not in cls.int_mapping():
            raise ValueError(f"`value` must be a valid enum key, got {value}.")
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: Any) -> MemoryUnit | None:
        """Convert an arbitrary value to a ``MemoryUnit`` enum.
    
        This method attempts to convert an arbitrary value to a ``MemoryUnit`` enum.
        It supports conversion from ``MemoryUnit``, ``str``, and ``int`` types.
        If the value is not of a supported type, ``None`` is returned.
    
        Args:
            value: The value to convert to a ``MemoryUnit`` enum.
    
        Returns:
            The corresponding ``MemoryUnit`` enum value, or ``None`` if the value
            is not of a supported type.
        """
        if isinstance(value, MemoryUnit):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None

# endregion


class Task(Enum):
    """Task types."""
    
    CLASSIFY  = "classify"              # classification
    DEBLUR    = "deblur"                # deblurring
    DEHAZE    = "dehaze"                # dehazing
    DEMOIRE   = "demoire"               # demoireing
    DENOISE   = "denoise"               # denoising
    DEPTH     = "depth"                 # depth estimation
    DERAIN    = "derain"                # deraining
    DESNOW    = "desnow"                # desnowing
    DETECT    = "detect"                # object detection
    INPAINT   = "inpaint"               # inpainting
    LLIE      = "llie"                  # low-light image enhancement
    NIGHTTIME = "nighttime"             # nighttime
    POSE      = "pose"                  # pose estimation
    RETOUCH   = "retouch"               # image retouching
    SEGMENT   = "segment"               # semantic segmentation
    SR        = "sr"                    # super-resolution
    TRACK     = "track"                 # object tracking
    UIE       = "uie"                   # underwater image enhancement
