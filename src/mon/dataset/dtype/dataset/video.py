#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Video Dataset Template.

This module implements the templates for video-only datasets."""

from __future__ import annotations

__all__ = [
    "VideoLoader",
    "VideoLoaderCV",
]

from abc import ABC
from typing import Any

import albumentations as A
import cv2

from mon import core
from mon.dataset.dtype import annotation
from mon.dataset.dtype.dataset import base
from mon.globals import Split

console             = core.console
ClassLabels         = annotation.ClassLabels
DatapointAttributes = annotation.DatapointAttributes
FrameAnnotation     = annotation.FrameAnnotation
ImageAnnotation     = annotation.ImageAnnotation


# region Video Loader

class VideoLoader(base.Dataset, ABC):
    """Base class for video loaders.

    Attributes:
        datapoint_attrs: Dict of attribute names and types, must include ``'frame'``: ``FrameAnnotation``.

    Args:
        root: Path to a video file or stream.
        split: Data split to use. Default is ``Split.PREDICT``.
        transform: Transformations for input/target. Default is ``None``.
        to_tensor: If ``True``, converts to ``torch.Tensor``. Default is ``False``.
        cache_data: If ``True``, caches data to disk. Default is ``False``.
        verbose: If ``True``, enables verbose output. Default is ``True``.
    """
    
    datapoint_attrs = DatapointAttributes({
        "frame": FrameAnnotation,
    })
    
    def __init__(
        self,
        root       : core.Path,
        split      : Split     = Split.PREDICT,
        transform  : A.Compose = None,
        to_tensor  : bool      = False,
        cache_data : bool      = False,
        verbose    : bool      = True,
        *args, **kwargs
    ):
        self.num_frames = 0
        super().__init__(
            root        = root,
            split       = split,
            transform   = transform,
            to_tensor   = to_tensor,
            cache_data  = cache_data,
            verbose     = verbose,
            *args, **kwargs
        )
    
    # region Magic Methods
    
    def __getitem__(self, index: int) -> dict:
        """Gets a datapoint and metadata at the specified index.
    
        Args:
            index: Index of the datapoint.
    
        Returns:
            Dictionary with datapoint and metadata.
        """
        datapoint = self.get_datapoint(index=index)
        meta      = self.get_meta(index=index)
        
        if self.transform:
            main_attr      = self.main_attribute
            args           = {k: v for k, v in datapoint.items() if v is not None}
            args["image"]  = args.pop(main_attr)
            transformed    = self.transform(**args)
            transformed[main_attr] = transformed.pop("image")
            datapoint     |= transformed
        
        if self.to_tensor:
            for k, v in datapoint.items():
                to_tensor_fn = self.datapoint_attrs.get_tensor_fn(k)
                if to_tensor_fn and v is not None:
                    datapoint[k] = to_tensor_fn(
                        v         = v,
                        keepdim   = False,
                        normalize = True
                    )
        
        return datapoint | {"meta": meta}
    
    def __len__(self) -> int:
        """Gets the total number of frames.
    
        Returns:
            Number of frames in the video.
        """
        return self.num_frames
    
    # endregion
    
    def init_transform(self, transform: A.Compose | Any = None):
        """Initializes transformation operations.
    
        Args:
            transform: Transformations to apply. Default is ``None``.
        """
        super().init_transform(transform=transform)
        if self.transform:
            additional_targets = self.datapoint_attrs.albumentation_target_types()
            additional_targets.pop(self.main_attribute, None)
            additional_targets.pop("meta", None)
            self.transform.add_targets(additional_targets)
    
    def filter_data(self):
        """Filters unwanted datapoints."""
        pass
    
    def verify_data(self):
        """Verifies the dataset integrity.
    
        Raises:
            RuntimeError: If no datapoints exist.
        """
        if self.__len__() <= 0:
            raise RuntimeError("No datapoints in the dataset")
        if self.verbose:
            console.log(f"Number of {self.split_str} datapoints: {self.__len__()}")

# endregion


# region Video Loader CV

class VideoLoaderCV(VideoLoader):
    """Loads video frames from a file or stream using ``cv2``.

    Args:
        root: Path to a video file or stream.
        split: Data split to use. Default is ``Split.PREDICT``.
        transform: Transformations to apply. Default is ``None``.
        to_tensor: If ``True``, converts to ``torch.Tensor``. Default is ``False``.
        cache_data: If ``True``, caches data to disk. Default is ``False``.
        verbose: If ``True``, enables verbose output. Default is ``True``.
    """
    
    def __init__(
        self,
        root       : core.Path,
        split      : Split     = Split.PREDICT,
        transform  : A.Compose = None,
        to_tensor  : bool      = False,
        cache_data : bool      = False,
        verbose    : bool      = True,
        *args, **kwargs
    ):
        self.video_capture = None
        super().__init__(
            root        = root,
            split       = split,
            transform   = transform,
            to_tensor   = to_tensor,
            cache_data  = cache_data,
            verbose     = verbose,
            *args, **kwargs
        )
    
    # region Properties
    
    @property
    def is_stream(self) -> bool:
        """Checks if the input is a video stream.
    
        Returns:
            ``True`` if input is a stream, ``False`` otherwise.
        """
        return self.root.is_video_stream() or self.num_frames == -1
    
    @property
    def format(self):
        """Gets the format of Mat objects.
    
        Returns:
            Format code from ``VideoCapture.retrieve()``; -1 for RAW streams.
        """
        return self.video_capture.get(cv2.CAP_PROP_FORMAT)
    
    @property
    def fourcc(self) -> str:
        """Gets the 4-character codec code.
    
        Returns:
            FourCC code as a string.
        """
        return str(self.video_capture.get(cv2.CAP_PROP_FOURCC))
    
    @property
    def fps(self) -> int:
        """Gets the frame rate.
    
        Returns:
            Frames per second as an integer.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FPS))
    
    @property
    def frame_height(self) -> int:
        """Gets the height of video frames.
    
        Returns:
            Frame height in pixels as an integer.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    @property
    def frame_width(self) -> int:
        """Gets the width of video frames.
    
        Returns:
            Frame width in pixels as an integer.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    @property
    def shape(self) -> tuple[int, int, int]:
        """Gets the shape of video frames.
    
        Returns:
            Tuple of ``(height, width, channels)`` as integers.
        """
        return self.frame_height, self.frame_width, 3
    
    @property
    def imgsz(self) -> tuple[int, int]:
        """Gets the image size of video frames.
    
        Returns:
            Tuple of ``(height, width)`` as integers.
        """
        return self.frame_height, self.frame_width
    
    @property
    def mode(self):
        """Gets the current capture mode.
    
        Returns:
            Backend-specific mode value.
        """
        return self.video_capture.get(cv2.CAP_PROP_MODE)
    
    @property
    def pos_avi_ratio(self) -> int:
        """Gets the relative position in the video.
    
        Returns:
            Integer from ``0`` (start) to ``1`` (end).
        """
        return int(self.video_capture.get(cv2.CAP_PROP_POS_AVI_RATIO))
    
    @property
    def pos_msec(self) -> int:
        """Gets the current position in milliseconds.
    
        Returns:
            Position in milliseconds as an integer.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_POS_MSEC))
    
    @property
    def pos_frames(self) -> int:
        """Gets the next frame index.
    
        Returns:
            0-based index of the next frame as an integer.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
    
    # endregion
    
    # region Initialization
    
    def get_data(self):
        """Gets video data from the root path.
    
        Raises:
            IOError: If root path is not a valid video file or stream.
        """
        root = core.Path(self.root)
        if root.is_video_file():
            self.video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)
            num_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        elif root.is_video_stream():
            self.video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)
            num_frames = -1
        else:
            raise IOError(f"Invalid video source: {self.root}")
        
        if self.num_frames != num_frames:
            self.num_frames = num_frames
    
    def reset(self):
        """Resets the video loader."""
        self.index = 0
        if isinstance(self.video_capture, cv2.VideoCapture):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.index)
    
    def close(self):
        """Closes and releases the video capture."""
        if isinstance(self.video_capture, cv2.VideoCapture):
            self.video_capture.release()
    
    # endregion
    
    def get_datapoint(self, index: int) -> dict:
        """Gets a datapoint at the specified index.
    
        Args:
            index: Index of the datapoint.
    
        Returns:
            Dictionary containing the datapoint data.
    
        Raises:
            StopIteration: If index exceeds frame count for non-streams.
            RuntimeError: If ``video_capture`` is not initialized.
        """
        if not self.is_stream and self.index >= self.num_frames:
            self.close()
            raise StopIteration
        
        if isinstance(self.video_capture, cv2.VideoCapture):
            ret_val, frame = self.video_capture.read()
        else:
            raise RuntimeError("[video_capture] has not been initialized")
        
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = FrameAnnotation(index=self.index, frame=frame, path=self.root)
        self.index += 1
        
        datapoint = self.new_datapoint
        for k, v in self.datapoints.items():
            if k == self.main_attribute:
                datapoint[k] = frame.data
            elif v and v[index] and hasattr(v[index], "data"):
                datapoint[k] = v[index].data
        return datapoint
    
    def get_meta(self, index: int = 0) -> dict:
        """Gets metadata at the specified index.
    
        Args:
            index: Index of the metadata. Default is ``0``.
    
        Returns:
            Dictionary with metadata from the main attribute.
        """
        return {
            "format"       : self.format,
            "fourcc"       : self.fourcc,
            "fps"          : self.fps,
            "frame_height" : self.frame_height,
            "frame_width"  : self.frame_width,
            "hash"         : self.root.stat().st_size if isinstance(self.root, core.Path) else None,
            "image_size"   : (self.frame_height, self.frame_width),
            "imgsz"        : (self.frame_height, self.frame_width),
            "index"        : index,
            "mode"         : self.mode,
            "name"         : str(self.root.name),
            "num_frames"   : self.num_frames,
            "path"         : self.root,
            "pos_avi_ratio": self.pos_avi_ratio,
            "pos_frames"   : self.pos_frames,
            "pos_msec"     : self.pos_msec,
            "shape"        : (self.frame_height, self.frame_width, 3),
            "split"        : self.split_str,
            "stem"         : str(self.root.stem),
        }
        
# endregion
