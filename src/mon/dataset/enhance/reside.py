#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RESIDE Datasets."""

from __future__ import annotations

__all__ = [
    "RESIDE_HSTS_Real",
    "RESIDE_HSTS_Real_DataModule",
    "RESIDE_HSTS_Synthetic",
    "RESIDE_HSTS_Synthetic_DataModule",
    "RESIDE_ITS",
    "RESIDE_ITS_DataModule",
    "RESIDE_OTS",
    "RESIDE_OTS_DataModule",
    "RESIDE_RTTS",
    "RESIDE_RTTS_DataModule",
    "RESIDE_SOTS_Indoor",
    "RESIDE_SOTS_Indoor_DataModule",
    "RESIDE_SOTS_Outdoor",
    "RESIDE_SOTS_Outdoor_DataModule",
    "RESIDE_URHI",
    "RESIDE_URHI_DataModule",
]

from typing import Literal

from mon import core
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "enhance"
DataModule          = core.DataModule
DatapointAttributes = core.DatapointAttributes
DepthMapAnnotation  = core.DepthMapAnnotation
ImageAnnotation     = core.ImageAnnotation
MultimodalDataset   = core.MultimodalDataset


@DATASETS.register(name="reside_hsts_real")
class RESIDE_HSTS_Real(MultimodalDataset):
    """RESIDE-HSTS-Real dataset consists of 10 real hazy images."""
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside" / "hsts" / "real" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images
        
        
@DATASETS.register(name="reside_hsts_synthetic")
class RESIDE_HSTS_Synthetic(MultimodalDataset):
    """RESIDE-HSTS-Synthetic dataset consists of ``10`` pairs of hazy and
    corresponding haze-free images.
    """
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside" / "hsts" / "synthetic" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images
        

@DATASETS.register(name="reside_its")
class RESIDE_ITS(MultimodalDataset):
    """RESIDE-ITS dataset consists of ``13,990`` pairs of hazy and corresponding
    haze-free images.
    """
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TRAIN, Split.VAL]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside" / "its" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = (list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        # Reference images
        ref_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                sequence    = images,
                description = f"Listing {self.__class__.__name__} {self.split_str} reference images"
            ):
                stem = str(img.path.stem).split("_")[0]
                path = img.path.replace("/image/", "/ref/")
                path = path.parent / f"{stem}.{img.path.suffix}"
                ref_images.append(ImageAnnotation(path=path.image_file(), root=pattern))
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images


@DATASETS.register(name="reside_ots")
class RESIDE_OTS(MultimodalDataset):
    """RESIDE-OTS dataset consists of ``73,135`` pairs of hazy and corresponding
    haze-free images.
    """
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TRAIN]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside" / "ots" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        # Reference images
        ref_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                sequence    = images,
                description = f"Listing {self.__class__.__name__} {self.split_str} reference images"
            ):
                stem = str(img.path.stem).split("_")[0]
                path = img.path.replace("/image/", "/ref/")
                path = path.parent / f"{stem}.{img.path.suffix}"
                ref_images.append(ImageAnnotation(path=path.image_file(), root=pattern))
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images
        

@DATASETS.register(name="reside_rtts")
class RESIDE_RTTS(MultimodalDataset):
    """RESIDE-RTTS dataset consists of ``4,322`` real hazy images."""
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside" / "rtts" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images
        

@DATASETS.register(name="reside_sots_indoor")
class RESIDE_SOTS_Indoor(MultimodalDataset):
    """RESIDE-SOTS-Indoor dataset consists of ``500`` pairs of hazy and
    corresponding haze-free images.
    """
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside" / "sots" / "indoor" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        # Reference images
        ref_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                sequence    = images,
                description = f"Listing {self.__class__.__name__} {self.split_str} hq images"
            ):
                stem = str(img.path.stem).split("_")[0]
                path = img.path.replace("/image/", "/ref/")
                path = path.parent / f"{stem}.{img.path.suffix}"
                ref_images.append(ImageAnnotation(path=path.image_file(), root=pattern))
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images
        

@DATASETS.register(name="reside_sots_outdoor")
class RESIDE_SOTS_Outdoor(MultimodalDataset):
    """RESIDE-SOTS-Outdoor dataset consists of ``500`` pairs of hazy and
    corresponding haze-free images.
    """
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside" / "sots" / "outdoor" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        # Reference images
        ref_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                sequence    = images,
                description = f"Listing {self.__class__.__name__} {self.split_str} reference images"
            ):
                stem = str(img.path.stem).split("_")[0]
                path = img.path.replace("/image/", "/ref/")
                path = path.parent / f"{stem}.{img.path.suffix}"
                ref_images.append(ImageAnnotation(path=path.image_file(), root=pattern))
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images


@DATASETS.register(name="reside_urhi")
class RESIDE_URHI(MultimodalDataset):
    """RESIDE-UHI dataset consists of ``4,809`` real hazy images."""
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside" / "urhi" / self.split_str / "image"
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images
        

@DATAMODULES.register(name="reside_hsts_real")
class RESIDE_HSTS_Real_DataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDE_HSTS_Real(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDE_HSTS_Real(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDE_HSTS_Real(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_hsts_synthetic")
class RESIDE_HSTS_Synthetic_DataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDE_HSTS_Synthetic(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDE_HSTS_Synthetic(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test = RESIDE_HSTS_Synthetic(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_its")
class RESIDE_ITS_DataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDE_ITS(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = RESIDE_ITS(split=Split.VAL, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDE_ITS(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_ots")
class RESIDE_OTS_DataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDE_OTS(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = RESIDE_ITS(split=Split.VAL, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDE_ITS(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_rtts")
class RESIDE_RTTS_DataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDE_RTTS(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDE_RTTS(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test = RESIDE_RTTS(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_sots_indoor")
class RESIDE_SOTS_Indoor_DataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDE_SOTS_Indoor(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDE_SOTS_Indoor(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test = RESIDE_SOTS_Indoor(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_sots_outdoor")
class RESIDE_SOTS_Outdoor_DataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDE_SOTS_Outdoor(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDE_SOTS_Outdoor(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDE_SOTS_Outdoor(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_urhi")
class RESIDE_URHI_DataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDE_URHI(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDE_URHI(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDE_URHI(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
