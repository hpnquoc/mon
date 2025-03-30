#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements RESIDE datasets."""

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

from mon import core, vision
from mon.globals import DATA_DIR, DATAMODULES, DATASETS


@DATASETS.register(name="reside_hsts_real")
class RESIDE_HSTS_Real(vision.VisionDataset):
    """Loads RESIDE-HSTS-Real dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``DATA_DIR / "enhance"``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    tasks : list[core.Task]    = [core.Task.DEHAZE]
    splits: list[core.Split]   = [core.Split.TEST]
    datapoint_attrs            = vision.DatapointAttributes({
        "image": vision.ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = DATA_DIR / "enhance", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "reside" if root.name != "reside" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        """Populates ``datapoints`` with image annotations for split."""
        patterns = [self.root / "hsts" / "real" / self.split_str / "image"]
        
        images: list[vision.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(vision.ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images
        
        
@DATASETS.register(name="reside_hsts_synthetic")
class RESIDE_HSTS_Synthetic(vision.VisionDataset):
    """Loads RESIDE-HSTS-Synthetic dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``DATA_DIR / "enhance"``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    tasks : list[core.Task]  = [core.Task.DEHAZE]
    splits: list[core.Split] = [core.Split.TEST]
    datapoint_attrs     = vision.DatapointAttributes({
        "image"    : vision.ImageAnnotation,
        "ref_image": vision.ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = DATA_DIR / "enhance", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "reside" if root.name != "reside" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        """Populates ``datapoints`` with image annotations for split."""
        patterns = [self.root / "hsts" / "synthetic" / self.split_str / "image"]
        
        images: list[vision.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(vision.ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images
        

@DATASETS.register(name="reside_its")
class RESIDE_ITS(vision.VisionDataset):
    """Loads RESIDE-ITS dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``DATA_DIR / "enhance"``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    tasks : list[core.Task]  = [core.Task.DEHAZE]
    splits: list[core.Split] = [core.Split.TRAIN, core.Split.VAL]
    datapoint_attrs     = vision.DatapointAttributes({
        "image"    : vision.ImageAnnotation,
        "ref_image": vision.ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = DATA_DIR / "enhance", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "reside" if root.name != "reside" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        """Populates ``datapoints`` with image and ref annotations."""
        patterns = [self.root / "its" / self.split_str / "image"]
        
        images: list[vision.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(vision.ImageAnnotation(path=path, root=pattern))
        
        ref_images: list[vision.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                sequence=images,
                description=f"Listing {self.__class__.__name__} {self.split_str} reference images"
            ):
                stem = str(img.path.stem).split("_")[0]
                path = img.path.replace("/image/", "/ref/")
                path = path.parent / f"{stem}.{img.path.suffix}"
                ref_images.append(vision.ImageAnnotation(path=path.image_file(), root=pattern))
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images


@DATASETS.register(name="reside_ots")
class RESIDE_OTS(vision.VisionDataset):
    """Loads RESIDE-OTS dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``DATA_DIR / "enhance"``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    tasks : list[core.Task]  = [core.Task.DEHAZE]
    splits: list[core.Split] = [core.Split.TRAIN]
    datapoint_attrs     = vision.DatapointAttributes({
        "image"    : vision.ImageAnnotation,
        "ref_image": vision.ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = DATA_DIR / "enhance", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "reside" if root.name != "reside" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        """Populates ``datapoints`` with image and ref annotations."""
        patterns = [self.root / "ots" / self.split_str / "image"]
        
        images: list[vision.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(vision.ImageAnnotation(path=path, root=pattern))
        
        ref_images: list[vision.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                sequence=images,
                description=f"Listing {self.__class__.__name__} {self.split_str} reference images"
            ):
                stem = str(img.path.stem).split("_")[0]
                path = img.path.replace("/image/", "/ref/")
                path = path.parent / f"{stem}.{img.path.suffix}"
                ref_images.append(vision.ImageAnnotation(path=path.image_file(), root=pattern))
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images
        

@DATASETS.register(name="reside_rtts")
class RESIDE_RTTS(vision.VisionDataset):
    """Loads RESIDE-RTTS dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``DATA_DIR / "enhance"``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    tasks : list[core.Task]  = [core.Task.DEHAZE]
    splits: list[core.Split] = [core.Split.TEST]
    datapoint_attrs     = vision.DatapointAttributes({
        "image": vision.ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = DATA_DIR / "enhance", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "reside" if root.name != "reside" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        """Populates ``datapoints`` with image annotations for split."""
        patterns = [self.root / "rtts" / self.split_str / "image"]
        
        images: list[vision.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(vision.ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images
        

@DATASETS.register(name="reside_sots_indoor")
class RESIDE_SOTS_Indoor(vision.VisionDataset):
    """Loads RESIDE-SOTS-Indoor dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``DATA_DIR / "enhance"``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    tasks : list[core.Task]  = [core.Task.DEHAZE]
    splits: list[core.Split] = [core.Split.TEST]
    datapoint_attrs     = vision.DatapointAttributes({
        "image"    : vision.ImageAnnotation,
        "ref_image": vision.ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = DATA_DIR / "enhance", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "reside" if root.name != "reside" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        """Populates ``datapoints`` with image and ref annotations."""
        patterns = [self.root / "sots" / "indoor" / self.split_str / "image"]
        
        images: list[vision.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(vision.ImageAnnotation(path=path, root=pattern))
        
        ref_images: list[vision.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                sequence    = images,
                description = f"Listing {self.__class__.__name__} {self.split_str} reference images"
            ):
                stem = str(img.path.stem).split("_")[0]
                path = img.path.replace("/image/", "/ref/")
                path = path.parent / f"{stem}.{img.path.suffix}"
                ref_images.append(vision.ImageAnnotation(path=path.image_file(), root=pattern))
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images
        

@DATASETS.register(name="reside_sots_outdoor")
class RESIDE_SOTS_Outdoor(vision.VisionDataset):
    """Loads RESIDE-SOTS-Outdoor dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``DATA_DIR / "enhance"``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    tasks : list[core.Task]  = [core.Task.DEHAZE]
    splits: list[core.Split] = [core.Split.TEST]
    datapoint_attrs     = vision.DatapointAttributes({
        "image"    : vision.ImageAnnotation,
        "ref_image": vision.ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = DATA_DIR / "enhance", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "reside" if root.name != "reside" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        """Populates ``datapoints`` with image and ref annotations."""
        patterns = [self.root / "sots" / "outdoor" / self.split_str / "image"]
        
        images: list[vision.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(vision.ImageAnnotation(path=path, root=pattern))
        
        ref_images: list[vision.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                sequence    = images,
                description = f"Listing {self.__class__.__name__} {self.split_str} reference images"
            ):
                stem = str(img.path.stem).split("_")[0]
                path = img.path.replace("/image/", "/ref/")
                path = path.parent / f"{stem}.{img.path.suffix}"
                ref_images.append(vision.ImageAnnotation(path=path.image_file(), root=pattern))
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images


@DATASETS.register(name="reside_urhi")
class RESIDE_URHI(vision.VisionDataset):
    """Loads RESIDE-URHI dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``DATA_DIR / "enhance"``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    tasks : list[core.Task]  = [core.Task.DEHAZE]
    splits: list[core.Split] = [core.Split.TEST]
    datapoint_attrs     = vision.DatapointAttributes({
        "image": vision.ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = DATA_DIR / "enhance", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "reside" if root.name != "reside" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        """Populates ``datapoints`` with image annotations for split."""
        patterns = [self.root / "urhi" / self.split_str / "image"]
        
        images: list[vision.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(vision.ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images
        

@DATAMODULES.register(name="reside_hsts_real")
class RESIDE_HSTS_Real_DataModule(core.DataModule):
    """Configures RESIDE_HSTS_Real datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """

    tasks: list[core.Task] = [core.Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDE_HSTS_Real(split=core.Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDE_HSTS_Real(split=core.Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDE_HSTS_Real(split=core.Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_hsts_synthetic")
class RESIDE_HSTS_Synthetic_DataModule(core.DataModule):
    """Configures RESIDE_HSTS_Synthetic datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """

    tasks: list[core.Task] = [core.Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDE_HSTS_Synthetic(split=core.Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDE_HSTS_Synthetic(split=core.Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDE_HSTS_Synthetic(split=core.Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_its")
class RESIDE_ITS_DataModule(core.DataModule):
    """Configures RESIDE_ITS datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """

    tasks: list[core.Task] = [core.Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDE_ITS(split=core.Split.TRAIN, **self.dataset_kwargs)
            self.val   = RESIDE_ITS(split=core.Split.VAL, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDE_ITS(split=core.Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_ots")
class RESIDE_OTS_DataModule(core.DataModule):
    """Configures RESIDE_OTS datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """

    tasks: list[core.Task] = [core.Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDE_OTS(split=core.Split.TRAIN, **self.dataset_kwargs)
            self.val   = RESIDE_ITS(split=core.Split.VAL, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDE_ITS(split=core.Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_rtts")
class RESIDE_RTTS_DataModule(core.DataModule):
    """Configures RESIDE_RTTS datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """

    tasks: list[core.Task] = [core.Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDE_RTTS(split=core.Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDE_RTTS(split=core.Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDE_RTTS(split=core.Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_sots_indoor")
class RESIDE_SOTS_Indoor_DataModule(core.DataModule):
    """Configures RESIDE_SOTS_Indoor datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """

    tasks: list[core.Task] = [core.Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDE_SOTS_Indoor(split=core.Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDE_SOTS_Indoor(split=core.Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDE_SOTS_Indoor(split=core.Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_sots_outdoor")
class RESIDE_SOTS_Outdoor_DataModule(core.DataModule):
    """Configures RESIDE_SOTS_Outdoor datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """

    tasks: list[core.Task] = [core.Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDE_SOTS_Outdoor(split=core.Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDE_SOTS_Outdoor(split=core.Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDE_SOTS_Outdoor(split=core.Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_urhi")
class RESIDE_URHI_DataModule(core.DataModule):
    """Configures RESIDE_URHI datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """

    tasks: list[core.Task] = [core.Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDE_URHI(split=core.Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDE_URHI(split=core.Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDE_URHI(split=core.Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
