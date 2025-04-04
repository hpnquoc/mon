#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements SICE datasets."""

__all__ = [
    "SICE",
    "SICEDataModule",
    "SICEGrad",
    "SICEGradDataModule",
    "SICEME",
    "SICEMEDataModule",
    "SICEMix",
    "SICEMixDataModule",
]

from typing import Literal

from mon import core, vision
from mon.constants import DATA_DIR, DATAMODULES, DATASETS, Split, Task

# ----- Alias -----
ClassLabels                    = core.ClassLabels
DatapointAttributes            = core.DatapointAttributes
DepthMapAnnotation             = vision.DepthMapAnnotation
ImageAnnotation                = vision.ImageAnnotation
SemanticSegmentationAnnotation = vision.SemanticSegmentationAnnotation
VisionDataset                  = vision.VisionDataset


# ----- Dataset -----
@DATASETS.register(name="sice")
class SICE(VisionDataset):
    """Loads SICE dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``DATA_DIR / "enhance"``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = DATA_DIR / "enhance", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "sice" if root.name != "sice" else root
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}.")
        super().__init__(root=root, *args, **kwargs)
    
    def list_data(self):
        patterns = [self.root / self.split_str / "image"]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images
        

@DATASETS.register(name="sice_grad")
class SICEGrad(VisionDataset):
    """Loads SICE-Grad dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``DATA_DIR / "enhance"``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = DATA_DIR / "enhance", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "sice_grad" if root.name != "sice_grad" else root
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}.")
        super().__init__(root=root, *args, **kwargs)
    
    def list_data(self):
        patterns = [self.root / self.split_str / "image"]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images


@DATASETS.register(name="sice_me")
class SICEME(VisionDataset):
    """Loads SICE-ME dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``DATA_DIR / "enhance"``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
        "depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = DATA_DIR / "enhance", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "sice_me" if root.name != "sice_me" else root
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}.")
        super().__init__(root=root, *args, **kwargs)
    
    def list_data(self):
        patterns = [self.root / self.split_str / "image"]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
    
        self.datapoints["image"] = images


@DATASETS.register(name="sice_mix")
class SICEMix(VisionDataset):
    """Loads SICE-Mix dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``DATA_DIR / "enhance"``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = DATA_DIR / "enhance", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "sice_mix" if root.name != "sice_mix" else root
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}.")
        super().__init__(root=root, *args, **kwargs)
    
    def list_data(self):
        patterns = [self.root / self.split_str / "image"]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images


# ----- DataModule -----
@DATAMODULES.register(name="sice")
class SICEDataModule(core.DataModule):
    """Configures SICE datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLIE]
    
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
            self.train = SICE(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = SICE(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = SICE(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="sice_grad")
class SICEGradDataModule(core.DataModule):
    """Configures SICEGrad datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLIE]
    
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
            self.train = SICEGrad(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = SICEGrad(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = SICEGrad(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="sice_me")
class SICEMEDataModule(core.DataModule):
    """Configures SICEME datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLIE]
    
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
            self.train = SICEME(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = SICEME(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = SICEME(split=Split.TEST,  **self.dataset_kwargs)
            
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="sice_mix")
class SICEMixDataModule(core.DataModule):
    """Configures SICEMix datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLIE]
    
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
            self.train = SICEMix(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = SICEMix(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = SICEMix(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
