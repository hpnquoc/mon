#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model.

This module implements the base class for all deep learning models.
"""

from __future__ import annotations

__all__ = [
    "ExtraModel",
    "Model",
    "load_weights",
]

from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse  # noqa: F401

import humps
import lightning.pytorch.utilities.types
import torch.hub
from torch import nn

from mon import core
from mon.globals import (
    LOSSES, LR_SCHEDULERS, LType, METRICS, OPTIMIZERS, Task,
)
from mon.nn import loss as L, metric as M

console       = core.console
error_console = core.error_console
StepOutput    = lightning.pytorch.utilities.types.STEP_OUTPUT
EpochOutput   = Any  # lightning.pytorch.utilities.types.EPOCH_OUTPUT


# region Utils

def is_image(image: torch.Tensor) -> bool:
    """Checks if input is a valid image tensor.

    Args:
        image: Tensor to validate.
    
    Returns:
        ``True`` if image is valid, ``False`` otherwise.
    """
    from mon.vision.dtype import image as I
    return I.is_image(image)
    
# endregion


# region Weights

def load_weights(
    model       : nn.Module,
    weights     : dict | str | core.Path,
    weights_only: bool = False
) -> dict | None:
    """Loads state dict from given weights into model.

    Args:
        model: Model to load weights into.
        weights: Weights as dict, str path, or core.Path.
        weights_only: Load only weights if ``True``. Default is ``False``.
    
    Returns:
        State dict if loaded, ``None`` otherwise.
    """
    if weights is None:
        return None

    path       = core.Path(weights["path"]) if isinstance(weights, dict) and "path" in weights     else None
    state_dict = weights                    if isinstance(weights, dict) and "path" not in weights else None

    if isinstance(weights, (str, core.Path)) and core.Path(weights).is_weights_file():
        path = core.Path(weights)

    if path and path.is_weights_file():
        state_dict = torch.load(str(path), map_location=model.device, weights_only=weights_only)
    elif state_dict is None:
        core.error_console.log(f"[yellow]Cannot load weights from: {weights}[/yellow]")
        return None

    return state_dict["state_dict"] if "state_dict" in state_dict else state_dict

# endregion


# region Model

class Model(lightning.LightningModule, ABC):
    """The base class for all machine learning models.
    
    Attributes:
        arch: The model's architecture or family. Default: ``None`` mean it will
            be `self.__class__.__name__`.
        name: The model's name. Default: ``None`` mean it will be
            `self.__class__.__name__`.
        tasks: A list of tasks that the model can perform.
        ltypes: A list of learning schemes that the model can perform.
        model_dir: The model's directory. Default: ``None``.
        zoo: A `dict` containing all pretrained weights of the model.
        
    Args:
        root: The root directory of the model. It is used to save the model
            checkpoint during training: ``{root}/{fullname}``.
        fullname: The model's fullname to save the checkpoint or weights. It
            should have the following format: {name}-{dataset}-{suffix}.
            Default: ``None`` mean it will be the same as `name`.
        weights: The model's weights. Any of:
            - A state `dict`.
            - A key in the `zoo`. Ex: ``'yolov8x_det_coco'``.
            - A path to an ``.pt``, ``.pth``, or ``.ckpt`` file.
        optimizer: Optimizer(s) for a training model. Default: ``None``.
        loss: Loss function for training the model. Default: ``None``.
        metrics: A list metrics for training, validating and testing model.
            Default: ``None``.
        debug: Debug mode. Default: ``False``.
        verbose: Verbosity. Default: ``True``.
    
    Example:
        LOADING WEIGHTS

        Case 01: Pre-define the weights file in `zoo` directory. Pre-define
        the metadata in `zoo`. Then define `weights` as a key in
        `zoo`.
            >>> zoo = {
            >>>     "imagenet": {
            >>>         "url"        : "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
            >>>         "path"       : "vgg19-imagenet.pth",  # Locate in ``zoo`` directory
            >>>         "num_classes": 1000,
            >>>         "map": {}
            >>>     },
            >>> }
            >>>
            >>> model = Model(
            >>>     weights="imagenet",
            >>> )

        Case 02: Define the full path to an ``.pt``, ``.pth``, or ``.ckpt`` file.
            >>> model = Model(
            >>>     weights="home/workspace/.../vgg19-imagenet.pth",
            >>> )
    """
    
    arch     : str         = ""         # The model's architecture.
    name     : str         = ""         # The model's name.
    tasks    : list[Task]  = []         # A list of tasks that the model can perform.
    ltypes   : list[LType] = []         # A list of learning types that the model can perform.
    model_dir: core.Path   = None       
    zoo      : dict        = {}         # A dictionary containing all pretrained weights of the model.
    
    def __init__(
        self,
        # Basic
        root     : core.Path = core.Path(),
        fullname : str  = None,
        # Network
        weights  : Any  = None,
        # Training
        optimizer: Any  = None,
        loss     : Any  = None,
        metrics  : Any  = None,
        # Misc
        debug    : bool = False,
        verbose  : bool = True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Misc
        self.debug         = debug
        self.verbose       = verbose
        # Basic
        self.init_name()
        self.root          = root
        self.fullname      = fullname
        # Network
        self.weights       = None
        self.assign_weights(weights)
        # Training
        self.optimizer     = optimizer
        self.loss          = None
        self.train_metrics = None
        self.val_metrics   = None
        self.test_metrics  = None
        self.init_loss(loss)
        self.init_metrics(metrics)
        
    # region Properties
    
    @property
    def fullname(self) -> str:
        """Returns the model's full name as name-suffix."""
        return self._fullname
    
    @fullname.setter
    def fullname(self, fullname: str):
        """Sets the model's full name, defaults to name if invalid.
    
        Args:
            fullname: Full name to set.
        """
        self._fullname = fullname if fullname not in [None, "None", ""] else self.name
    
    @property
    def root(self) -> core.Path:
        """Returns the root directory path."""
        return self._root
    
    @root.setter
    def root(self, root: Any):
        """Sets the root directory and updates related paths.
    
        Args:
            root: Path input to set as root directory.
        """
        self._root      = core.Path(root)
        self._debug_dir = self._root / "debug"
        self._ckpt_dir  = self._root
    
    @property
    def ckpt_dir(self) -> core.Path:
        """Returns the checkpoint directory path."""
        if self._ckpt_dir is None:
            self._ckpt_dir = self.root
        return self._ckpt_dir
    
    @property
    def debug_dir(self) -> core.Path:
        """Returns the debug directory path."""
        if self._debug_dir is None:
            self._debug_dir = self.root / "debug"
        return self._debug_dir
    
    @property
    def predicting(self) -> bool:
        """Checks if model is in predicting mode.
    
        Returns:
            ``True`` if in predicting mode, ``False`` otherwise.
        
        Notes:
            True when not training and not managed by ``lightning.Trainer``.
        """
        return not self.training and getattr(self, "_trainer", None) is None
    
    @property
    def debug(self) -> bool:
        """Returns debug mode status.
    
        Returns:
            ``True`` if in debug mode, ``False`` otherwise.
        
        Notes:
            Returns ``_debug`` if predicting, else ``True``.
        """
        return self._debug if self.predicting else True
    
    @debug.setter
    def debug(self, debug: bool):
        """Sets debug mode.
    
        Args:
            debug: Debug mode value to set.
        """
        self._debug = debug
    
    # endregion
    
    # region Initialization
    
    def init_name(self):
        """Sets the model's name if not already defined."""
        if not self.name:
            self.name = humps.kebabize(self.__class__.__name__).lower()
    
    def create_dir(self):
        """Creates root, checkpoint, and debug directories."""
        for path in [self.root, self.ckpt_dir, self.debug_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def init_weights(self, m: nn.Module):
        """Initializes the model's weights.
    
        Args:
            m: Module to initialize weights for.
        """
        pass
    
    def assign_weights(self, weights: Any, overwrite: bool = False):
        """Assigns pretrained weights to the model.
    
        Args:
            weights: Weights as dict, str, or Path; ``None`` to skip.
            overwrite: Overwrite existing weights if ``True``. Default is ``False``.
        
        Raises:
            ValueError: If weights path is invalid.
        """
        if weights is None:
            return
    
        if isinstance(weights, dict):
            self.weights = weights
        elif isinstance(weights, str) and weights in self.zoo:
            weights_dict = self.zoo[weights]
            url  = weights_dict.get("url",  None)
            path = weights_dict.get("path", None)
            if url and path:
                core.download_weights_from_url(url, path, overwrite)
            self.weights = weights_dict
        elif isinstance(weights, (str, core.Path)):
            weights_path = core.Path(weights)
            if not weights_path.is_weights_file():
                raise ValueError(f"[weights] must be a valid path to a weight file, but got [{weights_path}]")
            state_dict = torch.load(str(weights_path))
            self.weights = {
                "url"        : None,
                "path"       : weights_path,
                "num_classes": state_dict.get("num_classes", None)
            }
        else:
            self.weights = weights or self.weights
        
    def load_weights(self, weights: Any = None, overwrite: bool = False):
        """Loads intersecting weights into the model.
    
        Args:
            weights: Weights to load; None uses existing. Default is None.
            overwrite: Overwrite existing weights if True. Default is False.
        """
        self.assign_weights(weights, overwrite)
        state_dict = load_weights(self, self.weights, weights_only=True)
        if state_dict:
            self.load_state_dict(state_dict)
            if self.verbose:
                console.log(f"Loaded model's weights from: {self.weights}")
        
    def init_loss(self, loss: Any):
        """Sets the model's loss function.
    
        Args:
            loss: Loss as str, dict, or object.
        """
        if isinstance(loss, str):
            self.loss = LOSSES.build(name=loss)
        elif isinstance(loss, dict):
            self.loss = LOSSES.build(config=loss)
        else:
            self.loss = loss
    
        if isinstance(self.loss, L.Loss):
            self.loss.requires_grad = True
            self.loss.eval()
    
    def init_metrics(self, metrics: Any):
        """Assign metrics.
        
        Args:
            metrics: One of the 2 options:
                - Common metrics for all train_/val_/test_metrics:
                     metrics: {name: "accuracy"}
                - Define train_/val_/test_metrics separately:
                    metrics:
                        train:
                            - {name: "accuracy"}
                            - ...
                        val  : {name: "accuracy"}
                        test : ~
        """
        # This is a simple hack since LightningModule needs the metric to be
        # defined with self.<metric>. So, here we dynamically add the metric
        # attribute to the class.
        train_metrics = metrics.get("train") if isinstance(metrics, dict) else metrics
        self.train_metrics = self.create_metrics(train_metrics)
        if self.train_metrics:
            for metric in self.train_metrics:
                setattr(self, f"train/{metric.name}", metric)
    
        val_metrics = metrics.get("val") if isinstance(metrics, dict) else metrics
        self.val_metrics = self.create_metrics(val_metrics)
        if self.val_metrics:
            for metric in self.val_metrics:
                setattr(self, f"val/{metric.name}", metric)
    
        test_metrics = metrics.get("test") if isinstance(metrics, dict) else metrics
        self.test_metrics = self.create_metrics(test_metrics)
        if self.test_metrics:
            for metric in self.test_metrics:
                setattr(self, f"test/{metric.name}", metric)
    
    @staticmethod
    def create_metrics(metrics: Any) -> list | None:
        """Creates metrics from various input types.
    
        Args:
            metrics: Metric object, dict, list, tuple, or ``None``.
        Returns:
            List of metric objects or ``None`` if invalid.
        """
        if isinstance(metrics, M.Metric):
            if getattr(metrics, "name", None) is None:
                metrics.name = humps.depascalize(humps.pascalize(metrics.__class__.__name__))
            return [metrics]
        if isinstance(metrics, dict):
            return [METRICS.build(config=metrics)]
        if isinstance(metrics, (list, tuple)):
            return [METRICS.build(config=m) if isinstance(m, dict) else m for m in metrics]
        return None
    
    def configure_optimizers(self):
        """Configures optimizers and LR schedulers for optimization.

        Returns:
            One of: dict, list, tuple, Optimizer, or ``None``; see Notes.
        
        Raises:
            ValueError: If optimizer or scheduler config is invalid.
        
        Notes:
            Options:
            - Single optimizer
            - List/tuple of optimizers
            - Two lists: [optimizers], [schedulers]
            - Dict with 'optimizer' and optional 'lr_scheduler'
            - Tuple of dicts with optional 'frequency'
            - None for no optimization
        
        Examples:
            def configure_optimizers(self):
                optimizer = Adam(...)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": ReduceLROnPlateau(optimizer, ...),
                        "monitor": "metric_to_track",
                        "frequency": "indicates how often the metric is updated",
                        # If "monitor" references validation metrics, then
                        # "frequency" should be set to a multiple of
                        # "trainer.check_val_every_n_epoch".
                    },
                }
        """
        if self.optimizer is None:
            return None
        if not isinstance(self.optimizer, dict):
            raise ValueError("[optimizer] must be a dict")
    
        optimizer_config    = self.optimizer.get("optimizer")
        lr_scheduler_config = self.optimizer.get("lr_scheduler")
        network_params_only = self.optimizer.get("network_params_only", True)
    
        if optimizer_config is None:
            raise ValueError("[optimizer] must be a dict")
        optimizer = OPTIMIZERS.build(
            network             = self,
            config              = optimizer_config,
            network_params_only = network_params_only
        )
    
        if lr_scheduler_config:
            scheduler_config = lr_scheduler_config.get("scheduler")
            if scheduler_config is None:
                raise ValueError("[scheduler] must be defined")
            lr_scheduler_config["scheduler"] = LR_SCHEDULERS.build(
                optimizer = optimizer,
                config    = scheduler_config
            )
    
        self.optimizer = {
            "optimizer"   : optimizer,
            "lr_scheduler": lr_scheduler_config
        }
        return self.optimizer
    
    def compute_efficiency_score(self, *args, **kwargs) -> tuple[float, float]:
        """Computes model efficiency score (FLOPs, params).
    
        Returns:
            Tuple of (FLOPs, parameter count) as floats.
        """
        core.error_console.log("[yellow]This method has not been implemented yet![/yellow]")
        return 0.0, 0.0
    
    # endregion
    
    # region Forward Pass
    
    @abstractmethod
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        """Computes forward pass and loss.
    
        Args:
            datapoint: Dict with datapoint attributes.
        
        Returns:
            Dict of predictions, must include 'loss' and 'pred' keys.
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, datapoint: dict, outputs: dict, metrics: list[M.Metric] | None = None) -> dict:
        """Computes metrics for given predictions.
    
        Args:
            datapoint: Dict with datapoint attributes.
            outputs: Dict with model predictions.
            metrics: List of metric functions or None. Default is None.
        
        Returns:
            Dict of computed metric values.
        """
        pass
    
    @abstractmethod
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        """Performs forward pass of the model.
    
        Args:
            datapoint: Dict with datapoint attributes.
       
        Returns:
            Dict of predictions, empty by default.
        """
        pass
    
    # endregion
    
    # region Training
    
    def on_fit_start(self):
        """Runs at the start of model fitting."""
        self.create_dir()

    def training_step(self, batch: dict, batch_idx: int, *args, **kwargs) -> StepOutput:
        """Computes training loss and metrics.
    
        Args:
            batch: Dict with datapoint attributes from ``DataLoader``.
            batch_idx: Index of the current batch.
        
        Returns:
            Loss tensor, dict with ``'loss'``, or ``None`` to skip.
        """
        outputs  = self.forward_loss(datapoint=batch, *args, **kwargs)
        outputs |= self.compute_metrics(
            datapoint = batch,
            outputs   = outputs,
            metrics   = self.train_metrics
        )
    
        log_values = {"step": self.current_epoch}
        log_values |= {
            f"train/{k}": v
            for k, v in outputs.items()
            if v is not None and not is_image(v)
        }
        self.log_dict(
            dictionary     = log_values,
            prog_bar       = False,
            logger         = True,
            on_step        = False,
            on_epoch       = True,
            sync_dist      = True,
            rank_zero_only = False
        )
    
        return outputs.get("loss", None)

    def on_train_epoch_end(self):
        """Resets training metrics at epoch end."""
        if self.train_metrics:
            for metric in self.train_metrics:
                metric.reset()

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> StepOutput:
        """Processes validation batch and computes metrics.
    
        Args:
            batch: Dict with datapoint attributes from ``DataLoader``.
            batch_idx: Index of the current batch.
        Returns:
            Loss tensor, dict with ``'loss'``, or ``None`` to skip.
        """
        outputs  = self.forward_loss(datapoint=batch, *args, **kwargs)
        outputs |= self.compute_metrics(
            datapoint = batch,
            outputs   = outputs,
            metrics   = self.val_metrics
        )
    
        log_values  = {"step": self.current_epoch}
        log_values |= {
            f"val/{k}": v
            for k, v in outputs.items()
            if v is not None and not is_image(v)
        }
        self.log_dict(
            dictionary     = log_values,
            prog_bar       = False,
            logger         = True,
            on_step        = False,
            on_epoch       = True,
            sync_dist      = True,
            rank_zero_only = False
        )
    
        if self.should_log_images():
            data = batch | {"outputs": outputs}
            self.log_images(
                epoch = self.current_epoch,
                step  = self.global_step,
                data  = data
            )
    
        return outputs.get("loss", None)
    
    def on_validation_epoch_end(self):
        """Resets validation metrics at epoch end."""
        if self.val_metrics:
            for metric in self.val_metrics:
                metric.reset()

    def on_test_start(self):
        """Runs at the start of model testing."""
        self.create_dir()

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> StepOutput:
        """Processes test batch and computes metrics.
    
        Args:
            batch: Dict with datapoint attributes from ``DataLoader``.
            batch_idx: Index of the current batch.
        
        Returns:
            Loss tensor, dict with ``'loss'``, or ``None`` to skip.
        """
        outputs  = self.forward_loss(datapoint=batch, *args, **kwargs)
        outputs |= self.compute_metrics(
            datapoint = batch,
            outputs   = outputs,
            metrics   = self.test_metrics
        )
    
        log_values = {"step": self.current_epoch}
        log_values |= {
            f"test/{k}": v
            for k, v in outputs.items()
            if v is not None and not is_image(v)
        }
        self.log_dict(
            dictionary     = log_values,
            prog_bar       = False,
            logger         = True,
            on_step        = False,
            on_epoch       = True,
            sync_dist      = True,
            rank_zero_only = False
        )
    
        if self.should_log_images():
            data = batch | {"outputs": outputs}
            self.log_images(
                epoch = self.current_epoch,
                step  = self.global_step,
                data  = data
            )
    
        return outputs.get("loss", None)
    
    def on_test_epoch_end(self):
        """Resets test metrics at epoch end."""
        if self.test_metrics:
            for metric in self.test_metrics:
                metric.reset()
    
    # endregion
    
    # region Predicting
    
    def infer(self, datapoint: dict, *args, **kwargs) -> dict:
        """Infers model output with optional processing.
    
        Args:
            datapoint: Dict with datapoint attributes.
        
        Returns:
            Dict of model predictions.
        
        Notes:
            Override for custom pre/post-processing; defaults to forward.
        """
        return self.forward(datapoint, *args, **kwargs)
    
    # endregion
    
    # region Exporting
    
    def export_to_onnx(
        self,
        input_dims   : list[int] = None,
        file_path    : core.Path = None,
        export_params: bool      = True
    ):
        """Exports the model to ONNX format.
    
        Args:
            input_dims: Input dimensions as ``[C, H, W]`` or ``None``.
            file_path: Save path or None to use root/fullname. Default is ``None``.
            export_params: Export parameters if ``True``. Default is ``True``.
       
        Raises:
            ValueError: If ``input_dims`` is undefined.
        """
        if not file_path:
            file_path = self.root / f"{self.fullname}.onnx"
        if ".onnx" not in str(file_path):
            file_path = core.Path(f"{file_path}.onnx")
    
        if not input_dims:
            raise ValueError("[input_dims] must be defined")
    
        input_sample = torch.randn(input_dims)
        self.to_onnx(
            file_path     = file_path,
            input_sample  = input_sample,
            export_params = export_params
        )
    
    def export_to_torchscript(
        self,
        input_dims: list[int] = None,
        file_path : core.Path = None,
        method    : str       = "script"
    ):
        """Exports the model to ``TorchScript`` format.
    
        Args:
            input_dims: Input dimensions or None.
            file_path: Save path or None to use root/fullname. Default is ``None``.
            method: Export method: ``'script'`` or ``'trace'``. Default is ``'script'``.
        
        Raises:
            ValueError: If ``input_dims`` is undefined.
        """
        if not file_path:
            file_path = self.root / f"{self.fullname}.pt"
        if ".pt" not in str(file_path):
            file_path = core.Path(f"{file_path}.pt")
    
        if not input_dims:
            raise ValueError("[input_dims] must be defined")
    
        input_sample = torch.randn(input_dims)
        script       = self.to_torchscript(method=method, example_inputs=input_sample)
        torch.jit.save(script, file_path)
    
    # endregion
    
    # region Logging
    
    def should_log_images(self) -> bool:
        """Checks if debug images should be logged.
    
        Returns:
            ``True`` if conditions for logging images are met, ``False`` otherwise.
        """
        log_image_every_n_epochs = getattr(self.trainer, "log_image_every_n_epochs", 0)
        return (
            self.trainer.is_global_zero
            and log_image_every_n_epochs > 0
            and self.current_epoch % log_image_every_n_epochs == 0
        )
    
    def log_images(
        self,
        epoch    : int,
        step     : int,
        data     : dict,
        extension: str = ".jpg"
    ):
        """Logs debug images to ``debug_dir``.
    
        Args:
            epoch: Current epoch number.
            step: Current step number.
            data: Dict with images to log.
            extension: Image file extension. Default is ``'.jpg'``.
        """
        pass
    
    # endregion
    
# endregion


# region Extra Model

class ExtraModel(Model, ABC):
    """Wraps a third-party model for mon integration.

    Args:
        model: Third-party model to wrap, named ``'model'``.
    
    Notes:
        Define architecture and load weights; train with original scripts.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: nn.Module = None
    
    def load_weights(self, weights: Any = None, overwrite: bool = False):
        """Loads intersecting weights into the wrapped model.

        Args:
            weights: Weights to load; ``None`` uses existing. Default is ``None``.
            overwrite: Overwrite existing weights if ``True``. Default is ``False``.
        """
        self.assign_weights(weights, overwrite)
        state_dict = load_weights(self, self.weights, weights_only=False)
        if state_dict:
            self.model.load_state_dict(state_dict=state_dict)
            if self.verbose:
                console.log(f"Loaded model's weights from: {self.weights}")

# endregion
