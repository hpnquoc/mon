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
    from mon.vision.dtype import image as I
    return I.is_image(image)
    
# endregion


# region Weights

def load_weights(
    model       : nn.Module,
    weights     : dict | str | core.Path,
    weights_only: bool = False,
) -> dict | None:
    """Load state dict from the given ``weights``.

    Args:
        model: The model to load the weights into.
        weights: The weights to load. Can be a dictionary, a string path, or a
            ``core.Path`` object.
        weights_only: If ``True``, only load the weights. Default: ``False``.

    Returns:
        The state dictionary if loaded successfully, otherwise ``None``.
    """
    # First, check for ``None``.
    if weights is None:
        return None
    
    # Second, `weights` can be a dictionary.
    path       = core.Path(weights["path"]) if isinstance(weights, dict) and "path"     in weights else None
    state_dict = weights                    if isinstance(weights, dict) and "path" not in weights else None
    
    # Third, `weights` can be a path to a weight file.
    if isinstance(weights, (str, core.Path)) and core.Path(weights).is_weights_file():
        path = core.Path(weights)
    
    # Load state dict from path
    if path and path.is_weights_file():
        state_dict = torch.load(str(path), map_location=model.device, weights_only=weights_only)
    else:
        error_console.log(f"[yellow]Cannot load from weights from: {weights}!")
    
    # Check if the state_dict is nested
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict

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
        """Return the model's fullname = name-suffix"""
        return self._fullname
    
    @fullname.setter
    def fullname(self, fullname: str):
        """Specify the model's fullname. This value should only be defined once."""
        self._fullname = fullname if fullname not in [None, "None", ""] else self.name
    
    @property
    def root(self) -> core.Path:
        return self._root
    
    @root.setter
    def root(self, root: Any):
        self._root      = core.Path(root)
        self._debug_dir = self.root / "debug"
        self._ckpt_dir  = self.root
    
    @property
    def ckpt_dir(self) -> core.Path:
        if self._ckpt_dir is None:
            self._ckpt_dir = self.root
        return self._ckpt_dir
    
    @property
    def debug_dir(self) -> core.Path:
        if self._debug_dir is None:
            self._debug_dir = self.root / "debug"
        return self._debug_dir
    
    @property
    def predicting(self) -> bool:
        """Return ``True`` if the model is in predicting mode (not eval).
        
        This property is needed because, while in ``'validation'`` mode,
        ``training`` is also set to ``False``, so using
        ``self.training == False`` does not work.
        
        True ``'predicting'`` mode happens when ``_trainer`` is ``None``,
        i.e., not being handled by ``lightning.Trainer``.
        
        Returns:
            bool: ``True`` if the model is in predicting mode, ``False`` otherwise.
        """
        return not self.training and getattr(self, "_trainer", None) is None
    
    @property
    def debug(self) -> bool:
        """Return ``True`` if the model is in debug mode.
    
        This property checks if the model is in predicting mode. If it is,
        it returns the value of the ``_debug`` attribute. Otherwise, it always
        returns ``True``.
    
        Returns:
            bool: ``True`` if the model is in debug mode, ``False`` otherwise.
        """
        return self._debug if self.predicting else True
    
    @debug.setter
    def debug(self, debug: bool):
        """Set the debug mode."""
        self._debug = debug
    
    # endregion
    
    # region Initialization
    
    def init_name(self):
        """Specify the model's name. This value should only be defined once."""
        if not self.name:
            self.name = humps.kebabize(self.__class__.__name__).lower()
    
    def create_dir(self):
        """Create directories before training begins."""
        for path in [self.root, self.ckpt_dir, self.debug_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def init_weights(self, m: nn.Module):
        """Initialize the model's weights."""
        pass
    
    def assign_weights(self, weights: Any, overwrite: bool = False):
        """Assign pretrained weights to the model."""
        # First thing first, check if the `weights` is ``None``
        if weights is None:
            pass
        # Second, `weights` can be a `state_dict`.
        elif isinstance(weights, dict):
            pass
        # Third, `weights` can be a key in the `zoo` dictionary.
        elif isinstance(weights, str) and weights in self.zoo:
            weights: dict = self.zoo[weights]
            # Check if the weights' path exists and download if necessary.
            url  = weights.get("url",  None)
            path = weights.get("path", None)
            if url and path:
                core.download_weights_from_url(url, path, overwrite)
        # Fourth, `weights` can be a path to a weight file.
        elif isinstance(weights, (str, core.Path)):
            weights: core.Path = core.Path(weights)
            if not weights.is_weights_file():
                raise ValueError(f"`weights` must be a valid path to a weight file, but got {weights}.")
            # Load weights and check for `num_classes`.
            state_dict = torch.load(str(weights))
            weights    = {
                "url"        : None,
                "path"       : weights,
                "num_classes": state_dict.get("num_classes", None),
            }
        # OK! Done.
        self.weights = weights or self.weights
        
    def load_weights(self, weights: Any = None, overwrite: bool = False):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        # First assign new weights if it is valid.
        self.assign_weights(weights, overwrite)
        # Second, load state_dict
        state_dict = load_weights(self, self.weights, True)
        if state_dict:
            self.load_state_dict(state_dict)
            if self.verbose:
                console.log(f"Load model's weights from: {self.weights}!")
        
    def init_loss(self, loss: Any):
        """Specify the model's loss functions. This value should only be defined once."""
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
                    'metrics': {'name': 'accuracy'}
                  or,
                    'metrics': [{'name': 'accuracy'}, torchmetrics.Accuracy(), ...]
                
                - Define train_/val_/test_metrics separately:
                    'metrics': {
                        'train': ['name':'accuracy', torchmetrics.Accuracy(), ...],
                        'val':   torchmetrics.Accuracy(),
                        'test':  None,
                    }
        """
        # This is a simple hack since LightningModule needs the metric to be
        # defined with self.<metric>. So, here we dynamically add the metric
        # attribute to the class.
        
        # Train
        self.train_metrics = self.create_metrics(metrics.get("train") if isinstance(metrics, dict) else metrics)
        if self.train_metrics:
            for metric in self.train_metrics:
                setattr(self, f"train/{metric.name}", metric)
        
        # Val
        self.val_metrics = self.create_metrics(metrics.get("val") if isinstance(metrics, dict) else metrics)
        if self.val_metrics:
            for metric in self.val_metrics:
                setattr(self, f"val/{metric.name}", metric)
        
        # Test
        self.test_metrics = self.create_metrics(metrics.get("test") if isinstance(metrics, dict) else metrics)
        if self.test_metrics:
            for metric in self.test_metrics:
                setattr(self, f"test/{metric.name}", metric)
    
    @staticmethod
    def create_metrics(metrics: Any):
        """Create metrics."""
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
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally, you need one, but for GANs you might have
        multiple.
        
        Return:
            Any of these 6 options:
                - Single optimizer.
                - `list` or `tuple` of optimizers.
                - Two `list` - First `list` has multiple
                  optimizers, and the second has multiple LR schedulers (or
                  multiple lr_scheduler_config).
                - `dict`, with an ``'optimizer'`` key, and (optionally) a
                  ``'lr_scheduler'`` key whose value is a single LR scheduler or
                  lr_scheduler_config.
                - `tuple` of `dict` as described above, with an
                  optional ``'frequency'`` key.
                - ``None`` - Fit will run without any optimizer.
            
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
            
            def configure_optimizers(self):
                optimizer1 = Adam(...)
                optimizer2 = SGD(...)
                scheduler1 = ReduceLROnPlateau(optimizer1, ...)
                scheduler2 = LambdaLR(optimizer2, ...)
                return (
                    {
                        "optimizer": optimizer1,
                        "lr_scheduler": {
                            "scheduler": scheduler1,
                            "monitor": "metric_to_track",
                        },
                    },
                    {"optimizer": optimizer2, "lr_scheduler": scheduler2},
                )
        """
        if self.optimizer is None:
            return None
        if not isinstance(self.optimizer, dict):
            raise ValueError("`optimizer` must be a `dict`.")
        
        optimizer           = self.optimizer.get("optimizer")
        lr_scheduler        = self.optimizer.get("lr_scheduler")
        network_params_only = self.optimizer.get("network_params_only", True)
        
        # Define optimizer
        if optimizer is None:
            raise ValueError(f"`optimizer` must be a `dict`.")
        optimizer = OPTIMIZERS.build(
            network             = self,
            config              = optimizer,
            network_params_only = network_params_only
        )
        
        # Define learning rate scheduler
        if lr_scheduler:
            scheduler = lr_scheduler.get("scheduler")
            if scheduler is None:
                raise ValueError(f"`scheduler` must be defined.")
            else:
                lr_scheduler["scheduler"] = LR_SCHEDULERS.build(
                    optimizer = optimizer,
                    config    = scheduler
                )
        
        self.optimizer = {
            "optimizer"    : optimizer,
            "lr_scheduler" : lr_scheduler,
        }
        return self.optimizer
    
    def compute_efficiency_score(self, *args, **kwargs) -> tuple[float, float]:
        """Compute the efficiency score of the model, including FLOPs and number
        of parameters.
        """
        error_console.log(f"[yellow]This method has not been implemented yet!")
        return 0, 0
    
    # endregion
    
    # region Forward Pass
    
    @abstractmethod
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        """Forward pass, then compute the loss value.
        
        Args:
            datapoint: A `dict` containing the attributes of a datapoint.
            
        Return:
            A `dict` of all predictions with corresponding names. Note
            that the dictionary must contain the key ``'loss'`` and ``'pred'``.
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, datapoint: dict, outputs: dict, metrics: list[M.Metric] = None) -> dict:
        """Compute metrics.

        Args:
            datapoint: A `dict` containing the attributes of a datapoint.
            outputs: A `dict` containing all predictions.
            metrics: A list of metric functions to compute. Default: ``None``.
        """
        pass
    
    @abstractmethod
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        """Forward pass. This is the primary `forward` function of the
        model.
        
        Args:
            datapoint: A `dict` containing the attributes of a datapoint.
            
        Return:
            A `dict` of all predictions with corresponding names.
            Default: ``{}``.
        """
        pass
    
    # endregion
    
    # region Training
    
    def on_fit_start(self):
        """Called at the beginning of fit."""
        self.create_dir()

    def training_step(self, batch: dict, batch_idx: int, *args, **kwargs) -> StepOutput:
        """Here you compute and return the training loss, and some additional
        metrics for e.g., the progress bar or logger.
        
        Args:
            batch: The output of `~torch.utils.data.DataLoader`. It is a
                `dict` containing the attributes of a datapoint.
            batch_idx: An integer displaying index of this batch.
            
        Return:
            Any of:
                - The loss tensor.
                - A `dict`. Must include the key ``'loss'``.
                - ``None``, training will skip to the next batch.
        """
        # Forward
        outputs  = self.forward_loss(datapoint=batch, *args, **kwargs)
        outputs |= self.compute_metrics(
            datapoint = batch,
            outputs   = outputs,
            metrics   = self.train_metrics
        )
        # Log values
        log_values  = {"step": self.current_epoch}
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
            rank_zero_only = False,
        )
        # Return
        loss = outputs.get("loss", None)
        return loss

    def on_train_epoch_end(self):
        """Called in the training loop at the very end of the epoch."""
        if self.train_metrics:
            for metric in self.train_metrics:
                metric.reset()

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> StepOutput:
        """Operates on a single batch of data from the validation set. In this
        step, you might generate examples or calculate anything of interest like
        accuracy.
        
        Args:
            batch: The output of `~torch.utils.data.DataLoader`. It is a
                `dict` containing the attributes of a datapoint.
            batch_idx: An integer displaying index of this batch.
            
        Return:
            Any of:
                - The loss tensor.
                - A `dict`. Must include the key ``'loss'``.
                - ``None``, training will skip to the next batch.
        """
        # Forward
        outputs  = self.forward_loss(datapoint=batch, *args, **kwargs)
        outputs |= self.compute_metrics(
            datapoint = batch,
            outputs   = outputs,
            metrics   = self.val_metrics
        )
        # Log values
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
            rank_zero_only = False,
        )
        # Log images
        if self.should_log_images():
            data = batch | {"outputs": outputs}
            self.log_images(
                epoch = self.current_epoch,
                step  = self.global_step,
                data  = data,
            )
        # Return
        loss = outputs.get("loss", None)
        return loss
    
    def on_validation_epoch_end(self):
        """Called in the validation loop at the very end of the epoch."""
        if self.val_metrics:
            for metric in self.val_metrics:
                metric.reset()

    def on_test_start(self) -> None:
        """Called at the very beginning of testing."""
        self.create_dir()

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> StepOutput:
        """Operates on a single batch of data from the test set. In this step
        you'd normally generate examples or calculate anything of interest such
        as accuracy.

        Args:
            batch: The output of `~torch.utils.data.DataLoader`. It is a
                `dict` containing the attributes of a datapoint.
            batch_idx: An integer displaying index of this batch.
            
        Return:
            Any of:
                - The loss tensor.
                - A `dict`. Must include the key ``'loss'``.
                - ``None``, training will skip to the next batch.
        """
        # Forward
        outputs  = self.forward_loss(datapoint=batch, *args, **kwargs)
        outputs |= self.compute_metrics(
            datapoint = batch,
            outputs   = outputs,
            metrics   = self.test_metrics
        )
        # Log values
        log_values  = {"step": self.current_epoch}
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
            rank_zero_only = False,
        )
        # Log images
        if self.should_log_images():
            data = batch | {"outputs": outputs}
            self.log_images(
                epoch = self.current_epoch,
                step  = self.global_step,
                data  = data,
            )
        # Return
        loss = outputs.get("loss", None)
        return loss
    
    def on_test_epoch_end(self):
        """Called in the test loop at the very end of the epoch."""
        if self.test_metrics:
            for metric in self.test_metrics:
                metric.reset()
    
    # endregion
    
    # region Predicting
    
    def infer(self, datapoint: dict, *args, **kwargs) -> dict:
        """Infer the model on a single datapoint. This method is different from
        `forward()` in term that you may want to perform additional
        pre-processing or post-processing steps.
        
        Notes:
            If you want to perform specific pre-processing or post-processing
            steps, you should override this method.
        
        Args:
            datapoint: A `dict` containing the attributes of a datapoint.
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
        """Export the model to ``onnx`` format.

        Args:
            input_dims: Input dimensions in ``[C, H, W]`` format.
                Default: ``None``.
            file_path: Path to save the model. If ``None`` or empty, then save
                to `root`. Default: ``None``.
            export_params: Should export parameters? Default: ``True``.
        """
        if not file_path:
            file_path = self.root / f"{self.fullname}.onnx"
        if ".onnx" not in str(file_path):
            file_path = core.Path(str(file_path) + ".onnx")
        
        if not input_dims:
            raise ValueError("`input_dims` must be defined.")
        
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
        """Export the model to TorchScript format.

        Args:
            input_dims: Input dimensions. Default: ``None``.
            file_path: Path to save the model. If ``None`` or empty, then save
                to `root`. Default: ``None``.
            method: Whether to use TorchScript's `''script''` or ``'trace'``
                method. Default: ``'script'``.
        """
        if not file_path:
            file_path = self.root / f"{self.fullname}.pt"
        if ".pt" not in str(file_path):
            file_path = core.Path(str(file_path) + ".pt")
        
        if not input_dims:
            raise ValueError("`input_dims` must be defined.")
        
        input_sample = torch.randn(input_dims)
        script       = self.to_torchscript(method=method, example_inputs=input_sample)
        torch.jit.save(script, file_path)
    
    # endregion
    
    # region Logging
    
    def should_log_images(self) -> bool:
        """Check if we should save debug images."""
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
        """Log debug images to `debug_dir`.
        
        Args:
            epoch: The current epoch.
            step: The current step.
            data: A `dict` containing images to log.
            extension: The extension of the images. Default: ``'.jpg'``.
        """
        pass
    
    # endregion
    
# endregion


# region Extra Model

class ExtraModel(Model, ABC):
    """A wrapper model that wraps around another model defined in third-party
    source code. This is useful when we want to add the third-party models to
    `mon`'s models without reimplementing the entire model.
    
    Args:
        model: The model to wrap around. To make thing simple, we agree on
            the following naming convention: ``'model'``.
    
    Todo:
        Usually, we only need to define the model architecture and load the
        pretrained weights. The training should be performed using the original
        package's script.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: nn.Module = None
    
    def load_weights(self, weights: Any = None, overwrite: bool = False):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        # First assign new weights if it is valid.
        self.assign_weights(weights, overwrite)
        # Second, load state_dict
        state_dict = load_weights(self, self.weights, False)
        if state_dict:
            self.model.load_state_dict(state_dict=state_dict)
            if self.verbose:
                console.log(f"Load model's weights from: {self.weights}!")

# endregion
