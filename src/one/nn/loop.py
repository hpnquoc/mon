#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Loop modules such as: Trainer, Tester, Inferrer.
"""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from timeit import default_timer as timer

import torch.cuda
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.accelerators import HPUAccelerator
from pytorch_lightning.accelerators import IPUAccelerator
from pytorch_lightning.accelerators import MPSAccelerator
from pytorch_lightning.utilities import _HPU_AVAILABLE
from pytorch_lightning.utilities import _IPU_AVAILABLE
from torch import Tensor

from one.data import *
from one.nn.model import BaseModel

# H1: - Inferrer ---------------------------------------------------------------

'''
def build_tensorrt_engine(
    onnx_file : Path_,
    batch_size: int = 1
):
    """
    Initialize TensorRT engine and parse ONNX model.

    Args:
        onnx_file (Path_):
        batch_size (int):
    """
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)

    trt_logger = trt.Logger()
    builder    = trt.Builder(trt_logger)
    network    = builder.create_network()
    parser     = trt.OnnxParser(network, trt_logger)

    with open(str(onnx_file), "rb") as model:
        console.log("Beginning ONNX file parsing.")
        parser.parse(model.read())
    console.log("Completed parsing of ONNX file.")

    # Allow TensorRT to use up to 1GB of GPU memory for tactic selection
    builder.max_workspace_size = 1 << 30
    # We have only one image in batch
    builder.max_batch_size     = batch_size
    # Use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True

    # Generate TensorRT engine optimized for the target platform
    console.log("Building an engine...")
    engine  = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    console.log("Completed creating Engine.")
    return engine, context
'''


# H2: - Base Inferrer ----------------------------------------------------------

class Inferrer(metaclass=ABCMeta):
    """
    Base inference pipeline.
    """
    
    def __init__(
        self,
        source     : Path_ | None = None,
        root       : Path_ | None = RUNS_DIR / "infer",
        project    : str          = "",
        name       : str          = "exp",
        max_samples: int   | None = None,
        batch_size : int          = 1,
        shape      : Ints  | None = None,
        device     : int   | str  = "cpu",
        phase      : ModelPhase_  = "training",
        tensorrt   : bool         = True,
        save       : bool         = True,
        verbose    : bool         = True,
        *args, **kwargs
    ):
        self.source = source
        self.root = root
        self.project = project
        self.shape = shape
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.device = select_device(device=device, batch_size=batch_size)
        self.phase = phase
        self.tensorrt = tensorrt
        self.save = save
        self.verbose = verbose
        
        self.model: BaseModel | None = None
        self.data_loader = None
        self.data_writer = None
        self.logger = None
        
        if self.project is not None and self.project != "":
            self.root = self.root / self.project
        self.name = f"{name}-{get_next_version(str(self.root), name)}"
        self.output_dir = self.root / self.name
        
        console.log(f"Using: {self.device}.")
    
    @property
    def phase(self) -> ModelPhase:
        return self._phase
    
    @phase.setter
    def phase(self, phase: ModelPhase_ = "training"):
        """
        Assign the model's running phase.
        """
        self._phase = ModelPhase.from_value(phase)
    
    @property
    def root(self) -> Path:
        return self._root
    
    @root.setter
    def root(self, root: Path_ | None):
        """
        Assign the root directory of the model.

        Args:
            root (Path_ | None): The root directory to save the results.
        """
        if root is None:
            root = RUNS_DIR / "infer"
        else:
            root = Path(root)
        self._root = root
    
    @abstractmethod
    def init_data_loader(self):
        """
        Initialize data loader.
        """
        pass
    
    @abstractmethod
    def init_data_writer(self):
        """
        Initialize data writer.
        """
        pass
    
    def init_logger(self):
        self.logger = open(self.output_dir / "log.txt", "a", encoding="utf-8")
        self.logger.write(
            f"\n================================================================================\n"
            )
        self.logger.write(
            f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n\n"
            )
        self.logger.flush()
    
    @abstractmethod
    def run(self, model: BaseModel, source: Path_):
        """

        Args:
            model (BaseModel): Model.
            source (Path_): Data source.
        """
        pass
    
    @abstractmethod
    def preprocess(self, input: Tensor):
        """
        Preprocessing input.

        Args:
            input (Tensor): Input of shape [B, C, H, W].

        Returns:
            input (Tensor): Processed input image as  [B, C H, W].
        """
        pass
    
    @abstractmethod
    def postprocess(
        self,
        input: Tensor,
        pred : Tensor,
        *args, **kwargs
    ) -> Tensor:
        """
        Postprocessing prediction.

        Args:
            input (Tensor): Input of shape [B, C, H, W].
            pred (Tensor): Prediction of shape [B, C, H, W].

        Returns:
            results (Tensor): Results of shape [B, C, H, W].
        """
        pass
    
    def on_run_start(self):
        """
        Call before `run()` starts.
        """
        if self.save:
            create_dirs(paths=[self.output_dir], recreate=True)
        
        self.init_data_loader()
        self.init_data_writer()
        self.init_logger()
        
        self.model.phase = self.phase
        self.model.to(self.device)
    
    @abstractmethod
    def on_run_end(self):
        """
        Call after `run()` finishes.
        """
        pass


# H2: - Vision Inferrer --------------------------------------------------------

class VisionInferrer(Inferrer):
    """
    Online vision inference pipeline.
    """
    
    def __init__(
        self,
        source     : Path_ | None = None,
        root       : Path_ | None = RUNS_DIR / "infer",
        project    : str          = "",
        name       : str          = "exp",
        max_samples: int   | None = None,
        batch_size : int          = 1,
        shape      : Ints  | None = None,
        device     : int   | str  = "cpu",
        phase      : ModelPhase_  = "training",
        tensorrt   : bool         = True,
        save       : bool         = True,
        verbose    : bool         = True,
        *args, **kwargs
    ):
        super().__init__(
            source=source,
            root=root,
            project=project,
            name=name,
            max_samples=max_samples,
            batch_size=batch_size,
            shape=shape,
            device=device,
            phase=phase,
            tensorrt=tensorrt,
            save=save,
            verbose=verbose,
            *args, **kwargs
        )
    
    def init_data_loader(self):
        """
        Initialize data loader.
        """
        import one.vision.acquisition as io
        if isinstance(self.source, (DataLoader, DataModule)):
            pass
        elif is_image_file(self.source) or is_dir(self.source):
            self.data_loader = io.ImageLoader(
                source=self.source,
                max_samples=self.max_samples,
                batch_size=self.batch_size,
            )
        elif is_video_file(self.source):
            self.data_loader = io.VideoLoaderCV(
                source=self.source,
                max_samples=self.max_samples,
                batch_size=self.batch_size,
            )
        else:
            raise RuntimeError()
    
    def init_data_writer(self):
        """
        Initialize data writer.
        """
        import one.vision.acquisition as io
        if self.save:
            if is_image_file(self.source) or is_dir(self.source):
                self.data_writer = io.ImageWriter(dst=self.output_dir)
            elif is_video_file(self.source) \
                and isinstance(self.data_loader, io.VideoLoaderCV):
                self.data_writer = io.VideoWriterCV(
                    dst=self.output_dir,
                    shape=self.data_loader.shape,
                    frame_rate=30,
                )
            else:
                raise RuntimeError()
    
    def run(self, model: BaseModel, source: Path_):
        self.model = model
        self.source = source
        self.on_run_start()
        
        # Setup online learning
        if self.phase == ModelPhase.TRAINING:
            optimizer = torch.optim.SGD(
                params=model.parameters(),
                lr=0.01,
                weight_decay=0.01,
            )
        else:
            optimizer = None
        
        # Print info
        self.logger.write(f"{'Model':<22}: {model.name}\n")
        self.logger.write(f"{'Data':<22}: {model.fullname}\n")
        if hasattr(model, "params"):
            self.logger.write(f"{'Parameters':<22}: {model.params}\n")
        # if self.shape is not None and is_sequence_of_length(self.shape, 3):
        #     self.logger.write(f"{'MACs':<21}: {model.macs(self.shape)}\n")
        self.logger.write(f"{'Device':<22}: {self.model.device}\n")
        self.logger.write(f"{'TensorRT':<22}: {self.tensorrt}\n")
        self.logger.write(f"{'Image Size':<22}: {self.shape}\n")
        self.logger.flush()
        
        step_times = []
        used_memory = []
        start_time = timer()
        with progress_bar() as pbar:
            for batch_idx, batch in pbar.track(
                enumerate(self.data_loader),
                total=int(len(self.data_loader) / self.batch_size),
                description=f"[bright_yellow] Processing"
            ):
                # Frame capture
                images, indexes, files, rel_paths = batch
                
                # Pre-process
                input, size0, size1 = self.preprocess(images)
                
                # Process
                step_start_time = timer()
                if model.phase == ModelPhase.TRAINING:
                    pred, loss = self.model.forward_loss(
                        input=input, target=None
                        )
                else:
                    pred, loss = self.model.forward(input=input), None
                
                if torch.cuda.is_available():
                    total, used, free = get_gpu_memory()
                    used_memory.append(used)
                
                step_end_time = timer()
                step_times.append(step_end_time - step_start_time)
                
                # Online learning
                if optimizer is not None and loss is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                
                # Post-process
                pred = self.postprocess(
                    input=input,
                    pred=pred,
                    size0=size0,
                    size1=size1,
                )
                
                # Debug
                if self.verbose:
                    self.model.show_results(
                        input=images,
                        pred=pred,
                        max_n=self.batch_size,
                        nrow=self.batch_size,
                        save=False,
                    )
                if self.save:
                    self.data_writer.write_batch(
                        images=pred,
                        files=rel_paths,
                        denormalize=True,
                    )
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        
        end_time = timer()
        console.log(
            f"{'Used Memory':<22}: "
            f"{(sum(used_memory) / len(used_memory)):.9f} GB"
        )
        console.log(
            f"{'Total Time':<22}: {(end_time - start_time):.9f} seconds"
        )
        console.log(
            f"{'Average Time':<22}: "
            f"{((end_time - start_time) / len(step_times)):.9f} seconds"
        )
        console.log(
            f"{'Average Time (forward)':<22}: "
            f"{(sum(step_times) / len(step_times)):.9f} seconds"
        )
        
        self.logger.write(
            f"{'Used Memory':<22}: "
            f"{(sum(used_memory) / len(used_memory)):.9f} GB\n"
        )
        self.logger.write(
            f"{'Total Time':<22}: {(end_time - start_time):.9f} seconds\n"
        )
        self.logger.write(
            f"{'Average Time':<22}: "
            f"{((end_time - start_time) / len(step_times)):.9f} seconds\n"
        )
        self.logger.write(
            f"{'Average Time (forward)':<22}: "
            f"{(sum(step_times) / len(step_times)):.9f} seconds\n"
        )
        self.logger.flush()
        self.logger.close()
        self.on_run_end()
    
    def preprocess(self, input: Tensor) -> tuple[Tensor, Ints, Ints]:
        """
        Preprocessing input.

        Args:
            input (Tensor): Input of shape [B, C, H, W].

        Returns:
            input (Tensor): Processed input image as  [B, C H, W].
        	size0 (Ints): The original images' sizes.
            size1 (Ints): The resized images' sizes.
        """
        from one.vision.acquisition import get_image_size
        from one.vision.transformation import resize
        
        size0 = get_image_size(input)
        if self.shape:
            new_size = to_size(self.shape)
            if size0 != new_size:
                input = resize(
                    image=input,
                    size=new_size,
                    interpolation=InterpolationMode.BICUBIC
                )
            # images = [resize(i, self.shape) for i in images]
            # images = torch.stack(input)
        size1 = get_image_size(input)
        
        input = input.to(self.device)
        return input, size0, size1
    
    # noinspection PyMethodOverriding
    def postprocess(
        self,
        input: Tensor,
        pred : Tensor,
        size0: Ints,
        size1: Ints,
    ) -> Tensor:
        """
        Postprocessing prediction.

        Args:
            input (Tensor): Input of shape [B, C, H, W].
            pred (Tensor): Prediction of shape [B, C, H, W].
            size0 (Ints): The original images' sizes.
            size1 (Ints): The resized images' sizes.

        Returns:
            pred (Tensor): Results of shape [B, C, H, W].
        """
        from one.vision.transformation import resize
        
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        
        if size0 != size1:
            pred = pred if isinstance(pred, Tensor) else torch.from_numpy(pred)
            pred = resize(
                image=pred,
                size=size0,
                interpolation=InterpolationMode.BICUBIC
            )
        return pred
    
    def on_run_end(self):
        """
        Call after `run()` finishes.
        """
        if self.verbose:
            cv2.destroyAllWindows()
        if self.save and self.data_writer:
            self.data_writer.close()


# H1: - Trainer ----------------------------------------------------------------

class Trainer(pl.Trainer):
    """
    Override `pytorch_lightning.Trainer` with several methods and properties.

    Args:
        accelerator (str | Accelerator): Supports passing different accelerator
            types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps, "auto") as well as
            custom accelerator instances.
        accumulate_grad_batches (int | dict | None): Accumulates grads every k
            batches or as set up in the dict. Defaults to None.
        amp_backend (str): The mixed precision backend to use ("native" or
            "apex"). Defaults to "native".
        amp_level (str | None): The optimization level to use (O1, O2, etc...).
            By default it will be set to "O2" if `amp_backend='apex'`.
        auto_lr_find (bool | str): If set to True, will make `trainer.tune()`
            run a learning rate finder, trying to optimize initial learning for
            faster convergence. `trainer.tune()` method will set the suggested
            learning rate in `self.lr` or `self.learning_rate` in the
            `LightningModule`. To use a different key set a string instead of
            True with the key name. Defaults to False.
        auto_scale_batch_size (bool | str): If set to True, will `initially`
            run a batch size finder trying to find the largest batch size that
            fits into memory. The result will be stored in `self.batch_size` in
            the `LightningModule`. Additionally, can be set to either `power`
            that estimates the batch size through a power search or `binsearch`
            that estimates the batch size through a binary search.
            Defaults to False.
        auto_select_gpus (bool): If enabled and `gpus` or `devices` is an
            integer, pick available gpus automatically. This is especially
            useful when GPUs are configured to be in "exclusive mode", such
            that only one process at a time can access them. Defaults to False.
        benchmark (bool | None): The value (True or False) to set
            `torch.backends.cudnn.benchmark` to. The value for
            `torch.backends.cudnn.benchmark` set in the current session will be
            used (False if not manually set). If `deterministic` is set to True,
            this will default to False. Override to manually set a different
            value. Defaults to None.
        callbacks (list[Callback] | Callback | None): Add a callback or list of
            callbacks. Defaults to None.
        check_val_every_n_epoch (int | None): Perform a validation loop every
            after every n training epochs. If None, validation will be done
            solely based on the number of training batches, requiring
            `val_check_interval` to be an integer value. Defaults to 1.
        default_root_dir (str | None): Default path for logs and weights when
            no logger/ckpt_callback passed. Can be remote file paths such as
            `s3://mybucket/path` or 'hdfs://path/'. Defaults to os.getcwd().
        detect_anomaly (bool): Enable anomaly detection for the autograd engine.
            Defaults to False.
        deterministic (bool | str | None): If True, sets whether PyTorch
            operations must use deterministic algorithms. Set to "warn" to use
            deterministic algorithms whenever possible, throwing warnings on
            operations that don't support deterministic mode (requires PyTorch
            1.11+). If not set, defaults to False. Defaults to None.
        devices (list | int | str | None): Will be mapped to either gpus,
            tpu_cores, num_processes or ipus, based on the accelerator type.
        enable_checkpointing (bool): If True, enable checkpointing. It will
            configure a default `ModelCheckpoint` callback if there is no
            user-defined `ModelCheckpoint` in `callbacks`. Defaults to True.
        enable_model_summary (bool): Whether to enable model summarization by
            default. Defaults to True.
        enable_progress_bar (bool): Whether to enable to progress bar by
            default. Defaults to True.
        fast_dev_run (int | bool): Runs n if set to `n` (int) else 1 if set to
            True batch(es) of train, val and test to find any bugs (ie: a sort
            of unit test). Defaults to False.
        gradient_clip_val (int | float | None): The value at which to clip
            gradients. Passing `gradient_clip_val=None` disables gradient
            clipping. If using Automatic Mixed Precision (AMP), the gradients
            will be unscaled before. Defaults to None.
        gradient_clip_algorithm (str | None): The gradient clipping algorithm
            to use. Pass `gradient_clip_algorithm="value"` to clip by value,
            and `gradient_clip_algorithm="norm"` to clip by norm. By default,
            it will be set to "norm".
        limit_train_batches (int | float | None): How much of training dataset
            to check (float = fraction, int = num_batches). Defaults to 1.0.
        limit_val_batches (int | float | None): How much of validation dataset
            to check (float = fraction, int = num_batches). Defaults to 1.0.
        limit_test_batches (int | float | None): How much of test dataset to
            check (float = fraction, int = num_batches). Defaults to 1.0.
        limit_predict_batches (int | float | None): How much of prediction
            dataset to check (float = fraction, int = num_batches).
            Defaults to 1.0.
        logger (Logger | list[Logger] | None): Logger (or iterable collection
            of loggers) for experiment tracking.
            - If True uses the default `TensorBoardLogger`.
            - If False will disable logging.
            - If multiple loggers are provided and the `save_dir` property of
              that logger is not set, local files (checkpoints, profiler traces,
              etc.) are saved in `default_root_dir` rather than in the `log_dir`
              of the individual loggers.
            Defaults to True.
        log_every_n_steps (int): How often to log within steps. Defaults to 50.
        max_epochs (int | None): Stop training once this number of epochs is
            reached. Disabled by default (None).
            - If both `max_epochs` and `max_steps` are not specified, defaults
              to `max_epochs=1000`.
            - To enable infinite training, set `max_epochs=-1`.
        min_epochs (int | None): Force training for at least these many epochs.
            Disabled by default (None).
        max_steps (int): Stop training after this number of steps. Disabled by
            default (-1).
            - If `max_steps= 1` and `max_epochs=None`, will default  to
              `max_epochs = 1000`.
            - To enable infinite training, set `max_epochs=-1`.
        min_steps (int | None): Force training for at least these number of
            steps. Disabled by default (None).
        max_time (str | timedelta | dict[str, int] | None): Stop training after
            this amount of time has passed. Disabled by default (None).
            The time duration can be specified in the format DD:HH:MM:SS
            (days, hours, minutes seconds), as a :class:`datetime.timedelta`,
            or a dictionary with keys that will be passed to
            :class:`datetime.timedelta`.
        move_metrics_to_cpu (bool): Whether to force internal logged metrics to
            be moved to cpu. This can save some gpu memory, but can make
            training slower. Use with attention. Defaults to False.
        multiple_trainloader_mode (str): How to loop over the datasets when
            there are multiple train loaders.
            - In `max_size_cycle` mode, the trainer ends one epoch when the
              largest dataset is traversed, and smaller datasets reload when
              running out of their data.
            - In `min_size` mode, all the datasets reload when reaching the
              minimum length of datasets.
            Defaults to "max_size_cycle".
        num_nodes (int): Number of GPU nodes for distributed training.
            Defaults to 1.
        num_sanity_val_steps (int): Sanity check runs n validation batches
            before starting the training routine. Set it to -1 to run all
            batches in all validation dataloaders. Defaults to 2.
        overfit_batches (int | float): Over-fit a fraction of
        training/validation
            data (float) or a set number of batches (int). Defaults to 0.0.
        plugins: Plugins allow modification of core behavior like ddp and amp,
            and enable custom lightning plugins. Defaults to None.
        precision (int | str): Double precision (64), full precision (32),
            half precision (16) or bfloat16 precision (bf16). Can be used on
            CPU, GPU, TPUs, HPUs or IPUs. Defaults to 32.
        profiler (Profiler | str | None): To profile individual steps during
            training and assist in identifying bottlenecks. Defaults to None.
        reload_dataloaders_every_n_epochs (int): Set to a non-negative integer
            to reload dataloaders every n epochs. Defaults to 0.
        replace_sampler_ddp (bool): Explicitly enables or disables sampler
            replacement. If not specified this will toggle automatically when
            DDP is used. By default, it will add `shuffle=True` for train
            sampler and `shuffle=False` for val/test sampler. If you want to
            customize it, you can set `replace_sampler_ddp=False` and add your
            own distributed sampler.
        strategy (str | Strategy | None): Supports different training
            strategies with aliases as well custom strategies. Defaults to None.
        sync_batchnorm (bool): Synchronize batch norm layers between process
            groups/whole world. Defaults to False.
        track_grad_norm (int | float | str):
            - -1 no tracking. Otherwise tracks that p-norm.
            - May be set to 'inf' infinity-norm.
            - If using Automatic Mixed Precision (AMP), the gradients will be
              unscaled before logging them.
            Defaults to -1.
        val_check_interval (int | float | None): How often to check the
            validation set.
            - Pass a `float` in the range [0.0, 1.0] to check after a fraction
              of the training epoch.
            - Pass an `int` to check after a fixed number of training batches.
              An `int` value can only be higher than the number of training
              batches when `check_val_every_n_epoch=None`, which validates
              after every `N` training batches across epochs or during
              iteration-based training.
            Defaults to 1.0.
    """
    
    @pl.Trainer.current_epoch.setter
    def current_epoch(self, current_epoch: int):
        self.fit_loop.current_epoch = current_epoch
    
    @pl.Trainer.global_step.setter
    def global_step(self, global_step: int):
        self.fit_loop.global_step = global_step
    
    def _log_device_info(self):
        if CUDAAccelerator.is_available():
            gpu_available = True
            gpu_type = " (cuda)"
        elif MPSAccelerator.is_available():
            gpu_available = True
            gpu_type = " (mps)"
        else:
            gpu_available = False
            gpu_type = ""
        
        gpu_used = isinstance(
            self.accelerator, (CUDAAccelerator, MPSAccelerator)
            )
        console.log(
            f"GPU available: {gpu_available}{gpu_type}, used: {gpu_used}"
            )
        
        num_ipus = self.num_devices if isinstance(
            self.accelerator, IPUAccelerator
            ) else 0
        console.log(f"IPU available: {_IPU_AVAILABLE}, using: {num_ipus} IPUs")
        
        num_hpus = self.num_devices if isinstance(
            self.accelerator, HPUAccelerator
            ) else 0
        console.log(f"HPU available: {_HPU_AVAILABLE}, using: {num_hpus} HPUs")
        
        # Integrate MPS Accelerator here, once gpu maps to both
        if CUDAAccelerator.is_available() and not isinstance(
            self.accelerator, CUDAAccelerator
            ):
            console.log(
                "GPU available but not used. Set `accelerator` and `devices` "
                "using"
                f" `Trainer(accelerator='gpu', devices="
                f"{CUDAAccelerator.auto_device_count()})`.",
            )
        
        if _IPU_AVAILABLE and not isinstance(self.accelerator, IPUAccelerator):
            console.log(
                "IPU available but not used. Set `accelerator` and `devices` "
                "using"
                f" `Trainer(accelerator='ipu', devices="
                f"{IPUAccelerator.auto_device_count()})`."
            )
        
        if _HPU_AVAILABLE and not isinstance(self.accelerator, HPUAccelerator):
            console.log(
                "HPU available but not used. Set `accelerator` and `devices` "
                "using"
                f" `Trainer(accelerator='hpu', devices="
                f"{HPUAccelerator.auto_device_count()})`."
            )
        
        if MPSAccelerator.is_available() and not isinstance(
            self.accelerator, MPSAccelerator
            ):
            console.log(
                "MPS available but not used. Set `accelerator` and `devices` "
                "using"
                f" `Trainer(accelerator='mps', devices="
                f"{MPSAccelerator.auto_device_count()})`."
            )
