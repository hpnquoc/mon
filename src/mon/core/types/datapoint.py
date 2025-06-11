#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements a base data-point class for handling both dataset annotations/labels
and inference/prediction results.
"""

__all__ = [
    "BaseTensor",
    "DataLoaderMixin",
    "DatasetMixin",
]

import abc
from typing import Any

import numpy as np
import torch


# ----- Base -----
class BaseTensor(abc.ABC):
    """Base tensor class with additional methods for easy manipulation and
    device handling.

    This class provides a foundation for tensor-like objects with device
    management capabilities, supporting both PyTorch tensors and NumPy arrays.
    It includes methods for moving data between devices and converting between
    tensor types.

    Args:
        data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``.
            Can be ``None`` if not loaded yet (lazy loading).
        orig_shape: Original shape of the image in [H, W] format.
            Can be ``None`` if not known (lazy loading).

    Examples:
        >>> import torch
        >>> data        = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> orig_shape  = (720, 1280)
        >>> base_tensor = BaseTensor(data, orig_shape)
        >>> cpu_tensor  = base_tensor.cpu()
        >>> numpy_array = base_tensor.numpy()
        >>> gpu_tensor  = base_tensor.cuda()
    """

    def __init__(self, data: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]):
        # if not isinstance(data, torch.Tensor | np.ndarray):
        #     raise TypeError(f"[data] must be a torch.Tensor or numpy.ndarray, got {type(data)}.")

        self._data       = data
        self._orig_shape = orig_shape

    def __len__(self) -> int:
        """Return the length of the underlying data tensor.

        Returns:
            The number of elements in the first dimension of the data tensor.

        Examples:
            >>> data        = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> len(base_tensor)
            2
        """
        return len(self.data)

    def __getitem__(self, idx: int | list[int] | torch.Tensor):
        """Return a new ``BaseTensor`` instance containing the specified indexed
        elements of the data tensor.

        Args:
            idx: Index or indices to select from the data tensor.

        Returns:
            A new ``BaseTensor`` instance containing the indexed data.

        Examples:
            >>> data        = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> result      = base_tensor[0]  # Select the first row
            >>> print(result.data)
            tensor([1, 2, 3])
        """
        return self.__class__(self.data[idx], self.orig_shape)

    @property
    def data(self):
        """Returns the data if already loaded, otherwise calls the ``load()``
        method.

        This property allows for lazy loading of the data to avoid unnecessary
        memory usage (e.g., when the data is large and not immediately needed).
        """
        return self._data if self._data is not None else self.load()

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying data tensor.

        Returns:
            The shape of the data tensor.

        Examples:
            >>> data = torch.rand(100, 4)
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> print(base_tensor.shape)
            (100, 4)
        """
        return self.data.shape

    @property
    def orig_shape(self) -> tuple[int, ...]:
        """Return the original shape of the data tensor.

        Returns:
            The original shape of the data tensor as a tuple.

        Examples:
            >>> data        = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape  = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
            >>> print(base_tensor.orig_shape)
            (720, 1280)
        """
        return self._orig_shape

    def cpu(self):
        """Return a copy of the tensor stored in CPU memory.

        Returns:
            A new ``BaseTensor`` object with the data tensor moved to CPU memory.

        Examples:
            >>> data        = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> cpu_tensor  = base_tensor.cpu()
            >>> isinstance(cpu_tensor, BaseTensor)
            True
            >>> cpu_tensor.data.device
            device(type='cpu')
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def cuda(self):
        """Move the tensor to GPU memory.

        Returns:
            A new ``BaseTensor`` instance with the data moved to GPU memory if
            it's not already an ``np.ndarray`` array, otherwise returns self.

        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import BaseTensor
            >>> data        = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> gpu_tensor  = base_tensor.cuda()
            >>> print(gpu_tensor.data.device)
            cuda:0
        """
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def numpy(self):
        """Return a copy of the tensor as a numpy array.

        Returns:
            An ``np.ndarray`` array containing the same data as the original tensor.

        Examples:
            >>> data        = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape  = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
            >>> numpy_array = base_tensor.numpy()
            >>> print(type(numpy_array))
            <class 'numpy.ndarray'>
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def to(self, *args, **kwargs):
        """Return a copy of the tensor with the specified device and dtype.

        Args:
            *args: Variable length argument list to be passed to ``torch.Tensor.to()``.
            **kwargs: Arbitrary keyword arguments to be passed to ``torch.Tensor.to()``.

        Returns:
            A new ``BaseTensor`` instance with the data moved to the specified
            device and/or dtype.

        Examples:
            >>> base_tensor    = BaseTensor(torch.randn(3, 4), orig_shape=(480, 640))
            >>> cuda_tensor    = base_tensor.to("cuda")
            >>> float16_tensor = base_tensor.to(dtype=torch.float16)
        """
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def load(self):
        """Load the data into memory.

        This method should be overridden in subclasses to implement specific
        loading logic.

        Returns:
            The loaded data as a ``torch.Tensor`` or ``numpy.ndarray``.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError("Subclasses must implement the load method.")


class DatasetMixin(abc.ABC):
    """Mixin class for ``torch.utils.data.Dataset``.

    This class provides an interface for transforming the underlying data to
    tensors for use with PyTorch's Dataset.

    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``None``.
    """

    albumentation_target_type: str = None

    @staticmethod
    @abc.abstractmethod
    def to_tensor(
        data      : torch.Tensor | np.ndarray,
        orig_shape: tuple[int, int] = None,
        normalize : bool            = True,
        *args, **kwargs
    ) -> torch.Tensor:
        """Transforms the underlying data to a tensor.

        Args:
            data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``.
            orig_shape: Original shape of the image in [H, W] format. Default is ``None``.
            normalize: Normalize the underlying data. Default is ``True``.

        Returns:
            ``torch.Tensor`` of converted data.
        """
        pass


class DataLoaderMixin(abc.ABC):
    """Mixin class for ``torch.utils.data.DataLoader``.

    This class provides an interface for collating batches of data for use with
    PyTorch's DataLoader.
    """

    @staticmethod
    @abc.abstractmethod
    def collate_fn(batch: list[Any]) -> Any:
        """Collates batch data for ``torch.utils.data.DataLoader``.

        Args:
            batch: List of annotation objects.

        Returns:
            Collated data in suitable format.
        """
        pass


# ----- Basic Datapoints -----
