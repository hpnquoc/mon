#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Metric Module.

This module implements image metrics.
"""

from __future__ import annotations

__all__ = [
    "CustomMSSSIM",
    "CustomSSIM",
    "ErrorRelativeGlobalDimensionlessSynthesis",
    "GoodLookingImageMetric",
    "InceptionScore",
    "KernelInceptionDistance",
    "LearnedPerceptualImagePatchSimilarity",
    "MemorizationInformedFrechetInceptionDistance",
    "MultiScaleStructuralSimilarityIndexMeasure",
    "PeakSignalNoiseRatio",
    "PeakSignalNoiseRatioWithBlockedEffect",
    "PerceptualPathLength",
    "QualityWithNoReference",
    "RelativeAverageSpectralError",
    "RootMeanSquaredErrorUsingSlidingWindow",
    "SpatialCorrelationCoefficient",
    "SpatialDistortionIndex",
    "SpectralAngleMapper",
    "SpectralDistortionIndex",
    "StructuralSimilarityIndexMeasure",
    "TotalVariation",
    "UniversalImageQualityIndex",
    "VisualInformationFidelity",
    "custom_ms_ssim",
    "custom_ssim",
]

import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from mon.globals import METRICS


# region Image Metric

ErrorRelativeGlobalDimensionlessSynthesis    = torchmetrics.image.ErrorRelativeGlobalDimensionlessSynthesis
InceptionScore                               = torchmetrics.image.InceptionScore
KernelInceptionDistance                      = torchmetrics.image.KernelInceptionDistance
LearnedPerceptualImagePatchSimilarity        = torchmetrics.image.LearnedPerceptualImagePatchSimilarity
MemorizationInformedFrechetInceptionDistance = torchmetrics.image.MemorizationInformedFrechetInceptionDistance
MultiScaleStructuralSimilarityIndexMeasure   = torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure
PeakSignalNoiseRatio                         = torchmetrics.image.PeakSignalNoiseRatio
PeakSignalNoiseRatioWithBlockedEffect        = torchmetrics.image.PeakSignalNoiseRatioWithBlockedEffect
PerceptualPathLength                         = torchmetrics.image.PerceptualPathLength
QualityWithNoReference                       = torchmetrics.image.QualityWithNoReference
RelativeAverageSpectralError                 = torchmetrics.image.RelativeAverageSpectralError
RootMeanSquaredErrorUsingSlidingWindow       = torchmetrics.image.RootMeanSquaredErrorUsingSlidingWindow
SpatialCorrelationCoefficient                = torchmetrics.image.SpatialCorrelationCoefficient
SpatialDistortionIndex                       = torchmetrics.image.SpatialDistortionIndex
SpectralAngleMapper                          = torchmetrics.image.SpectralAngleMapper
SpectralDistortionIndex                      = torchmetrics.image.SpectralDistortionIndex
StructuralSimilarityIndexMeasure             = torchmetrics.image.StructuralSimilarityIndexMeasure
TotalVariation                               = torchmetrics.image.TotalVariation
UniversalImageQualityIndex                   = torchmetrics.image.UniversalImageQualityIndex
VisualInformationFidelity                    = torchmetrics.image.VisualInformationFidelity

METRICS.register(name="error_relative_global_dimensionless_synthesis",    module=ErrorRelativeGlobalDimensionlessSynthesis)
METRICS.register(name="inception_score",                                  module=InceptionScore)
METRICS.register(name="kernel_inception_distance",                        module=KernelInceptionDistance)
METRICS.register(name="learned_perceptual_image_patch_similarity",        module=LearnedPerceptualImagePatchSimilarity)
METRICS.register(name="lpips",                                            module=LearnedPerceptualImagePatchSimilarity)
METRICS.register(name="memorization_informed_frechet_inception_distance", module=MemorizationInformedFrechetInceptionDistance)
METRICS.register(name="multiscale_ssim",                                  module=MultiScaleStructuralSimilarityIndexMeasure)
METRICS.register(name="multiscale_structural_similarity_index_measure",   module=MultiScaleStructuralSimilarityIndexMeasure)
METRICS.register(name="peak_signal_noise_ratio",                          module=PeakSignalNoiseRatio)
METRICS.register(name="peak_signal_noise_ratio_with_blocked_effect",      module=PeakSignalNoiseRatioWithBlockedEffect)
METRICS.register(name="perceptual_path_length",                           module=PerceptualPathLength)
METRICS.register(name="psnr",                                             module=PeakSignalNoiseRatio)
METRICS.register(name="quality_with_no_reference",                        module=QualityWithNoReference)
METRICS.register(name="relative_average_spectral_error",                  module=RelativeAverageSpectralError)
METRICS.register(name="root_mean_squared_error_using_sliding_window",     module=RootMeanSquaredErrorUsingSlidingWindow)
METRICS.register(name="spatial_correlation_coefficient",                  module=SpatialCorrelationCoefficient)
METRICS.register(name="spatial_distortion_index",                         module=SpatialDistortionIndex)
METRICS.register(name="spectral_angle_mapper",                            module=SpectralAngleMapper)
METRICS.register(name="spectral_distortion_index",                        module=SpectralDistortionIndex)
METRICS.register(name="ssim",                                             module=StructuralSimilarityIndexMeasure)
METRICS.register(name="structural_similarity_index_measure",              module=StructuralSimilarityIndexMeasure)
METRICS.register(name="total_variation",                                  module=TotalVariation)
METRICS.register(name="universal_image_quality_index",                    module=UniversalImageQualityIndex)
METRICS.register(name="visual_information_fidelity",                      module=VisualInformationFidelity)

# endregion


# region Custom SSIM/MS-SSIM

def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    """Create 1D gauss kernel.
    
    Args:
        size: The size of gauss kernel.
        sigma: Sigma of normal distribution.
    
    Returns:
       1D kernel ``[1 x 1 x size]``.
    
    References:
        https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    """
    coords  = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g       = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g      /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def _gaussian_filter(input: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """ Blur input with 1D kernel.
    
    Args:
        input: A batch of tensors to be blurred.
        window: 1D Gaussian kernel.
    
    Returns:
        Blurred tensors.
    
    References:
        https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    """
    assert all([ws == 1 for ws in window.shape[1:-1]]), window.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    c   = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= window.shape[-1]:
            out = conv(out, weight=window.transpose(2 + i, -1), stride=1, padding=0, groups=c)

    return out


def _custom_ssim(
    image1      : torch.Tensor,
    image2      : torch.Tensor,
    data_range  : float,
    window      : torch.Tensor,
    size_average: bool = True,
    k           : tuple[float, float] = (0.01, 0.03)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate ssim index for attr:`img1` and attr:`img2`.
    
    Args:
        image1: Image to be compared.
        image2: Image to be compared.
        data_range: Value range of input images (usually ``1.0`` or ``255``).
        window: 1D Gaussian kernel.
        K: Kernel sizes. Default: ``(0.01, 0.03)``.
        size_average: If ``True``, ssim of all images will be averaged as a scalar.
    
    References:
        https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    """
    k1, k2 = k
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    window = window.to(image1.device, dtype=image1.dtype)

    mu1 = _gaussian_filter(image1, window)
    mu2 = _gaussian_filter(image2, window)

    mu1_sq  = mu1.pow(2)
    mu2_sq  = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (_gaussian_filter(image1 * image1, window) - mu1_sq)
    sigma2_sq = compensation * (_gaussian_filter(image2 * image2, window) - mu2_sq)
    sigma12   = compensation * (_gaussian_filter(image1 * image2, window) - mu1_mu2)

    cs_map   = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def custom_ssim(
    image1           : torch.Tensor,
    image2           : torch.Tensor,
    data_range       : float = 255,
    size_average     : bool  = True,
    window_size      : int   = 11,
    window_sigma     : float = 1.5,
    window           : torch.Tensor = None,
    k                : tuple[float, float] = (0.01, 0.03),
    non_negative_ssim: bool = False,
) -> torch.Tensor:
    """Interface of :obj:`_custom_ssim`.
    
    References:
        `<https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py>`__
    """
    if not image1.shape == image2.shape:
        raise ValueError(f"`image1` and `image2` must have the same dimensions, "
                         f"but got {image1.shape} and {image2.shape}.")

    for d in range(len(image1.shape) - 1, 1, -1):
        image1 = image1.squeeze(dim=d)
        image2 = image2.squeeze(dim=d)

    if len(image1.shape) not in (4, 5):
        raise ValueError(f"`image1` and `image2` must be 4D or 5D tensors, "
                         f"but got {image1.shape}.")

    # if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")
    
    if window is not None:  # set win_size
        window_size = window.shape[-1]

    if not (window_size % 2 == 1):
        raise ValueError("`window_size` must be odd.")

    if window is None:
        window = _fspecial_gauss_1d(window_size, window_sigma)
        window = window.repeat([image1.shape[1]] + [1] * (len(image1.shape) - 1))

    ssim_per_channel, cs = _custom_ssim(image1, image2, data_range=data_range, window=window, size_average=False, k=k)
    if non_negative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def custom_ms_ssim(
    image1      : torch.Tensor,
    image2      : torch.Tensor,
    data_range  : float        = 255,
    size_average: bool         = True,
    window_size : int          = 11,
    window_sigma: float        = 1.5,
    window      : torch.Tensor = None,
    weights     : list[float]  = None,
    k           : tuple[float , float] = (0.01, 0.03),
) -> torch.Tensor:
    """Interface of MS-SSIM.
    
    References:
        https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    """
    if not image1.shape == image2.shape:
        raise ValueError(f"`image1` and `image2` must have the same dimensions, "
                         f"but got {image1.shape} and {image2.shape}.")

    for d in range(len(image1.shape) - 1, 1, -1):
        image1 = image1.squeeze(dim=d)
        image2 = image2.squeeze(dim=d)

    # if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if len(image1.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(image1.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"`image1` and `image2` must be 4D or 5D tensors, "
                         f"but got {image1.shape}")

    if window is not None:  # set win_size
        window_size = window.shape[-1]

    if not (window_size % 2 == 1):
        raise ValueError("`window_size` should be odd.")

    smaller_side = min(image1.shape[-2:])
    assert smaller_side > (window_size - 1) * (2 ** 4), (
        "`image1` and `image2` must be larger than %d due to the 4 "
        "downsamplings in ms-ssim." % ((window_size - 1) * (2 ** 4))
    )

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights_tensor = image1.new_tensor(weights)

    if window is None:
        window = _fspecial_gauss_1d(window_size, window_sigma)
        window = window.repeat([image1.shape[1]] + [1] * (len(image1.shape) - 1))

    levels = weights_tensor.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _custom_ssim(image1, image2, window=window, data_range=data_range, size_average=False, k=k)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in image1.shape[2:]]
            image1 = avg_pool(image1, kernel_size=2, padding=padding)
            image2 = avg_pool(image2, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # type: ignore  # (batch, channel)
    mcs_and_ssim     = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val      = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class CustomSSIM(nn.Module):

    def __init__(
        self,
        data_range  : float = 255,
        size_average: bool  = True,
        window_size : int   = 11,
        window_sigma: float = 1.5,
        channel     : int   = 3,
        spatial_dims: int   = 2,
        k: tuple[float, float] = (0.01, 0.03),
        non_negative_ssim: bool = False,
    ):
        super().__init__()
        self.window_size  = window_size
        self.window       = _fspecial_gauss_1d(window_size, window_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range   = data_range
        self.k            = k
        self.non_negative_ssim = non_negative_ssim
    
    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        return custom_ssim(
            image1            = image1,
            image2            = image2,
            data_range        = self.data_range,
            size_average      = self.size_average,
            window            = self.window,
            k                 = self.k,
            non_negative_ssim = self.non_negative_ssim,
        )


class CustomMSSSIM(torch.nn.Module):
    
    def __init__(
        self,
        data_range  : float = 255,
        size_average: bool  = True,
        window_size : int   = 11,
        window_sigma: float = 1.5,
        channel     : int   = 3,
        spatial_dims: int   = 2,
        weights     : list[float] = None,
        k           : tuple[float, float] = (0.01, 0.03),
    ):
        super().__init__()
        self.window_size  = window_size
        self.window       = _fspecial_gauss_1d(window_size, window_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range   = data_range
        self.weights      = weights
        self.k            = k

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        return custom_ms_ssim(
            image1       = image1,
            image2       = image2,
            data_range   = self.data_range,
            size_average = self.size_average,
            window       = self.window,
            weights      = self.weights,
            k            = self.k,
        )
    
# endregion


# region Good-Looking Image Metric

class GoodLookingImageMetric(nn.Module):
    """A good-looking image is one with a low well-exposedness, but high
    contrast and saturation.
    
    References:
        https://github.com/VinAIResearch/PSENet-Image-Enhancement/blob/main/source/iqa.py
    """
    
    def __init__(self, exposed_level: float = 0.5, pool_size: int = 25):
        super().__init__()
        self.exposed_level = exposed_level
        self.pool_size     = pool_size
        self.mean_pool     = nn.Sequential(nn.ReflectionPad2d(self.pool_size // 2), nn.AvgPool2d(self.pool_size, stride=1))
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        eps         = 1 / 255.0
        max_rgb     = torch.max(images, dim=1, keepdim=True)[0]
        min_rgb     = torch.min(images, dim=1, keepdim=True)[0]
        saturation  = (max_rgb - min_rgb + eps) / (max_rgb + eps)
        mean_rgb    = self.mean_pool(images).mean(dim=1, keepdim=True)
        exposedness = torch.abs(mean_rgb - self.exposed_level) + eps
        contrast    = self.mean_pool(images * images).mean(dim=1, keepdim=True) - mean_rgb ** 2
        return torch.mean((saturation * contrast) / exposedness, dim=[1], keepdim=True)
    
# endregion
