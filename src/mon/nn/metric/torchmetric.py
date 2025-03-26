#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""TorchMetric.

This module provides an interface to :mod:`torchmetrics` library.
"""

from __future__ import annotations

__all__ = [
	"SensitivityAtSpecificity",
    "AUROC",
    "Accuracy",
    "AveragePrecision",
    "CalibrationError",
    "CohenKappa",
    "ConcordanceCorrCoef",
    "ConfusionMatrix",
    "CosineSimilarity",
    "CramersV",
    "CriticalSuccessIndex",
    "Dice",
    "ErrorRelativeGlobalDimensionlessSynthesis",
    "ExactMatch",
    "ExplainedVariance",
    "F1Score",
    "FBetaScore",
    "FleissKappa",
    "HammingDistance",
    "HingeLoss",
    "InceptionScore",
    "JaccardIndex",
    "KLDivergence",
    "KendallRankCorrCoef",
    "KernelInceptionDistance",
    "LearnedPerceptualImagePatchSimilarity",
    "LogCoshError",
    "MatthewsCorrCoef",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanSquaredError",
    "MeanSquaredLogError",
    "MemorizationInformedFrechetInceptionDistance",
    "MinkowskiDistance",
    "MultiScaleStructuralSimilarityIndexMeasure",
    "PeakSignalNoiseRatio",
    "PeakSignalNoiseRatioWithBlockedEffect",
    "PearsonCorrCoef",
    "PearsonsContingencyCoefficient",
    "PerceptualPathLength",
    "Precision",
    "PrecisionAtFixedRecall",
    "PrecisionRecallCurve",
    "QualityWithNoReference",
    "R2Score",
    "ROC",
    "Recall",
    "RecallAtFixedPrecision",
    "RelativeAverageSpectralError",
    "RelativeSquaredError",
    "RootMeanSquaredErrorUsingSlidingWindow",
    "SpatialCorrelationCoefficient",
    "SpatialDistortionIndex",
    "SpearmanCorrCoef",
    "Specificity",
    "SpecificityAtSensitivity",
    "SpectralAngleMapper",
    "SpectralDistortionIndex",
    "StatScores",
    "StructuralSimilarityIndexMeasure",
    "SymmetricMeanAbsolutePercentageError",
    "TheilsU",
    "TotalVariation",
    "TschuprowsT",
    "TweedieDevianceScore",
    "UniversalImageQualityIndex",
    "VisualInformationFidelity",
    "WeightedMeanAbsolutePercentageError",
]

import torchmetrics

from mon.globals import METRICS


# region Classification

AUROC                    = torchmetrics.classification.AUROC
Accuracy                 = torchmetrics.classification.Accuracy
AveragePrecision         = torchmetrics.classification.AveragePrecision
CalibrationError         = torchmetrics.classification.CalibrationError
CohenKappa               = torchmetrics.classification.CohenKappa
ConfusionMatrix          = torchmetrics.classification.ConfusionMatrix
Dice                     = torchmetrics.classification.Dice
ExactMatch               = torchmetrics.classification.ExactMatch
F1Score                  = torchmetrics.classification.F1Score
FBetaScore               = torchmetrics.classification.FBetaScore
HammingDistance          = torchmetrics.classification.HammingDistance
HingeLoss                = torchmetrics.classification.HingeLoss
JaccardIndex             = torchmetrics.classification.JaccardIndex
MatthewsCorrCoef         = torchmetrics.classification.MatthewsCorrCoef
Precision                = torchmetrics.classification.Precision
PrecisionAtFixedRecall   = torchmetrics.classification.PrecisionAtFixedRecall
PrecisionRecallCurve     = torchmetrics.classification.PrecisionRecallCurve
ROC                      = torchmetrics.classification.ROC
Recall                   = torchmetrics.classification.Recall
RecallAtFixedPrecision   = torchmetrics.classification.RecallAtFixedPrecision
SensitivityAtSpecificity = torchmetrics.classification.SensitivityAtSpecificity
Specificity              = torchmetrics.classification.Specificity
SpecificityAtSensitivity = torchmetrics.classification.SpecificityAtSensitivity
StatScores               = torchmetrics.classification.StatScores

METRICS.register(name="auroc",                      module=AUROC)
METRICS.register(name="accuracy",                   module=Accuracy)
METRICS.register(name="average_precision",          module=AveragePrecision)
METRICS.register(name="calibration_error",          module=CalibrationError)
METRICS.register(name="cohen_kappa",                module=CohenKappa)
METRICS.register(name="confusion_matrix",           module=ConfusionMatrix)
METRICS.register(name="dice",                       module=Dice)
METRICS.register(name="exact_match",                module=ExactMatch)
METRICS.register(name="f1_score ",                  module=F1Score)
METRICS.register(name="f_beta_score",               module=FBetaScore)
METRICS.register(name="hamming_distance",           module=HammingDistance)
METRICS.register(name="hinge_loss",                 module=HingeLoss)
METRICS.register(name="jaccard_index",              module=JaccardIndex)
METRICS.register(name="matthews_corr_coef",         module=MatthewsCorrCoef)
METRICS.register(name="precision",                  module=Precision)
METRICS.register(name="precision_at_fixed_recall",  module=PrecisionAtFixedRecall)
METRICS.register(name="precision_recall_curve",     module=PrecisionRecallCurve)
METRICS.register(name="roc",                        module=ROC)
METRICS.register(name="recall",                     module=Recall)
METRICS.register(name="recall_at_fixed_precision",  module=RecallAtFixedPrecision)
METRICS.register(name="sensitivity_at_specificity", module=SensitivityAtSpecificity)
METRICS.register(name="specificity",                module=Specificity)
METRICS.register(name="specificity_at_sensitivity", module=SpecificityAtSensitivity)
METRICS.register(name="stat_scores",                module=StatScores)

# endregion


# region Image

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


# region Nominal

CramersV                       = torchmetrics.nominal.CramersV
FleissKappa                    = torchmetrics.nominal.FleissKappa
PearsonsContingencyCoefficient = torchmetrics.nominal.PearsonsContingencyCoefficient
TheilsU                        = torchmetrics.nominal.TheilsU
TschuprowsT                    = torchmetrics.nominal.TschuprowsT

METRICS.register(name="cramers_v",                        module=CramersV)
METRICS.register(name="fleiss_kappa",                     module=FleissKappa)
METRICS.register(name="pearsons_contingency_coefficient", module=PearsonsContingencyCoefficient)
METRICS.register(name="theils_u",                         module=TheilsU)
METRICS.register(name="tschuprows_t",                     module=TschuprowsT)

# endregion


# region Regression

ConcordanceCorrCoef                  = torchmetrics.regression.ConcordanceCorrCoef
CosineSimilarity                     = torchmetrics.regression.CosineSimilarity
CriticalSuccessIndex                 = torchmetrics.regression.CriticalSuccessIndex
ExplainedVariance                    = torchmetrics.regression.ExplainedVariance
KLDivergence                         = torchmetrics.regression.KLDivergence
KendallRankCorrCoef                  = torchmetrics.regression.KendallRankCorrCoef
LogCoshError                         = torchmetrics.regression.LogCoshError
MeanAbsoluteError                    = torchmetrics.regression.MeanAbsoluteError
MeanAbsolutePercentageError          = torchmetrics.regression.MeanAbsolutePercentageError
MeanSquaredError                     = torchmetrics.regression.MeanSquaredError
MeanSquaredLogError                  = torchmetrics.regression.MeanSquaredLogError
MinkowskiDistance                    = torchmetrics.regression.MinkowskiDistance
PearsonCorrCoef                      = torchmetrics.regression.PearsonCorrCoef
R2Score                              = torchmetrics.regression.R2Score
RelativeSquaredError                 = torchmetrics.regression.RelativeSquaredError
SpearmanCorrCoef                     = torchmetrics.regression.SpearmanCorrCoef
SymmetricMeanAbsolutePercentageError = torchmetrics.regression.SymmetricMeanAbsolutePercentageError
TweedieDevianceScore                 = torchmetrics.regression.TweedieDevianceScore
WeightedMeanAbsolutePercentageError  = torchmetrics.regression.WeightedMeanAbsolutePercentageError

METRICS.register(name="concordance_corr_coef",                    module=ConcordanceCorrCoef)
METRICS.register(name="cosine_similarity",                        module=CosineSimilarity)
METRICS.register(name="critical_success_index",                   module=CriticalSuccessIndex)
METRICS.register(name="explained_variance",                       module=ExplainedVariance)
METRICS.register(name="kendall_rank_corr_coef",                   module=KendallRankCorrCoef)
METRICS.register(name="kl_divergence",                            module=KLDivergence)
METRICS.register(name="log_cosh_error",                           module=LogCoshError)
METRICS.register(name="mae",                                      module=MeanAbsoluteError)
METRICS.register(name="mean_absolute_error",                      module=MeanAbsoluteError)
METRICS.register(name="mean_absolute_percentage_error",           module=MeanAbsolutePercentageError)
METRICS.register(name="mean_squared_error",                       module=MeanSquaredError)
METRICS.register(name="mean_squared_log_error",                   module=MeanSquaredLogError)
METRICS.register(name="minkowski_distance",                       module=MinkowskiDistance)
METRICS.register(name="mse",                                      module=MeanSquaredError)
METRICS.register(name="pearson_corr_coef",                        module=PearsonCorrCoef)
METRICS.register(name="r2_score",                                 module=R2Score)
METRICS.register(name="relative_squared_error",                   module=RelativeSquaredError)
METRICS.register(name="spearman_corr_coef",                       module=SpearmanCorrCoef)
METRICS.register(name="symmetric_mean_absolute_percentage_error", module=SymmetricMeanAbsolutePercentageError)
METRICS.register(name="tweedie_deviance_score",                   module=TweedieDevianceScore)
METRICS.register(name="weighted_mean_absolute_percentage_error",  module=WeightedMeanAbsolutePercentageError)

# endregion
