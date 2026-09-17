"""Typed configuration choices shared by parsing and validation."""

from typing import Literal, get_args

Device = Literal["auto", "cpu", "cuda", "mps"]
Precision = Literal["float32", "float16", "bfloat16"]
ResizeMode = Literal["stretch", "shortest", "longest", "none"]
CropMode = Literal["none", "center"]
PadMode = Literal["none", "center"]
PreprocessingInterpolation = Literal["nearest", "bilinear", "bicubic", "lanczos"]
HeadFusion = Literal["mean", "max", "none"]
ForegroundSide = Literal["high", "low"]
RGBFitScope = Literal["foreground", "all"]
ProjectionMode = Literal["fit", "load"]
VisualizationInterpolation = Literal[
    "nearest",
    "bilinear",
    "bilinear_mask",
    "anyup",
    "anyup_mask",
    "anyup_soft",
    "anyup_soft_mask",
]
GridFormat = Literal["pdf", "png", "svg"]
NormalizationMode = Literal["per_map", "shared", "fixed"]
ImageFormat = Literal["jpeg", "png", "tiff", "webp"]
RawFormat = Literal["npy", "npz"]
OverwritePolicy = Literal["replace", "error", "skip"]

DEVICE_CHOICES = frozenset(get_args(Device))
PRECISION_CHOICES = frozenset(get_args(Precision))
RESIZE_CHOICES = frozenset(get_args(ResizeMode))
CROP_CHOICES = frozenset(get_args(CropMode))
PAD_CHOICES = frozenset(get_args(PadMode))
PREPROCESSING_INTERPOLATION_CHOICES = frozenset(get_args(PreprocessingInterpolation))
HEAD_FUSION_CHOICES = frozenset(get_args(HeadFusion))
FOREGROUND_SIDE_CHOICES = frozenset(get_args(ForegroundSide))
RGB_FIT_SCOPE_CHOICES = frozenset(get_args(RGBFitScope))
PROJECTION_CHOICES = frozenset(get_args(ProjectionMode))
VISUALIZATION_INTERPOLATION_CHOICES = frozenset(get_args(VisualizationInterpolation))
GRID_FORMAT_CHOICES = frozenset(get_args(GridFormat))
NORMALIZATION_CHOICES = frozenset(get_args(NormalizationMode))
IMAGE_FORMAT_CHOICES = frozenset(get_args(ImageFormat))
RAW_FORMAT_CHOICES = frozenset(get_args(RawFormat))
OVERWRITE_CHOICES = frozenset(get_args(OverwritePolicy))
ANYUP_INTERPOLATIONS = frozenset(
    {"anyup", "anyup_mask", "anyup_soft", "anyup_soft_mask"}
)
SOFT_ANYUP_INTERPOLATIONS = frozenset({"anyup_soft", "anyup_soft_mask"})
