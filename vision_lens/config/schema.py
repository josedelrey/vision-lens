"""Configuration types, choices, defaults, and structural metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, get_args

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

AttentionLayers = Literal["all"] | tuple[int, ...]
AnalysisMethod = Literal["attention", "rollout", "gradcam", "patch_pca"]
VisualizationOutputSize = Literal["match"] | tuple[int, int] | None


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    backend: str
    name: str
    pretrained: bool = True
    options: dict[str, Any] | None = None


@dataclass(frozen=True)
class InputConfig:
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    heatmaps: bool = True
    overlays: bool = True
    grids: bool = True
    raw_arrays: bool = False
    image_format: ImageFormat = "png"
    raw_format: RawFormat = "npy"
    overwrite: OverwritePolicy = "error"


@dataclass(frozen=True)
class AttentionAnalysisConfig:
    layers: AttentionLayers
    heads: tuple[int, ...] | None = None
    head_fusion: HeadFusion = "mean"
    method: Literal["attention"] = field(default="attention", init=False)


@dataclass(frozen=True)
class RolloutAnalysisConfig:
    layers: AttentionLayers
    heads: tuple[int, ...] | None = None
    head_fusion: HeadFusion = "mean"
    method: Literal["rollout"] = field(default="rollout", init=False)


@dataclass(frozen=True)
class GradCAMAnalysisConfig:
    target_layer: str | None = None
    target_class: int | None = None
    method: Literal["gradcam"] = field(default="gradcam", init=False)


@dataclass(frozen=True)
class PatchPCAAnalysisConfig:
    foreground_separation: bool | None = None
    foreground_threshold: float | Literal["auto"] | None = None
    foreground_side: ForegroundSide | None = None
    rgb_fit_scope: RGBFitScope | None = None
    projection: ProjectionMode = "fit"
    projection_path: Path | None = None
    save_projection: Path | None = None
    method: Literal["patch_pca"] = field(default="patch_pca", init=False)


AnalysisConfig = (
    AttentionAnalysisConfig
    | RolloutAnalysisConfig
    | GradCAMAnalysisConfig
    | PatchPCAAnalysisConfig
)


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: int = 672
    resize: ResizeMode = "stretch"
    crop: CropMode = "none"
    pad: PadMode = "none"
    interpolation: PreprocessingInterpolation | None = None
    normalize: bool = True
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    device: Device = "auto"
    batch_size: int = 8
    workers: int = 0
    precision: Precision = "float32"
    seed: int | None = None


@dataclass(frozen=True)
class ColormapSpec:
    name: str
    black_threshold: int
    black_blend_width: int
    black_transparent: bool


@dataclass(frozen=True)
class OverlayAlphaCurveSpec:
    steepness: float
    midpoint: float = 0.5


@dataclass(frozen=True)
class VisualizationConfig:
    tile_size: tuple[int, int] | None = None
    columns: int | None = None
    items_per_grid: int | None = None
    spacing: int | None = None
    padding: int | None = None
    labels: bool | None = None
    background: str | None = None
    dpi: int | None = None
    output_size: VisualizationOutputSize = None
    interpolation: VisualizationInterpolation = "bilinear"
    anyup_query_chunk_size: int | None = None
    overlay_alpha: float = 0.45
    cmap: str = "viridis"
    cmap_black: tuple[int, int, bool] | None = None
    grid_format: GridFormat = "png"
    normalization: NormalizationMode = "per_map"
    normalization_range: tuple[float, float] | None = None
    overlay_alpha_curve: OverlayAlphaCurveSpec | None = None

    @property
    def render_cmap(self) -> str | ColormapSpec:
        if self.cmap_black is None:
            return self.cmap
        threshold, blend_width, transparent = self.cmap_black
        return ColormapSpec(self.cmap, threshold, blend_width, transparent)

    @property
    def overlay_alpha_curve_steepness(self) -> float | None:
        if self.overlay_alpha_curve is None:
            return None
        return self.overlay_alpha_curve.steepness

    @property
    def overlay_alpha_curve_midpoint(self) -> float:
        if self.overlay_alpha_curve is None:
            return 0.5
        return self.overlay_alpha_curve.midpoint


@dataclass(frozen=True)
class VideoConfig:
    start_time: float = 0.0
    end_time: float | None = None
    sampling_rate: float | Literal["auto"] = 5.0
    frame_limit: int | None = None
    pca_fit_frames: int = 32
    temporal_smoothing: float = 0.0
    codec: str = "libx264"


@dataclass(frozen=True)
class VisionLensConfig:
    input: InputConfig
    model: ModelConfig
    preprocessing: PreprocessingConfig
    analysis: AnalysisConfig
    runtime: RuntimeConfig
    visualization: VisualizationConfig
    output: OutputConfig
    video: VideoConfig | None = None


DEFAULT_INPUT_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.webp")
DEFAULT_ANALYSIS_METHOD = "attention"
PATCH_PCA_IMAGE_FIT_DEFAULTS = {
    "foreground_separation": True,
    "foreground_threshold": 0.5,
    "foreground_side": "high",
    "rgb_fit_scope": "foreground",
}

KNOWN_FIXED_IMAGE_SIZES = {
    ("timm", "vit_small_patch8_224.dino"): 224,
}
KNOWN_VIT_DEPTHS = {
    "vit_small_patch8_224.dino": 12,
    "hf_hub:timm/vit_small_patch14_reg4_dinov2.lvd142m": 12,
    "hf_hub:timm/vit_base_patch14_dinov2.lvd142m": 12,
}
KNOWN_VIT_HEADS = {
    "vit_small_patch8_224.dino": 6,
    "hf_hub:timm/vit_small_patch14_reg4_dinov2.lvd142m": 6,
    "hf_hub:timm/vit_base_patch14_dinov2.lvd142m": 12,
}

SECTION_CONFIG_TYPES = {
    "model": ModelConfig,
    "preprocessing": PreprocessingConfig,
    "runtime": RuntimeConfig,
    "visualization": VisualizationConfig,
    "output": OutputConfig,
    "video": VideoConfig,
}
SECTION_KEYS = {
    "input": {"files", "folders", "patterns", "recursive", "limit"},
    **{
        section: {field.name for field in fields(config_type)}
        for section, config_type in SECTION_CONFIG_TYPES.items()
    },
}
TOP_LEVEL_KEYS = set(SECTION_KEYS) | {"analysis"}
VIDEO_OUTPUT_KEYS = SECTION_KEYS["output"] - {"grids", "image_format"}
GRID_VISUALIZATION_KEYS = {
    "tile_size",
    "columns",
    "items_per_grid",
    "spacing",
    "padding",
    "labels",
    "background",
    "dpi",
    "grid_format",
}
ANALYSIS_CONFIG_TYPES = {
    "attention": AttentionAnalysisConfig,
    "rollout": RolloutAnalysisConfig,
    "gradcam": GradCAMAnalysisConfig,
    "patch_pca": PatchPCAAnalysisConfig,
}
ANALYSIS_KEYS = {
    method: {field.name for field in fields(config_type)}
    for method, config_type in ANALYSIS_CONFIG_TYPES.items()
}


def _dataclass_defaults(config_type: type[Any]) -> dict[str, Any]:
    defaults = {}
    for dataclass_field in fields(config_type):
        if dataclass_field.default is not MISSING:
            defaults[dataclass_field.name] = dataclass_field.default
        elif dataclass_field.default_factory is not MISSING:
            defaults[dataclass_field.name] = dataclass_field.default_factory()
    return defaults


SECTION_DEFAULTS = {
    section: _dataclass_defaults(config_type)
    for section, config_type in SECTION_CONFIG_TYPES.items()
}
ANALYSIS_DEFAULTS = {
    method: _dataclass_defaults(config_type)
    for method, config_type in ANALYSIS_CONFIG_TYPES.items()
}


def config_section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return dict(section)
