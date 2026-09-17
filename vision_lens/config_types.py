from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from vision_lens.config_options import (
    CropMode,
    Device,
    ForegroundSide,
    GridFormat,
    HeadFusion,
    ImageFormat,
    NormalizationMode,
    OverwritePolicy,
    PadMode,
    Precision,
    PreprocessingInterpolation,
    ProjectionMode,
    RawFormat,
    ResizeMode,
    RGBFitScope,
    VisualizationInterpolation,
)

AttentionLayers = Literal["all"] | tuple[int, ...]
AnalysisMethod = Literal["attention", "rollout", "gradcam", "patch_pca"]
VisualizationOutputSize = Literal["match"] | tuple[int, int] | None

METHOD_TASKS = {
    "attention": "vit_attention",
    "rollout": "vit_rollout",
    "gradcam": "gradcam",
    "patch_pca": "patch_pca",
}


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

    @property
    def task(self) -> str:
        return METHOD_TASKS[self.analysis.method]
