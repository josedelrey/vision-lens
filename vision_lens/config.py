from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Literal
from warnings import warn

import yaml

HeadFusion = Literal["mean", "max", "none"]
Device = Literal["auto", "cpu", "cuda", "mps"]
AttentionLayers = Literal["all"] | tuple[int, ...]
AnalysisMethod = Literal["attention", "rollout", "gradcam", "patch_pca"]
Precision = Literal["float32", "float16", "bfloat16"]
NormalizationMode = Literal["per_map", "shared", "fixed"]
VisualizationOutputSize = Literal["match"] | tuple[int, int] | None
VisualizationInterpolation = Literal[
    "nearest",
    "bilinear",
    "bilinear_mask",
    "anyup",
    "anyup_mask",
    "anyup_soft",
    "anyup_soft_mask",
]

DEFAULT_INPUT_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.webp")

TOP_LEVEL_KEYS = {
    "input",
    "model",
    "preprocessing",
    "analysis",
    "runtime",
    "visualization",
    "output",
    "video",
}
SECTION_KEYS = {
    "input": {"files", "folders", "patterns", "recursive", "limit"},
    "model": {"architecture", "backend", "name", "pretrained", "options"},
    "preprocessing": {
        "image_size",
        "resize",
        "crop",
        "pad",
        "interpolation",
        "normalize",
        "mean",
        "std",
    },
    "runtime": {"batch_size", "device", "workers", "precision", "seed"},
    "visualization": {
        "tile_size",
        "columns",
        "items_per_grid",
        "spacing",
        "padding",
        "labels",
        "background",
        "dpi",
        "output_size",
        "interpolation",
        "anyup_query_chunk_size",
        "overlay_alpha",
        "overlay_alpha_curve",
        "cmap",
        "cmap_black",
        "grid_format",
        "normalization",
        "normalization_range",
    },
    "output": {
        "directory",
        "heatmaps",
        "overlays",
        "grids",
        "raw_arrays",
        "image_format",
        "raw_format",
        "overwrite",
    },
    "video": {
        "start_time",
        "end_time",
        "sampling_rate",
        "frame_limit",
        "pca_fit_frames",
        "temporal_smoothing",
        "codec",
    },
}
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
ANALYSIS_KEYS = {
    "attention": {"method", "layers", "heads", "head_fusion"},
    "rollout": {"method", "layers", "heads", "head_fusion"},
    "gradcam": {"method", "target_layer", "target_class"},
    "patch_pca": {
        "method",
        "foreground_separation",
        "foreground_threshold",
        "foreground_side",
        "rgb_fit_scope",
        "projection",
        "projection_path",
        "save_projection",
    },
}
METHOD_TASKS = {
    "attention": "vit_attention",
    "rollout": "vit_rollout",
    "gradcam": "gradcam",
    "patch_pca": "patch_pca",
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
    image_format: str = "png"
    raw_format: str = "npy"
    overwrite: Literal["replace", "error", "skip"] = "error"


@dataclass(frozen=True)
class AnalysisConfig:
    method: AnalysisMethod
    layers: AttentionLayers | None = None
    heads: tuple[int, ...] | None = None
    head_fusion: HeadFusion | None = None
    target_layer: str | None = None
    target_class: int | None = None
    foreground_separation: bool | None = None
    foreground_threshold: float | Literal["auto"] | None = None
    foreground_side: Literal["high", "low"] | None = None
    rgb_fit_scope: Literal["foreground", "all"] | None = None
    projection: Literal["fit", "load"] | None = None
    projection_path: Path | None = None
    save_projection: Path | None = None


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: int = 672
    resize: Literal["stretch", "shortest", "longest", "none"] = "stretch"
    crop: Literal["none", "center"] = "none"
    pad: Literal["none", "center"] = "none"
    interpolation: str | None = None
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
    grid_format: str = "png"
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


DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig()
DEFAULT_RUNTIME_CONFIG = RuntimeConfig()
DEFAULT_VISUALIZATION_CONFIG = VisualizationConfig()
DEFAULT_OUTPUT_CONFIG = OutputConfig(Path("."))
DEFAULT_VIDEO_CONFIG = VideoConfig()


def load_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> VisionLensConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if raw_config is None:
        raw_config = {}
    elif not isinstance(raw_config, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")
    return parse_config(
        raw_config,
        base_dir=_project_root(config_path.parent),
        overrides=overrides,
    )


def parse_config(
    raw_config: dict[str, Any],
    base_dir: Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> VisionLensConfig:
    base = _project_root(Path.cwd()) if base_dir is None else Path(base_dir).resolve()
    resolved = _deep_merge(raw_config, overrides or {})
    _validate_keys(resolved)
    _validate_mode_overrides(raw_config, overrides or {})

    input_section = _section(resolved, "input")
    model_section = _section(resolved, "model")
    preprocessing_section = _section(resolved, "preprocessing")
    analysis_section = _section(resolved, "analysis")
    runtime_section = _section(resolved, "runtime")
    visualization_section = _section(resolved, "visualization")
    output_section = _section(resolved, "output")
    video_section = _section(resolved, "video")
    method = _analysis_method(analysis_section.get("method", "attention"))
    _reject_unknown_keys("analysis", analysis_section, ANALYSIS_KEYS[method])

    config = VisionLensConfig(
        input=_parse_input(input_section, base),
        model=ModelConfig(
            architecture=_required_str(model_section, "architecture", "model"),
            backend=_required_str(model_section, "backend", "model"),
            name=_required_str(model_section, "name", "model"),
            pretrained=_bool(
                model_section.get("pretrained", True),
                "model.pretrained",
            ),
            options=_optional_mapping(
                model_section.get("options"),
                "model.options",
            ),
        ),
        preprocessing=PreprocessingConfig(
            image_size=_positive_int(
                preprocessing_section.get(
                    "image_size", DEFAULT_PREPROCESSING_CONFIG.image_size
                ),
                "preprocessing.image_size",
            ),
            resize=_choice(
                preprocessing_section.get(
                    "resize", DEFAULT_PREPROCESSING_CONFIG.resize
                ),
                "preprocessing.resize",
                {"stretch", "shortest", "longest", "none"},
            ),
            crop=_choice(
                preprocessing_section.get("crop", DEFAULT_PREPROCESSING_CONFIG.crop),
                "preprocessing.crop",
                {"none", "center"},
            ),
            pad=_choice(
                preprocessing_section.get("pad", DEFAULT_PREPROCESSING_CONFIG.pad),
                "preprocessing.pad",
                {"none", "center"},
            ),
            interpolation=_optional_choice(
                preprocessing_section.get("interpolation"),
                "preprocessing.interpolation",
                {"nearest", "bilinear", "bicubic", "lanczos"},
            ),
            normalize=_bool(
                preprocessing_section.get(
                    "normalize", DEFAULT_PREPROCESSING_CONFIG.normalize
                ),
                "preprocessing.normalize",
            ),
            mean=_optional_triplet(
                preprocessing_section.get("mean"),
                "preprocessing.mean",
            ),
            std=_optional_triplet(
                preprocessing_section.get("std"),
                "preprocessing.std",
                positive=True,
            ),
        ),
        analysis=_parse_analysis(
            analysis_section,
            method,
            base,
            is_video="video" in resolved,
        ),
        runtime=RuntimeConfig(
            device=_device(
                runtime_section.get("device", DEFAULT_RUNTIME_CONFIG.device)
            ),
            batch_size=_positive_int(
                runtime_section.get("batch_size", DEFAULT_RUNTIME_CONFIG.batch_size),
                "runtime.batch_size",
            ),
            workers=_non_negative_int_value(
                runtime_section.get("workers", DEFAULT_RUNTIME_CONFIG.workers),
                "runtime.workers",
            ),
            precision=_choice(
                runtime_section.get("precision", DEFAULT_RUNTIME_CONFIG.precision),
                "runtime.precision",
                {"float32", "float16", "bfloat16"},
            ),
            seed=_optional_seed(
                runtime_section.get("seed"),
                "runtime.seed",
            ),
        ),
        visualization=VisualizationConfig(
            tile_size=_optional_size(
                visualization_section.get("tile_size"),
                "visualization.tile_size",
            ),
            columns=_optional_positive_int(
                visualization_section.get("columns"),
                "visualization.columns",
            ),
            items_per_grid=_optional_positive_int(
                visualization_section.get("items_per_grid"),
                "visualization.items_per_grid",
            ),
            spacing=_optional_non_negative_int(
                visualization_section.get("spacing"),
                "visualization.spacing",
            ),
            padding=_optional_non_negative_int(
                visualization_section.get("padding"),
                "visualization.padding",
            ),
            labels=_optional_bool(
                visualization_section.get("labels"),
                "visualization.labels",
            ),
            background=_optional_string(
                visualization_section.get("background"),
                "visualization.background",
            ),
            dpi=_optional_positive_int(
                visualization_section.get("dpi"),
                "visualization.dpi",
            ),
            output_size=_visualization_output_size(
                visualization_section.get("output_size"),
            ),
            interpolation=_choice(
                visualization_section.get(
                    "interpolation", DEFAULT_VISUALIZATION_CONFIG.interpolation
                ),
                "visualization.interpolation",
                {
                    "nearest",
                    "bilinear",
                    "bilinear_mask",
                    "anyup",
                    "anyup_mask",
                    "anyup_soft",
                    "anyup_soft_mask",
                },
            ),
            anyup_query_chunk_size=_optional_positive_int(
                visualization_section.get("anyup_query_chunk_size"),
                "visualization.anyup_query_chunk_size",
            ),
            overlay_alpha=_unit_interval(
                visualization_section.get(
                    "overlay_alpha", DEFAULT_VISUALIZATION_CONFIG.overlay_alpha
                ),
                "visualization.overlay_alpha",
            ),
            overlay_alpha_curve=_overlay_alpha_curve(
                visualization_section.get("overlay_alpha_curve")
            ),
            cmap=_optional_str(
                visualization_section.get("cmap", DEFAULT_VISUALIZATION_CONFIG.cmap),
                "visualization.cmap",
            ),
            cmap_black=_cmap_black(visualization_section.get("cmap_black")),
            grid_format=_grid_format(
                visualization_section.get(
                    "grid_format", DEFAULT_VISUALIZATION_CONFIG.grid_format
                )
            ),
            normalization=_choice(
                visualization_section.get(
                    "normalization", DEFAULT_VISUALIZATION_CONFIG.normalization
                ),
                "visualization.normalization",
                {"per_map", "shared", "fixed"},
            ),
            normalization_range=_optional_range(
                visualization_section.get("normalization_range"),
                "visualization.normalization_range",
            ),
        ),
        output=OutputConfig(
            directory=_resolve_path(
                _required_str(output_section, "directory", "output"),
                base,
            ),
            heatmaps=_bool(
                output_section.get("heatmaps", DEFAULT_OUTPUT_CONFIG.heatmaps),
                "output.heatmaps",
            ),
            overlays=_bool(
                output_section.get("overlays", method != "patch_pca"),
                "output.overlays",
            ),
            grids=(
                False
                if "video" in resolved
                else _bool(output_section.get("grids", True), "output.grids")
            ),
            raw_arrays=_bool(
                output_section.get("raw_arrays", DEFAULT_OUTPUT_CONFIG.raw_arrays),
                "output.raw_arrays",
            ),
            image_format=_image_format(
                output_section.get("image_format", DEFAULT_OUTPUT_CONFIG.image_format)
            ),
            raw_format=_raw_format(
                output_section.get("raw_format", DEFAULT_OUTPUT_CONFIG.raw_format)
            ),
            overwrite=_choice(
                output_section.get("overwrite", DEFAULT_OUTPUT_CONFIG.overwrite),
                "output.overwrite",
                {"replace", "error", "skip"},
            ),
        ),
        video=(
            None
            if "video" not in resolved
            else VideoConfig(
                start_time=_non_negative_number(
                    video_section.get("start_time", DEFAULT_VIDEO_CONFIG.start_time),
                    "video.start_time",
                ),
                end_time=_optional_non_negative_number(
                    video_section.get("end_time"),
                    "video.end_time",
                ),
                sampling_rate=_video_sampling_rate(
                    video_section.get(
                        "sampling_rate", DEFAULT_VIDEO_CONFIG.sampling_rate
                    ),
                ),
                frame_limit=_optional_positive_int(
                    video_section.get("frame_limit"),
                    "video.frame_limit",
                ),
                pca_fit_frames=_positive_int(
                    video_section.get(
                        "pca_fit_frames", DEFAULT_VIDEO_CONFIG.pca_fit_frames
                    ),
                    "video.pca_fit_frames",
                ),
                temporal_smoothing=_unit_interval(
                    video_section.get(
                        "temporal_smoothing", DEFAULT_VIDEO_CONFIG.temporal_smoothing
                    ),
                    "video.temporal_smoothing",
                ),
                codec=_optional_str(
                    video_section.get("codec", DEFAULT_VIDEO_CONFIG.codec),
                    "video.codec",
                ),
            )
        ),
    )
    validate_config(config)
    _validate_applicable_settings(resolved, config)
    return config


def validate_config(config: VisionLensConfig) -> None:
    method = config.analysis.method
    applicable = _applicable_setting_keys(config)
    if config.video is not None:
        if config.preprocessing.crop != "none" or config.preprocessing.pad != "none":
            raise ValueError(
                "Video aspect-ratio preprocessing requires preprocessing.crop and "
                "preprocessing.pad to be 'none'."
            )
        if (
            config.video.end_time is not None
            and config.video.end_time <= config.video.start_time
        ):
            raise ValueError("video.end_time must be greater than video.start_time.")
        if isinstance(config.visualization.output_size, tuple) and any(
            value % 2 for value in config.visualization.output_size
        ):
            raise ValueError(
                "visualization.output_size values must be even numbers for video."
            )
    actual_pair = (config.model.architecture, config.model.backend)
    expected_pair = ("cnn", "torchvision") if method == "gradcam" else ("vit", "timm")
    if actual_pair != expected_pair:
        raise ValueError(
            f"analysis.method={method!r} requires model.architecture="
            f"{expected_pair[0]!r} and model.backend={expected_pair[1]!r}; got "
            f"architecture={actual_pair[0]!r}, backend={actual_pair[1]!r}."
        )

    options = config.model.options or {}
    if config.model.backend == "timm":
        reserved_options = {"img_size", "pretrained"}
        if config.video is not None:
            reserved_options.add("dynamic_img_size")
    else:
        reserved_options = {"weights"}
    conflicts = sorted(reserved_options & options.keys())
    if conflicts:
        raise ValueError(
            "model.options cannot override managed loader argument(s): "
            f"{', '.join(conflicts)}."
        )

    fixed_size = KNOWN_FIXED_IMAGE_SIZES.get((config.model.backend, config.model.name))
    if fixed_size is not None and config.preprocessing.image_size != fixed_size:
        raise ValueError(
            f"model {config.model.name!r} requires preprocessing.image_size="
            f"{fixed_size}; got {config.preprocessing.image_size}. The full image "
            "will be resized to this model size without cropping."
        )

    if method in {"attention", "rollout"}:
        _validate_known_attention_constraints(config)

    if config.preprocessing.crop != "none" and config.preprocessing.pad != "none":
        raise ValueError(
            "preprocessing.crop and preprocessing.pad cannot both be enabled."
        )
    if config.preprocessing.resize == "stretch" and (
        config.preprocessing.crop != "none" or config.preprocessing.pad != "none"
    ):
        raise ValueError(
            "preprocessing.resize='stretch' already produces the exact model size; "
            "crop and pad must be 'none'."
        )
    if (
        "interpolation" in applicable["visualization"]
        and config.visualization.anyup_query_chunk_size is not None
        and config.visualization.interpolation
        not in {"anyup", "anyup_mask", "anyup_soft", "anyup_soft_mask"}
    ):
        raise ValueError(
            "visualization.anyup_query_chunk_size requires "
            "visualization.interpolation to be 'anyup', 'anyup_mask', or "
            "'anyup_soft', or 'anyup_soft_mask'."
        )
    if (
        "interpolation" in applicable["visualization"]
        and config.visualization.interpolation
        in {"anyup_soft", "anyup_soft_mask"}
        and config.visualization.anyup_query_chunk_size is None
    ):
        raise ValueError(
            "soft AnyUp interpolation requires visualization.anyup_query_chunk_size."
        )
    if (
        "normalization" in applicable["visualization"]
        and config.visualization.normalization == "fixed"
        and config.visualization.normalization_range is None
    ):
        raise ValueError(
            "visualization.normalization_range is required when normalization='fixed'."
        )
    if (
        "normalization" in applicable["visualization"]
        and config.visualization.normalization != "fixed"
        and config.visualization.normalization_range is not None
    ):
        raise ValueError(
            "visualization.normalization_range is only valid when "
            "normalization='fixed'."
        )
    if not any(
        (
            config.output.heatmaps,
            config.output.overlays,
            config.output.grids,
            config.output.raw_arrays,
            config.analysis.save_projection is not None,
        )
    ):
        raise ValueError("At least one output type must be enabled.")
    if method == "patch_pca":
        if config.output.overlays:
            raise ValueError("output.overlays is not supported for patch PCA.")
        if (
            config.analysis.projection == "load"
            and config.analysis.projection_path is None
        ):
            raise ValueError(
                "analysis.projection_path is required when projection='load'."
            )
        if (
            config.analysis.projection == "load"
            and config.analysis.projection_path is not None
            and not config.analysis.projection_path.is_file()
        ):
            raise ValueError(
                "PCA projection file does not exist: "
                f"{config.analysis.projection_path}."
            )

    if "cmap" in applicable["visualization"]:
        _validate_colormap(config.visualization.cmap)
    if (
        "background" in applicable["visualization"]
        and config.visualization.background is not None
    ):
        _validate_color(config.visualization.background, "visualization.background")
    if (
        method == "patch_pca"
        and config.video is None
        and len(config.input.paths) == 1
        and config.output.grids
        and not config.output.heatmaps
        and not config.output.raw_arrays
    ):
        warn(
            "A single-image, grids-only patch-PCA run will export a one-tile "
            "comparison grid.",
            UserWarning,
            stacklevel=2,
        )


def config_to_dict(config: VisionLensConfig) -> dict[str, Any]:
    resolved: dict[str, dict[str, Any]] = {}
    resolved["input"] = {"files": [str(path) for path in config.input.paths]}
    resolved["model"] = {
        "architecture": config.model.architecture,
        "backend": config.model.backend,
        "name": config.model.name,
        "pretrained": config.model.pretrained,
        "options": config.model.options,
    }
    resolved["preprocessing"] = {
        "image_size": config.preprocessing.image_size,
        "resize": config.preprocessing.resize,
        "crop": config.preprocessing.crop,
        "pad": config.preprocessing.pad,
        "interpolation": config.preprocessing.interpolation,
        "normalize": config.preprocessing.normalize,
        "mean": (
            None
            if config.preprocessing.mean is None
            else list(config.preprocessing.mean)
        ),
        "std": (
            None if config.preprocessing.std is None else list(config.preprocessing.std)
        ),
    }
    resolved["analysis"] = _analysis_to_dict(config.analysis)
    resolved["runtime"] = {
        "batch_size": config.runtime.batch_size,
        "device": config.runtime.device,
        "workers": config.runtime.workers,
        "precision": config.runtime.precision,
        "seed": config.runtime.seed,
    }
    resolved["visualization"] = {
        "tile_size": (
            None
            if config.visualization.tile_size is None
            else list(config.visualization.tile_size)
        ),
        "columns": config.visualization.columns,
        "items_per_grid": config.visualization.items_per_grid,
        "spacing": config.visualization.spacing,
        "padding": config.visualization.padding,
        "labels": config.visualization.labels,
        "background": config.visualization.background,
        "dpi": config.visualization.dpi,
        "output_size": (
            list(config.visualization.output_size)
            if isinstance(config.visualization.output_size, tuple)
            else config.visualization.output_size
        ),
        "interpolation": config.visualization.interpolation,
        "anyup_query_chunk_size": config.visualization.anyup_query_chunk_size,
        "overlay_alpha": config.visualization.overlay_alpha,
        "overlay_alpha_curve": (
            None
            if config.visualization.overlay_alpha_curve is None
            else {
                "steepness": config.visualization.overlay_alpha_curve.steepness,
                "midpoint": config.visualization.overlay_alpha_curve.midpoint,
            }
        ),
        "cmap": config.visualization.cmap,
        "cmap_black": (
            None
            if config.visualization.cmap_black is None
            else {
                "threshold": config.visualization.cmap_black[0],
                "blend_width": config.visualization.cmap_black[1],
                "transparent": config.visualization.cmap_black[2],
            }
        ),
        "grid_format": config.visualization.grid_format,
        "normalization": config.visualization.normalization,
        "normalization_range": (
            None
            if config.visualization.normalization_range is None
            else list(config.visualization.normalization_range)
        ),
    }
    resolved["output"] = {
        "directory": str(config.output.directory),
        "heatmaps": config.output.heatmaps,
        "overlays": config.output.overlays,
        "grids": config.output.grids,
        "raw_arrays": config.output.raw_arrays,
        "image_format": config.output.image_format,
        "raw_format": config.output.raw_format,
        "overwrite": config.output.overwrite,
    }
    if config.video is not None:
        resolved["video"] = {
            "start_time": config.video.start_time,
            "end_time": config.video.end_time,
            "sampling_rate": config.video.sampling_rate,
            "frame_limit": config.video.frame_limit,
            "pca_fit_frames": config.video.pca_fit_frames,
            "temporal_smoothing": config.video.temporal_smoothing,
            "codec": config.video.codec,
        }

    applicable = _applicable_setting_keys(config)
    return {
        section: {
            key: value
            for key, value in values.items()
            if key in applicable[section]
        }
        for section, values in resolved.items()
    }


def resolved_config_yaml(config: VisionLensConfig) -> str:
    return yaml.safe_dump(config_to_dict(config), sort_keys=False)


def _parse_input(section: dict[str, Any], base_dir: Path) -> InputConfig:
    raw_files = section.get("files", [])
    raw_folders = section.get("folders", [])
    files = tuple(
        _resolve_path(path, base_dir) for path in _list(raw_files, "input.files")
    )
    folders = tuple(
        _resolve_path(path, base_dir) for path in _list(raw_folders, "input.folders")
    )
    patterns = tuple(
        _non_empty_string(pattern, "input.patterns")
        for pattern in _list(
            section.get("patterns", list(DEFAULT_INPUT_PATTERNS)),
            "input.patterns",
            allow_empty=False,
        )
    )
    recursive = _bool(section.get("recursive", False), "input.recursive")
    limit = _optional_positive_int(section.get("limit"), "input.limit")

    missing_files = [path for path in files if not path.is_file()]
    if missing_files:
        paths = ", ".join(str(path) for path in missing_files)
        raise ValueError(f"Input file(s) do not exist: {paths}.")
    invalid_folders = [path for path in folders if not path.is_dir()]
    if invalid_folders:
        paths = ", ".join(str(path) for path in invalid_folders)
        raise ValueError(f"Input folder(s) do not exist: {paths}.")

    selected = list(files)
    for folder in folders:
        for pattern in patterns:
            matches = folder.rglob(pattern) if recursive else folder.glob(pattern)
            selected.extend(
                sorted(path.resolve() for path in matches if path.is_file())
            )

    unique = tuple(dict.fromkeys(selected))
    if limit is not None:
        unique = unique[:limit]
    if not unique:
        raise ValueError("input must select at least one existing file.")

    return InputConfig(paths=unique)


def _parse_analysis(
    section: dict[str, Any],
    method: AnalysisMethod,
    base_dir: Path,
    *,
    is_video: bool = False,
) -> AnalysisConfig:
    if method in {"attention", "rollout"}:
        return AnalysisConfig(
            method=method,
            layers=_attention_layers(section.get("layers")),
            heads=_optional_non_negative_ints(
                section.get("heads"),
                "analysis.heads",
            ),
            head_fusion=_head_fusion(section.get("head_fusion", "mean")),
        )
    if method == "gradcam":
        target_layer = section.get("target_layer")
        return AnalysisConfig(
            method=method,
            target_layer=(
                None
                if target_layer is None
                else _optional_str(target_layer, "analysis.target_layer")
            ),
            target_class=_optional_non_negative_int(
                section.get("target_class"),
                "analysis.target_class",
            ),
        )
    projection = _choice(
        section.get("projection", "fit"),
        "analysis.projection",
        {"fit", "load"},
    )
    if projection == "load":
        return AnalysisConfig(
            method=method,
            projection=projection,
            projection_path=_optional_path(
                section.get("projection_path"),
                base_dir,
                "analysis.projection_path",
            ),
        )
    save_projection = _optional_path(
        section.get("save_projection"),
        base_dir,
        "analysis.save_projection",
    )
    if is_video:
        return AnalysisConfig(
            method=method,
            projection=projection,
            save_projection=save_projection,
        )
    foreground_separation = _bool(
        section.get("foreground_separation", True),
        "analysis.foreground_separation",
    )
    if not foreground_separation:
        return AnalysisConfig(
            method=method,
            foreground_separation=False,
            projection=projection,
            save_projection=save_projection,
        )
    foreground_threshold = _foreground_threshold(
        section.get("foreground_threshold", 0.5),
    )
    foreground_side = _foreground_side(section.get("foreground_side", "high"))
    rgb_fit_scope = _choice(
        section.get("rgb_fit_scope", "foreground"),
        "analysis.rgb_fit_scope",
        {"foreground", "all"},
    )
    return AnalysisConfig(
        method=method,
        foreground_separation=foreground_separation,
        foreground_threshold=foreground_threshold,
        foreground_side=foreground_side,
        rgb_fit_scope=rgb_fit_scope,
        projection=projection,
        save_projection=save_projection,
    )


def _analysis_to_dict(
    analysis: AnalysisConfig,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {"method": analysis.method}
    if analysis.method in {"attention", "rollout"}:
        resolved["layers"] = (
            analysis.layers if analysis.layers == "all" else list(analysis.layers or ())
        )
        resolved["heads"] = None if analysis.heads is None else list(analysis.heads)
        resolved["head_fusion"] = analysis.head_fusion
    elif analysis.method == "gradcam":
        resolved["target_layer"] = analysis.target_layer
        resolved["target_class"] = analysis.target_class
    else:
        resolved["projection"] = analysis.projection
        if analysis.projection == "load":
            resolved["projection_path"] = (
                None
                if analysis.projection_path is None
                else str(analysis.projection_path)
            )
            return resolved
        resolved["foreground_separation"] = analysis.foreground_separation
        resolved["foreground_threshold"] = analysis.foreground_threshold
        resolved["foreground_side"] = analysis.foreground_side
        resolved["rgb_fit_scope"] = analysis.rgb_fit_scope
        resolved["save_projection"] = (
            None if analysis.save_projection is None else str(analysis.save_projection)
        )
    return resolved


def _validate_keys(config: dict[str, Any]) -> None:
    _reject_unknown_keys("top level", config, TOP_LEVEL_KEYS)
    for section_name, allowed in SECTION_KEYS.items():
        if section_name == "output" and "video" in config:
            allowed = VIDEO_OUTPUT_KEYS
        _reject_unknown_keys(section_name, _section(config, section_name), allowed)


def _applicable_setting_keys(config: VisionLensConfig) -> dict[str, set[str]]:
    method = config.analysis.method
    is_video = config.video is not None
    rendered = config.output.heatmaps or config.output.overlays or config.output.grids
    spatial_output = rendered or (
        config.output.raw_arrays and (method != "patch_pca" or not is_video)
    )

    preprocessing = {"image_size", "interpolation", "normalize"}
    if not is_video:
        preprocessing.update({"resize", "crop", "pad"})
    if config.preprocessing.normalize:
        preprocessing.update({"mean", "std"})

    if method == "attention":
        analysis = {"method", "layers", "heads", "head_fusion"}
    elif method == "rollout":
        analysis = {"method", "layers"}
        if not is_video and config.output.grids:
            analysis.update({"heads", "head_fusion"})
    elif method == "gradcam":
        analysis = {"method", "target_layer", "target_class"}
    elif config.analysis.projection == "load":
        analysis = {"method", "projection", "projection_path"}
    else:
        analysis = {"method", "projection", "save_projection"}
        if not is_video:
            analysis.add("foreground_separation")
            if config.analysis.foreground_separation:
                analysis.update(
                    {"foreground_threshold", "foreground_side", "rgb_fit_scope"}
                )

    runtime = {"batch_size", "device", "precision", "seed"}
    if not is_video:
        runtime.add("workers")

    visualization: set[str] = set()
    if spatial_output:
        visualization.update({"output_size", "interpolation"})
        if config.visualization.interpolation in {
            "anyup",
            "anyup_mask",
            "anyup_soft",
            "anyup_soft_mask",
        }:
            visualization.add("anyup_query_chunk_size")
    if method != "patch_pca" and rendered:
        visualization.update({"normalization", "cmap", "cmap_black"})
        if config.visualization.normalization == "fixed":
            visualization.add("normalization_range")
        if config.output.overlays or config.output.grids:
            visualization.update({"overlay_alpha", "overlay_alpha_curve"})
    if not is_video and config.output.grids:
        visualization.update(GRID_VISUALIZATION_KEYS)
        if method == "patch_pca":
            visualization.discard("labels")

    output = {"directory", "heatmaps", "raw_arrays", "overwrite"}
    if method != "patch_pca":
        output.add("overlays")
    if not is_video:
        output.add("grids")
    if not is_video and (config.output.heatmaps or config.output.overlays):
        output.add("image_format")
    if config.output.raw_arrays:
        output.add("raw_format")

    video: set[str] = set()
    if is_video:
        video.update({"start_time", "end_time", "sampling_rate", "frame_limit"})
        if method == "patch_pca" and config.analysis.projection == "fit":
            video.add("pca_fit_frames")
        if method != "patch_pca" or config.output.heatmaps:
            video.add("temporal_smoothing")
        if config.output.heatmaps or config.output.overlays:
            video.add("codec")

    return {
        "input": set(SECTION_KEYS["input"]),
        "model": set(SECTION_KEYS["model"]),
        "preprocessing": preprocessing,
        "analysis": analysis,
        "runtime": runtime,
        "visualization": visualization,
        "output": output,
        "video": video,
    }


def _validate_applicable_settings(
    raw_config: dict[str, Any],
    config: VisionLensConfig,
) -> None:
    input_section = _section(raw_config, "input")
    if not input_section.get("folders"):
        _reject_present_keys(
            "input",
            input_section,
            {"patterns", "recursive"},
            "input.patterns and input.recursive require input.folders",
        )

    applicable = _applicable_setting_keys(config)
    analysis_section = _section(raw_config, "analysis")
    _reject_present_keys(
        "analysis",
        analysis_section,
        set(analysis_section) - applicable["analysis"],
        "the settings do not affect this workflow",
    )
    for section_name in SECTION_KEYS:
        section = _section(raw_config, section_name)
        present_but_unused = set(section) - applicable[section_name]
        if present_but_unused:
            _reject_present_keys(
                section_name,
                section,
                present_but_unused,
                "the settings do not affect this workflow",
            )


def _reject_present_keys(
    section_name: str,
    section: dict[str, Any],
    keys: set[str],
    reason: str,
) -> None:
    present = sorted(keys & section.keys())
    if present:
        qualified = ", ".join(f"{section_name}.{key}" for key in present)
        raise ValueError(f"Setting(s) {qualified} are not applicable: {reason}.")


def _reject_unknown_keys(
    section_name: str,
    section: dict[str, Any],
    allowed: set[str],
) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        keys = ", ".join(unknown)
        allowed_keys = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unknown key(s) in {section_name}: {keys}. Allowed keys: {allowed_keys}."
        )


def _validate_colormap(name: str) -> None:
    from matplotlib import colormaps

    if name not in colormaps:
        raise ValueError(
            "visualization.cmap is not a known Matplotlib colormap: " f"{name!r}."
        )


def _validate_color(value: str, field_name: str) -> None:
    from matplotlib.colors import is_color_like

    if not is_color_like(value):
        raise ValueError(f"{field_name} is not a valid Matplotlib color: {value!r}.")


def _validate_known_attention_constraints(config: VisionLensConfig) -> None:
    analysis = config.analysis
    depth = KNOWN_VIT_DEPTHS.get(config.model.name)
    if depth is not None and analysis.layers != "all":
        assert analysis.layers is not None
        invalid_layers = [layer for layer in analysis.layers if layer >= depth]
        if invalid_layers:
            raise ValueError(
                f"analysis.layers contains {invalid_layers}; model "
                f"{config.model.name!r} has layers 0 through {depth - 1}."
            )

    head_count = KNOWN_VIT_HEADS.get(config.model.name)
    if head_count is not None and analysis.heads is not None:
        invalid_heads = [head for head in analysis.heads if head >= head_count]
        if invalid_heads:
            raise ValueError(
                f"analysis.heads contains {invalid_heads}; model "
                f"{config.model.name!r} has heads 0 through {head_count - 1}."
            )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _validate_mode_overrides(
    raw_config: dict[str, Any],
    overrides: dict[str, Any],
) -> None:
    if not overrides:
        return

    raw_analysis = _section(raw_config, "analysis")
    override_analysis = _section(overrides, "analysis")
    mode_settings = {
        "method": raw_analysis.get("method", "attention"),
        "projection": raw_analysis.get("projection", "fit"),
        "foreground_separation": raw_analysis.get("foreground_separation", True),
    }
    changed = [
        f"analysis.{key}"
        for key, current in mode_settings.items()
        if key in override_analysis and override_analysis[key] != current
    ]
    if "video" in overrides and "video" not in raw_config:
        changed.append("video workflow")
    if changed:
        settings = ", ".join(changed)
        raise ValueError(
            f"CLI/config overrides cannot switch conditional mode setting(s): "
            f"{settings}. Edit or create a configuration file for that workflow."
        )


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be a mapping.")
    return section


def _optional_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, field_name)


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} values must be non-empty strings.")
    return value


def _required_str(section: dict[str, Any], key: str, section_name: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section_name}.{key} must be a non-empty string.")
    return value


def _list(value: Any, field_name: str, *, allow_empty: bool = True) -> list[Any]:
    if not isinstance(value, list) or (not value and not allow_empty):
        suffix = "a non-empty list" if not allow_empty else "a list"
        raise ValueError(f"{field_name} must be {suffix}.")
    return value


def _attention_layers(value: Any) -> AttentionLayers:
    if value == "all":
        return "all"
    if not isinstance(value, list) or not value:
        raise ValueError("analysis.layers must be `all` or a non-empty list.")
    return tuple(_non_negative_ints(value, "analysis.layers"))


def _optional_mapping(value: Any, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping or null.")
    return dict(value)


def _resolve_path(path: Any, base_dir: Path) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Paths must be non-empty strings.")
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved.resolve()


def _project_root(start: Path) -> Path:
    for candidate in (start, Path.cwd()):
        path = candidate.resolve()
        for directory in (path, *path.parents):
            if (directory / "pyproject.toml").is_file():
                return directory
    return Path.cwd().resolve()


def _optional_path(value: Any, base_dir: Path, field_name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path or null.")
    return _resolve_path(value, base_dir)


def _non_negative_ints(values: list[Any], field_name: str) -> list[int]:
    parsed = [_non_negative_int(value, field_name) for value in values]
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return parsed


def _optional_non_negative_ints(value: Any, field_name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list or null.")
    return tuple(_non_negative_ints(value, field_name))


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} values must be non-negative integers.")
    return value


def _non_negative_int_value(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _optional_non_negative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int_value(value, field_name)


def _optional_seed(value: Any, field_name: str) -> int | None:
    seed = _optional_non_negative_int(value, field_name)
    if seed is not None and seed > 2**32 - 1:
        raise ValueError(f"{field_name} must be at most 2**32 - 1.")
    return seed


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be true or false.")
    return value


def _optional_bool(value: Any, field_name: str) -> bool | None:
    if value is None:
        return None
    return _bool(value, field_name)


def _choice(value: Any, field_name: str, allowed: set[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {options}.")
    return value


def _optional_choice(value: Any, field_name: str, allowed: set[str]) -> str | None:
    if value is None:
        return None
    return _choice(value, field_name, allowed)


def _optional_triplet(
    value: Any,
    field_name: str,
    *,
    positive: bool = False,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field_name} must be a list of three numbers or null.")
    parsed = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(f"{field_name} must contain only numbers.")
        number = float(item)
        if not isfinite(number):
            raise ValueError(f"{field_name} values must be finite.")
        if positive and number <= 0:
            raise ValueError(f"{field_name} values must be greater than zero.")
        parsed.append(number)
    return (parsed[0], parsed[1], parsed[2])


def _optional_size(value: Any, field_name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        size = _positive_int(value, field_name)
        return (size, size)
    if isinstance(value, list) and len(value) == 2:
        return (
            _positive_int(value[0], field_name),
            _positive_int(value[1], field_name),
        )
    raise ValueError(
        f"{field_name} must be a positive integer, [width, height], or null."
    )


def _visualization_output_size(value: Any) -> VisualizationOutputSize:
    if value == "match":
        return "match"
    try:
        return _optional_size(value, "visualization.output_size")
    except ValueError as error:
        raise ValueError(
            "visualization.output_size must be 'match', a positive integer, "
            "[width, height], or null."
        ) from error


def _optional_range(value: Any, field_name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field_name} must be [minimum, maximum] or null.")
    if any(
        isinstance(item, bool) or not isinstance(item, int | float) for item in value
    ):
        raise ValueError(f"{field_name} must contain only numbers.")
    minimum, maximum = float(value[0]), float(value[1])
    if not isfinite(minimum) or not isfinite(maximum):
        raise ValueError(f"{field_name} values must be finite.")
    if maximum <= minimum:
        raise ValueError(f"{field_name} maximum must be greater than its minimum.")
    return (minimum, maximum)


def _analysis_method(value: Any) -> AnalysisMethod:
    return _choice(value, "analysis.method", set(ANALYSIS_KEYS))


def _head_fusion(value: Any) -> HeadFusion:
    return _choice(value, "analysis.head_fusion", {"mean", "max", "none"})


def _foreground_side(value: Any) -> Literal["high", "low"]:
    return _choice(value, "analysis.foreground_side", {"high", "low"})


def _device(value: Any) -> Device:
    return _choice(value, "runtime.device", {"auto", "cpu", "cuda", "mps"})


def _unit_interval(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return float(value)


def _foreground_threshold(value: Any) -> float | Literal["auto"]:
    if value == "auto":
        return "auto"
    return _unit_interval(value, "analysis.foreground_threshold")


def _overlay_alpha_curve(value: Any) -> OverlayAlphaCurveSpec | None:
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or "steepness" not in value
        or not set(value) <= {"steepness", "midpoint"}
    ):
        raise ValueError(
            "visualization.overlay_alpha_curve must be null or contain steepness "
            "and optional midpoint."
        )
    steepness = _positive_number(
        value["steepness"],
        "visualization.overlay_alpha_curve.steepness",
    )
    if not isfinite(steepness):
        raise ValueError("visualization.overlay_alpha_curve.steepness must be finite.")
    midpoint = _unit_interval(
        value.get("midpoint", 0.5),
        "visualization.overlay_alpha_curve.midpoint",
    )
    return OverlayAlphaCurveSpec(steepness=steepness, midpoint=midpoint)


def _cmap_black(value: Any) -> tuple[int, int, bool] | None:
    if value is None:
        return None
    required = {"threshold", "blend_width", "transparent"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            "visualization.cmap_black must be null or contain exactly "
            "threshold, blend_width, and transparent."
        )
    threshold = value["threshold"]
    blend_width = value["blend_width"]
    transparent = value["transparent"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int)
        or not 0 <= threshold <= 254
    ):
        raise ValueError("visualization.cmap_black.threshold must be from 0 to 254.")
    if (
        isinstance(blend_width, bool)
        or not isinstance(blend_width, int)
        or not 1 <= blend_width <= 255 - threshold
    ):
        raise ValueError(
            "visualization.cmap_black.blend_width must be from 1 to "
            "255 minus threshold."
        )
    if not isinstance(transparent, bool):
        raise ValueError("visualization.cmap_black.transparent must be a boolean.")
    return threshold, blend_width, transparent


def _non_negative_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a finite non-negative number.")
    number = float(value)
    if not isfinite(number) or number < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number.")
    return number


def _optional_non_negative_number(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _non_negative_number(value, field_name)


def _positive_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a finite positive number.")
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise ValueError(f"{field_name} must be a finite positive number.")
    return number


def _video_sampling_rate(value: Any) -> float | Literal["auto"]:
    if value == "auto":
        return "auto"
    return _positive_number(value, "video.sampling_rate")


def _grid_format(value: Any) -> str:
    return _choice(value, "visualization.grid_format", {"pdf", "png", "svg"})


def _image_format(value: Any) -> str:
    return _choice(value, "output.image_format", {"jpeg", "png", "tiff", "webp"})


def _raw_format(value: Any) -> str:
    return _choice(value, "output.raw_format", {"npy", "npz"})
