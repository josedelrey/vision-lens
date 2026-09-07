from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from vision_lens.presets import get_preset

HeadFusion = Literal["mean", "max", "none"]
Device = Literal["auto", "cpu", "cuda", "mps"]
AttentionLayers = Literal["all"] | tuple[int, ...]
AnalysisMethod = Literal["attention", "rollout", "gradcam", "patch_pca"]
Precision = Literal["float32", "float16", "bfloat16"]
NormalizationMode = Literal["per_map", "shared", "fixed"]

DEFAULT_INPUT_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.webp")

TOP_LEVEL_KEYS = {
    "preset",
    "input",
    "model",
    "preprocessing",
    "analysis",
    "runtime",
    "visualization",
    "output",
}
SECTION_KEYS = {
    "input": {"paths", "files", "folders", "patterns", "recursive", "limit"},
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
        "overlay_alpha",
        "cmap",
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
}
ANALYSIS_KEYS = {
    "attention": {"method", "layers", "heads", "head_fusion"},
    "rollout": {"method", "layers", "heads", "head_fusion"},
    "gradcam": {"method", "target_layer", "target_class"},
    "patch_pca": {
        "method",
        "foreground_threshold",
        "foreground_side",
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
    files: tuple[Path, ...] = ()
    folders: tuple[Path, ...] = ()
    patterns: tuple[str, ...] = DEFAULT_INPUT_PATTERNS
    recursive: bool = False
    limit: int | None = None


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
class AttentionConfig:
    layers: AttentionLayers
    heads: tuple[int, ...] | None = None
    head_fusion: HeadFusion = "mean"


@dataclass(frozen=True)
class PatchPCAConfig:
    foreground_threshold: float = 0.5
    foreground_side: Literal["high", "low"] = "high"


@dataclass(frozen=True)
class AnalysisConfig:
    method: AnalysisMethod
    layers: AttentionLayers | None = None
    heads: tuple[int, ...] | None = None
    head_fusion: HeadFusion | None = None
    target_layer: str | None = None
    target_class: int | None = None
    foreground_threshold: float | None = None
    foreground_side: Literal["high", "low"] | None = None
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
class VisualizationConfig:
    tile_size: tuple[int, int] | None = None
    columns: int | None = None
    items_per_grid: int | None = None
    spacing: int | None = None
    padding: int | None = None
    labels: bool | None = None
    background: str | None = None
    dpi: int | None = None
    overlay_alpha: float = 0.45
    cmap: str = "viridis"
    grid_format: str = "png"
    normalization: NormalizationMode = "per_map"
    normalization_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class VisionLensConfig:
    input: InputConfig
    model: ModelConfig
    preprocessing: PreprocessingConfig
    analysis: AnalysisConfig
    runtime: RuntimeConfig
    visualization: VisualizationConfig
    output: OutputConfig
    preset: str | None = None

    @property
    def task(self) -> str:
        return METHOD_TASKS[self.analysis.method]

    @property
    def images(self) -> InputConfig:
        """Compatibility alias for the pre-release Python API."""
        return self.input

    @property
    def attention(self) -> AttentionConfig | None:
        if self.analysis.method not in {"attention", "rollout"}:
            return None
        assert self.analysis.layers is not None
        assert self.analysis.head_fusion is not None
        return AttentionConfig(
            layers=self.analysis.layers,
            heads=self.analysis.heads,
            head_fusion=self.analysis.head_fusion,
        )

    @property
    def patch_pca(self) -> PatchPCAConfig | None:
        if self.analysis.method != "patch_pca":
            return None
        assert self.analysis.foreground_threshold is not None
        assert self.analysis.foreground_side is not None
        return PatchPCAConfig(
            foreground_threshold=self.analysis.foreground_threshold,
            foreground_side=self.analysis.foreground_side,
        )


def load_config(
    path: str | Path,
    *,
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> VisionLensConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    if not isinstance(raw_config, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")

    return parse_config(
        raw_config,
        base_dir=config_path.parent,
        preset=preset,
        overrides=overrides,
    )


def load_preset(
    name: str,
    *,
    base_dir: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> VisionLensConfig:
    return parse_config(
        {},
        base_dir=base_dir,
        preset=name,
        overrides=overrides,
    )


def parse_config(
    raw_config: dict[str, Any],
    base_dir: Path | None = None,
    *,
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> VisionLensConfig:
    base = Path.cwd().resolve() if base_dir is None else Path(base_dir).resolve()
    preset_name = _preset_name(raw_config, preset)
    resolved = get_preset(preset_name) if preset_name is not None else {}
    user_config = {key: value for key, value in raw_config.items() if key != "preset"}
    resolved = _deep_merge(resolved, user_config)
    resolved = _deep_merge(resolved, overrides or {})
    _validate_keys(resolved)

    input_section = _section(resolved, "input")
    model_section = _section(resolved, "model")
    preprocessing_section = _section(resolved, "preprocessing")
    analysis_section = _section(resolved, "analysis")
    runtime_section = _section(resolved, "runtime")
    visualization_section = _section(resolved, "visualization")
    output_section = _section(resolved, "output")
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
                preprocessing_section.get("image_size", 672),
                "preprocessing.image_size",
            ),
            resize=_choice(
                preprocessing_section.get("resize", "stretch"),
                "preprocessing.resize",
                {"stretch", "shortest", "longest", "none"},
            ),
            crop=_choice(
                preprocessing_section.get("crop", "none"),
                "preprocessing.crop",
                {"none", "center"},
            ),
            pad=_choice(
                preprocessing_section.get("pad", "none"),
                "preprocessing.pad",
                {"none", "center"},
            ),
            interpolation=_optional_choice(
                preprocessing_section.get("interpolation"),
                "preprocessing.interpolation",
                {"nearest", "bilinear", "bicubic", "lanczos"},
            ),
            normalize=_bool(
                preprocessing_section.get("normalize", True),
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
        analysis=_parse_analysis(analysis_section, method, base),
        runtime=RuntimeConfig(
            device=_device(runtime_section.get("device", "auto")),
            batch_size=_positive_int(
                runtime_section.get("batch_size", 8),
                "runtime.batch_size",
            ),
            workers=_non_negative_int_value(
                runtime_section.get("workers", 0),
                "runtime.workers",
            ),
            precision=_choice(
                runtime_section.get("precision", "float32"),
                "runtime.precision",
                {"float32", "float16", "bfloat16"},
            ),
            seed=_optional_non_negative_int(
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
            overlay_alpha=_unit_interval(
                visualization_section.get("overlay_alpha", 0.45),
                "visualization.overlay_alpha",
            ),
            cmap=_optional_str(
                visualization_section.get("cmap", "viridis"),
                "visualization.cmap",
            ),
            grid_format=_grid_format(visualization_section.get("grid_format", "png")),
            normalization=_choice(
                visualization_section.get("normalization", "per_map"),
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
            heatmaps=_bool(output_section.get("heatmaps", True), "output.heatmaps"),
            overlays=_bool(
                output_section.get("overlays", method != "patch_pca"),
                "output.overlays",
            ),
            grids=_bool(output_section.get("grids", True), "output.grids"),
            raw_arrays=_bool(
                output_section.get("raw_arrays", False),
                "output.raw_arrays",
            ),
            image_format=_image_format(output_section.get("image_format", "png")),
            raw_format=_raw_format(output_section.get("raw_format", "npy")),
            overwrite=_choice(
                output_section.get("overwrite", "error"),
                "output.overwrite",
                {"replace", "error", "skip"},
            ),
        ),
        preset=preset_name,
    )
    validate_config(config)
    return config


def validate_config(config: VisionLensConfig) -> None:
    missing_inputs = [path for path in config.input.paths if not path.is_file()]
    if missing_inputs:
        paths = ", ".join(str(path) for path in missing_inputs)
        raise ValueError(f"Input file(s) do not exist: {paths}.")

    method = config.analysis.method
    actual_pair = (config.model.architecture, config.model.backend)
    expected_pair = ("cnn", "torchvision") if method == "gradcam" else ("vit", "timm")
    if actual_pair != expected_pair:
        raise ValueError(
            f"analysis.method={method!r} requires model.architecture="
            f"{expected_pair[0]!r} and model.backend={expected_pair[1]!r}; got "
            f"architecture={actual_pair[0]!r}, backend={actual_pair[1]!r}."
        )

    options = config.model.options or {}
    if "img_size" in options:
        raise ValueError(
            "model.options.img_size is not allowed; use "
            "preprocessing.image_size as the authoritative input size."
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
        config.visualization.normalization == "fixed"
        and config.visualization.normalization_range is None
    ):
        raise ValueError(
            "visualization.normalization_range is required when normalization='fixed'."
        )
    if (
        config.visualization.normalization != "fixed"
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


def config_to_dict(config: VisionLensConfig) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    if config.preset is not None:
        resolved["preset"] = config.preset
    resolved["input"] = {
        "files": [str(path) for path in config.input.paths],
        "folders": [],
        "patterns": list(config.input.patterns),
        "recursive": False,
        "limit": None,
    }
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
        "mean": None
        if config.preprocessing.mean is None
        else list(config.preprocessing.mean),
        "std": None
        if config.preprocessing.std is None
        else list(config.preprocessing.std),
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
        "overlay_alpha": config.visualization.overlay_alpha,
        "cmap": config.visualization.cmap,
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
    return resolved


def resolved_config_yaml(config: VisionLensConfig) -> str:
    return yaml.safe_dump(config_to_dict(config), sort_keys=False)


def _parse_input(section: dict[str, Any], base_dir: Path) -> InputConfig:
    if "paths" in section and "files" in section:
        raise ValueError("input.paths and input.files cannot both be set.")

    raw_files = section.get("files", section.get("paths", []))
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

    return InputConfig(
        paths=unique,
        files=files,
        folders=folders,
        patterns=patterns,
        recursive=recursive,
        limit=limit,
    )


def _parse_analysis(
    section: dict[str, Any],
    method: AnalysisMethod,
    base_dir: Path,
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
    return AnalysisConfig(
        method=method,
        foreground_threshold=_unit_interval(
            section.get("foreground_threshold", 0.5),
            "analysis.foreground_threshold",
        ),
        foreground_side=_foreground_side(section.get("foreground_side", "high")),
        projection=_choice(
            section.get("projection", "fit"),
            "analysis.projection",
            {"fit", "load"},
        ),
        projection_path=_optional_path(
            section.get("projection_path"),
            base_dir,
            "analysis.projection_path",
        ),
        save_projection=_optional_path(
            section.get("save_projection"),
            base_dir,
            "analysis.save_projection",
        ),
    )


def _analysis_to_dict(analysis: AnalysisConfig) -> dict[str, Any]:
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
        resolved["foreground_threshold"] = analysis.foreground_threshold
        resolved["foreground_side"] = analysis.foreground_side
        resolved["projection"] = analysis.projection
        resolved["projection_path"] = (
            None if analysis.projection_path is None else str(analysis.projection_path)
        )
        resolved["save_projection"] = (
            None if analysis.save_projection is None else str(analysis.save_projection)
        )
    return resolved


def _validate_keys(config: dict[str, Any]) -> None:
    _reject_unknown_keys("top level", config, TOP_LEVEL_KEYS)
    for section_name, allowed in SECTION_KEYS.items():
        _reject_unknown_keys(section_name, _section(config, section_name), allowed)


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


def _preset_name(raw_config: dict[str, Any], selected: str | None) -> str | None:
    configured = raw_config.get("preset")
    if configured is not None and (
        not isinstance(configured, str) or not configured.strip()
    ):
        raise ValueError("preset must be a non-empty string.")
    if selected is not None and (not isinstance(selected, str) or not selected.strip()):
        raise ValueError("preset must be a non-empty string.")
    return selected if selected is not None else configured


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


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


def _required_list(section: dict[str, Any], key: str, section_name: str) -> list[Any]:
    value = section.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{section_name}.{key} must be a non-empty list.")
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
    if value not in allowed:
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
    if maximum <= minimum:
        raise ValueError(f"{field_name} maximum must be greater than its minimum.")
    return (minimum, maximum)


def _analysis_method(value: Any) -> AnalysisMethod:
    allowed = set(ANALYSIS_KEYS)
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"analysis.method must be one of: {options}.")
    return value


def _head_fusion(value: Any) -> HeadFusion:
    allowed = {"mean", "max", "none"}
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"analysis.head_fusion must be one of: {options}.")
    return value


def _foreground_side(value: Any) -> Literal["high", "low"]:
    allowed = {"high", "low"}
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"analysis.foreground_side must be one of: {options}.")
    return value


def _device(value: Any) -> Device:
    allowed = {"auto", "cpu", "cuda", "mps"}
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"runtime.device must be one of: {options}.")
    return value


def _unit_interval(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return float(value)


def _grid_format(value: Any) -> str:
    allowed = {"pdf", "png", "svg"}
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"visualization.grid_format must be one of: {options}.")
    return value


def _image_format(value: Any) -> str:
    allowed = {"jpeg", "png", "tiff", "webp"}
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"output.image_format must be one of: {options}.")
    return value


def _raw_format(value: Any) -> str:
    allowed = {"npy", "npz"}
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"output.raw_format must be one of: {options}.")
    return value
