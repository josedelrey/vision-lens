"""Convert between external configuration data and the resolved schema."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import fields, is_dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

import yaml

from vision_lens.config.schema import (
    ALPHA_FORMAT_CHOICES,
    ANALYSIS_DEFAULTS,
    ANALYSIS_KEYS,
    CROP_CHOICES,
    DEFAULT_ANALYSIS_METHOD,
    DEFAULT_INPUT_PATTERNS,
    DEVICE_CHOICES,
    FOREGROUND_SIDE_CHOICES,
    GRID_FORMAT_CHOICES,
    HEAD_FUSION_CHOICES,
    IMAGE_FORMAT_CHOICES,
    NORMALIZATION_CHOICES,
    OVERWRITE_CHOICES,
    PAD_CHOICES,
    PATCH_PCA_IMAGE_FIT_DEFAULTS,
    PRECISION_CHOICES,
    PREPROCESSING_INTERPOLATION_CHOICES,
    PROJECTION_CHOICES,
    RAW_FORMAT_CHOICES,
    RESIZE_CHOICES,
    RGB_FIT_SCOPE_CHOICES,
    ROLLOUT_GRID_CHOICES,
    SECTION_DEFAULTS,
    SECTION_KEYS,
    TOP_LEVEL_KEYS,
    VIDEO_OUTPUT_KEYS,
    VISUALIZATION_INTERPOLATION_CHOICES,
    AnalysisConfig,
    AnalysisMethod,
    AttentionAnalysisConfig,
    AttentionLayers,
    Device,
    GradCAMAnalysisConfig,
    HeadFusion,
    InputConfig,
    ModelConfig,
    OutputConfig,
    OverlayAlphaCurveSpec,
    PatchPCAAnalysisConfig,
    PreprocessingConfig,
    RolloutAnalysisConfig,
    RuntimeConfig,
    VideoConfig,
    VisionLensConfig,
    VisualizationConfig,
    VisualizationOutputSize,
    config_section,
)
from vision_lens.config.validation import (
    applicable_setting_keys,
    finite_number,
    validate_applicable_settings,
    validate_config,
)


def load_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> VisionLensConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if raw_config is None:
        raw_config = {}
    elif not isinstance(raw_config, Mapping):
        raise ValueError("Config file must contain a YAML mapping at the top level.")
    return parse_config(
        raw_config,
        base_dir=_project_root(config_path.parent),
        overrides=overrides,
    )


def parse_config(
    raw_config: Mapping[str, Any],
    base_dir: Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> VisionLensConfig:
    if not isinstance(raw_config, Mapping):
        raise ValueError("raw_config must be a mapping.")
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ValueError("overrides must be a mapping or None.")
    base = _project_root(Path.cwd()) if base_dir is None else Path(base_dir).resolve()
    resolved = _deep_merge(raw_config, overrides or {})
    _validate_keys(resolved)
    _validate_mode_overrides(raw_config, overrides or {})

    input_section = config_section(resolved, "input")
    model_section = config_section(resolved, "model")
    preprocessing_section = config_section(resolved, "preprocessing")
    analysis_section = config_section(resolved, "analysis")
    runtime_section = config_section(resolved, "runtime")
    visualization_section = config_section(resolved, "visualization")
    output_section = config_section(resolved, "output")
    video_section = config_section(resolved, "video")
    method = _analysis_method(analysis_section.get("method", DEFAULT_ANALYSIS_METHOD))
    _reject_unknown_keys("analysis", analysis_section, ANALYSIS_KEYS[method])

    config = VisionLensConfig(
        input=_parse_input(input_section, base),
        model=ModelConfig(
            architecture=_required_str(model_section, "architecture", "model"),
            backend=_required_str(model_section, "backend", "model"),
            name=_required_str(model_section, "name", "model"),
            pretrained=_bool(
                model_section.get(
                    "pretrained", SECTION_DEFAULTS["model"]["pretrained"]
                ),
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
                    "image_size", SECTION_DEFAULTS["preprocessing"]["image_size"]
                ),
                "preprocessing.image_size",
            ),
            resize=_choice(
                preprocessing_section.get(
                    "resize", SECTION_DEFAULTS["preprocessing"]["resize"]
                ),
                "preprocessing.resize",
                RESIZE_CHOICES,
            ),
            crop=_choice(
                preprocessing_section.get(
                    "crop", SECTION_DEFAULTS["preprocessing"]["crop"]
                ),
                "preprocessing.crop",
                CROP_CHOICES,
            ),
            pad=_choice(
                preprocessing_section.get(
                    "pad", SECTION_DEFAULTS["preprocessing"]["pad"]
                ),
                "preprocessing.pad",
                PAD_CHOICES,
            ),
            interpolation=_optional_choice(
                preprocessing_section.get("interpolation"),
                "preprocessing.interpolation",
                PREPROCESSING_INTERPOLATION_CHOICES,
            ),
            normalize=_bool(
                preprocessing_section.get(
                    "normalize", SECTION_DEFAULTS["preprocessing"]["normalize"]
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
                runtime_section.get("device", SECTION_DEFAULTS["runtime"]["device"])
            ),
            batch_size=_positive_int(
                runtime_section.get(
                    "batch_size", SECTION_DEFAULTS["runtime"]["batch_size"]
                ),
                "runtime.batch_size",
            ),
            workers=_non_negative_int_value(
                runtime_section.get("workers", SECTION_DEFAULTS["runtime"]["workers"]),
                "runtime.workers",
            ),
            precision=_choice(
                runtime_section.get(
                    "precision", SECTION_DEFAULTS["runtime"]["precision"]
                ),
                "runtime.precision",
                PRECISION_CHOICES,
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
                    "interpolation",
                    SECTION_DEFAULTS["visualization"]["interpolation"],
                ),
                "visualization.interpolation",
                VISUALIZATION_INTERPOLATION_CHOICES,
            ),
            anyup_query_chunk_size=_optional_positive_int(
                visualization_section.get("anyup_query_chunk_size"),
                "visualization.anyup_query_chunk_size",
            ),
            overlay_alpha=_unit_interval(
                visualization_section.get(
                    "overlay_alpha",
                    SECTION_DEFAULTS["visualization"]["overlay_alpha"],
                ),
                "visualization.overlay_alpha",
            ),
            overlay_alpha_curve=_overlay_alpha_curve(
                visualization_section.get("overlay_alpha_curve")
            ),
            cmap=_non_empty_string(
                visualization_section.get(
                    "cmap", SECTION_DEFAULTS["visualization"]["cmap"]
                ),
                "visualization.cmap",
            ),
            cmap_black=_cmap_black(visualization_section.get("cmap_black")),
            grid_format=_grid_format(
                visualization_section.get(
                    "grid_format", SECTION_DEFAULTS["visualization"]["grid_format"]
                )
            ),
            rollout_grid=_choice(
                visualization_section.get(
                    "rollout_grid",
                    SECTION_DEFAULTS["visualization"]["rollout_grid"],
                ),
                "visualization.rollout_grid",
                ROLLOUT_GRID_CHOICES,
            ),
            normalization=_choice(
                visualization_section.get(
                    "normalization",
                    SECTION_DEFAULTS["visualization"]["normalization"],
                ),
                "visualization.normalization",
                NORMALIZATION_CHOICES,
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
                output_section.get("heatmaps", SECTION_DEFAULTS["output"]["heatmaps"]),
                "output.heatmaps",
            ),
            overlays=_bool(
                output_section.get(
                    "overlays",
                    method != "patch_pca" and SECTION_DEFAULTS["output"]["overlays"],
                ),
                "output.overlays",
            ),
            transparent_overlays=_bool(
                output_section.get(
                    "transparent_overlays",
                    SECTION_DEFAULTS["output"]["transparent_overlays"],
                ),
                "output.transparent_overlays",
            ),
            grids=(
                False
                if "video" in resolved
                else _bool(
                    output_section.get("grids", SECTION_DEFAULTS["output"]["grids"]),
                    "output.grids",
                )
            ),
            raw_arrays=_bool(
                output_section.get(
                    "raw_arrays", SECTION_DEFAULTS["output"]["raw_arrays"]
                ),
                "output.raw_arrays",
            ),
            image_format=_image_format(
                output_section.get(
                    "image_format", SECTION_DEFAULTS["output"]["image_format"]
                )
            ),
            raw_format=_raw_format(
                output_section.get(
                    "raw_format", SECTION_DEFAULTS["output"]["raw_format"]
                )
            ),
            overwrite=_choice(
                output_section.get(
                    "overwrite", SECTION_DEFAULTS["output"]["overwrite"]
                ),
                "output.overwrite",
                OVERWRITE_CHOICES,
            ),
        ),
        video=(
            None
            if "video" not in resolved
            else VideoConfig(
                start_time=_non_negative_number(
                    video_section.get(
                        "start_time", SECTION_DEFAULTS["video"]["start_time"]
                    ),
                    "video.start_time",
                ),
                end_time=_optional_non_negative_number(
                    video_section.get("end_time"),
                    "video.end_time",
                ),
                sampling_rate=_video_sampling_rate(
                    video_section.get(
                        "sampling_rate", SECTION_DEFAULTS["video"]["sampling_rate"]
                    ),
                ),
                frame_limit=_optional_positive_int(
                    video_section.get("frame_limit"),
                    "video.frame_limit",
                ),
                pca_fit_frames=_positive_int(
                    video_section.get(
                        "pca_fit_frames", SECTION_DEFAULTS["video"]["pca_fit_frames"]
                    ),
                    "video.pca_fit_frames",
                ),
                temporal_smoothing=_unit_interval(
                    video_section.get(
                        "temporal_smoothing",
                        SECTION_DEFAULTS["video"]["temporal_smoothing"],
                    ),
                    "video.temporal_smoothing",
                ),
                codec=_non_empty_string(
                    video_section.get("codec", SECTION_DEFAULTS["video"]["codec"]),
                    "video.codec",
                ),
                alpha_format=_choice(
                    video_section.get(
                        "alpha_format", SECTION_DEFAULTS["video"]["alpha_format"]
                    ),
                    "video.alpha_format",
                    ALPHA_FORMAT_CHOICES,
                ),
            )
        ),
    )
    validate_applicable_settings(resolved, config)
    validate_config(config)
    return config


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
        _relative_glob_pattern(pattern, "input.patterns")
        for pattern in _list(
            section.get("patterns", list(DEFAULT_INPUT_PATTERNS)),
            "input.patterns",
            allow_empty=False,
        )
    )
    recursive = _bool(section.get("recursive", False), "input.recursive")
    limit = _optional_positive_int(section.get("limit"), "input.limit")

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
        config_type = (
            AttentionAnalysisConfig if method == "attention" else RolloutAnalysisConfig
        )
        return config_type(
            layers=_attention_layers(section.get("layers")),
            heads=_optional_non_negative_ints(
                section.get("heads"),
                "analysis.heads",
            ),
            head_fusion=_head_fusion(
                section.get("head_fusion", ANALYSIS_DEFAULTS[method]["head_fusion"])
            ),
        )
    if method == "gradcam":
        target_layer = section.get("target_layer")
        return GradCAMAnalysisConfig(
            target_layer=(
                None
                if target_layer is None
                else _non_empty_string(target_layer, "analysis.target_layer")
            ),
            target_class=_optional_non_negative_int(
                section.get("target_class"),
                "analysis.target_class",
            ),
        )
    projection = _choice(
        section.get("projection", ANALYSIS_DEFAULTS[method]["projection"]),
        "analysis.projection",
        PROJECTION_CHOICES,
    )
    if projection == "load":
        return PatchPCAAnalysisConfig(
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
        return PatchPCAAnalysisConfig(
            projection=projection,
            save_projection=save_projection,
        )
    foreground_separation = _bool(
        section.get(
            "foreground_separation",
            PATCH_PCA_IMAGE_FIT_DEFAULTS["foreground_separation"],
        ),
        "analysis.foreground_separation",
    )
    if not foreground_separation:
        return PatchPCAAnalysisConfig(
            foreground_separation=False,
            projection=projection,
            save_projection=save_projection,
        )
    foreground_threshold = _foreground_threshold(
        section.get(
            "foreground_threshold",
            PATCH_PCA_IMAGE_FIT_DEFAULTS["foreground_threshold"],
        ),
    )
    foreground_side = _foreground_side(
        section.get("foreground_side", PATCH_PCA_IMAGE_FIT_DEFAULTS["foreground_side"])
    )
    rgb_fit_scope = _choice(
        section.get("rgb_fit_scope", PATCH_PCA_IMAGE_FIT_DEFAULTS["rgb_fit_scope"]),
        "analysis.rgb_fit_scope",
        RGB_FIT_SCOPE_CHOICES,
    )
    return PatchPCAAnalysisConfig(
        foreground_separation=foreground_separation,
        foreground_threshold=foreground_threshold,
        foreground_side=foreground_side,
        rgb_fit_scope=rgb_fit_scope,
        projection=projection,
        save_projection=save_projection,
    )


def _validate_keys(config: dict[str, Any]) -> None:
    _reject_unknown_keys("top level", config, TOP_LEVEL_KEYS)
    for section_name, allowed in SECTION_KEYS.items():
        if section_name == "output" and "video" in config:
            allowed = VIDEO_OUTPUT_KEYS
        _reject_unknown_keys(
            section_name,
            config_section(config, section_name),
            allowed,
        )


def _reject_unknown_keys(
    section_name: str,
    section: dict[str, Any],
    allowed: set[str],
) -> None:
    if not all(isinstance(key, str) for key in section):
        raise ValueError(f"{section_name} keys must be strings.")
    unknown = sorted(set(section) - allowed)
    if unknown:
        keys = ", ".join(unknown)
        allowed_keys = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unknown key(s) in {section_name}: {keys}. Allowed keys: {allowed_keys}."
        )


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _validate_mode_overrides(
    raw_config: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> None:
    if not overrides:
        return

    raw_analysis = config_section(raw_config, "analysis")
    override_analysis = config_section(overrides, "analysis")
    mode_settings = {
        "method": raw_analysis.get("method", DEFAULT_ANALYSIS_METHOD),
        "projection": raw_analysis.get(
            "projection", ANALYSIS_DEFAULTS["patch_pca"]["projection"]
        ),
        "foreground_separation": raw_analysis.get(
            "foreground_separation",
            PATCH_PCA_IMAGE_FIT_DEFAULTS["foreground_separation"],
        ),
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


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, field_name)


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _relative_glob_pattern(value: Any, field_name: str) -> str:
    pattern = _non_empty_string(value, field_name)
    windows_pattern = PureWindowsPath(pattern)
    if (
        Path(pattern).is_absolute()
        or windows_pattern.is_absolute()
        or windows_pattern.drive
        or ".." in Path(pattern).parts
        or ".." in windows_pattern.parts
    ):
        raise ValueError(
            f"{field_name} must contain relative glob patterns without '..'."
        )
    return pattern


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
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping or null.")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings.")
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
    values = _list(value, field_name, allow_empty=False)
    return tuple(_non_negative_ints(values, field_name))


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


def _choice(value: Any, field_name: str, allowed: Set[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {options}.")
    return value


def _optional_choice(value: Any, field_name: str, allowed: Set[str]) -> str | None:
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
        try:
            number = finite_number(item, field_name)
        except ValueError as error:
            raise ValueError(
                f"{field_name} must contain only finite numbers."
            ) from error
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
    try:
        minimum = finite_number(value[0], field_name)
        maximum = finite_number(value[1], field_name)
    except ValueError as error:
        raise ValueError(f"{field_name} values must be finite.") from error
    if maximum <= minimum:
        raise ValueError(f"{field_name} maximum must be greater than its minimum.")
    return (minimum, maximum)


def _analysis_method(value: Any) -> AnalysisMethod:
    return _choice(value, "analysis.method", set(ANALYSIS_KEYS))


def _head_fusion(value: Any) -> HeadFusion:
    return _choice(value, "analysis.head_fusion", HEAD_FUSION_CHOICES)


def _foreground_side(value: Any) -> Literal["high", "low"]:
    return _choice(value, "analysis.foreground_side", FOREGROUND_SIDE_CHOICES)


def _device(value: Any) -> Device:
    return _choice(value, "runtime.device", DEVICE_CHOICES)


def _unit_interval(value: Any, field_name: str) -> float:
    try:
        number = finite_number(value, field_name)
    except ValueError as error:
        raise ValueError(f"{field_name} must be between 0 and 1.") from error
    if not 0 <= number <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return number


def _foreground_threshold(value: Any) -> float | Literal["auto"]:
    if value == "auto":
        return "auto"
    return _unit_interval(value, "analysis.foreground_threshold")


def _overlay_alpha_curve(value: Any) -> OverlayAlphaCurveSpec | None:
    if value is None:
        return None
    if (
        not isinstance(value, Mapping)
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
    midpoint = _unit_interval(
        value.get("midpoint", 0.5),
        "visualization.overlay_alpha_curve.midpoint",
    )
    return OverlayAlphaCurveSpec(steepness=steepness, midpoint=midpoint)


def _cmap_black(value: Any) -> tuple[int, int, bool] | None:
    if value is None:
        return None
    required = {"threshold", "blend_width", "transparent"}
    if not isinstance(value, Mapping) or set(value) != required:
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
    try:
        number = finite_number(value, field_name)
    except ValueError as error:
        raise ValueError(
            f"{field_name} must be a finite non-negative number."
        ) from error
    if number < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number.")
    return number


def _optional_non_negative_number(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _non_negative_number(value, field_name)


def _positive_number(value: Any, field_name: str) -> float:
    try:
        number = finite_number(value, field_name)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a finite positive number.") from error
    if number <= 0:
        raise ValueError(f"{field_name} must be a finite positive number.")
    return number


def _video_sampling_rate(value: Any) -> float | Literal["auto"]:
    if value == "auto":
        return "auto"
    return _positive_number(value, "video.sampling_rate")


def _grid_format(value: Any) -> str:
    return _choice(value, "visualization.grid_format", GRID_FORMAT_CHOICES)


def _image_format(value: Any) -> str:
    return _choice(value, "output.image_format", IMAGE_FORMAT_CHOICES)


def _raw_format(value: Any) -> str:
    return _choice(value, "output.raw_format", RAW_FORMAT_CHOICES)


def config_to_dict(config: VisionLensConfig) -> dict[str, Any]:
    validate_config(config)
    resolved = {
        "input": {"files": [str(path) for path in config.input.paths]},
        "model": _config_dataclass_to_dict(config.model),
        "preprocessing": _config_dataclass_to_dict(config.preprocessing),
        "analysis": _config_dataclass_to_dict(config.analysis),
        "runtime": _config_dataclass_to_dict(config.runtime),
        "visualization": _config_dataclass_to_dict(config.visualization),
        "output": _config_dataclass_to_dict(config.output),
    }
    if config.video is not None:
        resolved["video"] = _config_dataclass_to_dict(config.video)

    applicable = applicable_setting_keys(config)
    return {
        section: {
            key: value for key, value in values.items() if key in applicable[section]
        }
        for section, values in resolved.items()
    }


def resolved_config_yaml(config: VisionLensConfig) -> str:
    return yaml.safe_dump(config_to_dict(config), sort_keys=False)


def _config_dataclass_to_dict(value: Any) -> dict[str, Any]:
    resolved = {
        field.name: _config_value(getattr(value, field.name)) for field in fields(value)
    }
    if isinstance(value, VisualizationConfig) and value.cmap_black is not None:
        threshold, blend_width, transparent = value.cmap_black
        resolved["cmap_black"] = {
            "threshold": threshold,
            "blend_width": blend_width,
            "transparent": transparent,
        }
    return resolved


def _config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _config_dataclass_to_dict(value)
    if isinstance(value, tuple | list):
        return [_config_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _config_value(item) for key, item in value.items()}
    return value
