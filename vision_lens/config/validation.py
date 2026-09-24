"""Validate resolved configuration values and workflow applicability."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

from vision_lens.config.media import partition_media_paths, split_media_configs
from vision_lens.config.schema import (
    ALPHA_FORMAT_CHOICES,
    ANALYSIS_CONFIG_TYPES,
    ANALYSIS_DEFAULTS,
    ANALYSIS_KEYS,
    ANYUP_INTERPOLATIONS,
    CROP_CHOICES,
    DEVICE_CHOICES,
    FOREGROUND_SIDE_CHOICES,
    GRID_FORMAT_CHOICES,
    GRID_VISUALIZATION_KEYS,
    HEAD_FUSION_CHOICES,
    IMAGE_FORMAT_CHOICES,
    KNOWN_FIXED_IMAGE_SIZES,
    KNOWN_VIT_DEPTHS,
    KNOWN_VIT_HEADS,
    NORMALIZATION_CHOICES,
    OVERWRITE_CHOICES,
    PAD_CHOICES,
    PRECISION_CHOICES,
    PREPROCESSING_INTERPOLATION_CHOICES,
    PROJECTION_CHOICES,
    RAW_FORMAT_CHOICES,
    RESIZE_CHOICES,
    RGB_FIT_SCOPE_CHOICES,
    ROLLOUT_GRID_CHOICES,
    SECTION_DEFAULTS,
    SECTION_KEYS,
    SOFT_ANYUP_INTERPOLATIONS,
    VISUALIZATION_INTERPOLATION_CHOICES,
    AttentionAnalysisConfig,
    GradCAMAnalysisConfig,
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
    config_section,
)
from vision_lens.output.artifacts import (
    validate_artifact_paths,
    validate_branch_artifact_paths,
)


def validate_config(
    config: VisionLensConfig,
    *,
    _branch_layout: bool = False,
) -> None:
    _validate_resolved_values(config)
    image_paths, video_paths = partition_media_paths(config.input.paths)
    if video_paths and config.video is None:
        raise ValueError("Video inputs require resolved video settings.")
    if not video_paths and config.video is not None:
        raise ValueError("Video settings require at least one video input.")

    applicable = applicable_setting_keys(config)
    validate_canonical_settings(config, applicable)
    if image_paths and video_paths:
        validate_precision_device_pair(config.runtime.precision, config.runtime.device)
        image_config, video_config = split_media_configs(config)
        assert image_config is not None and video_config is not None
        validate_config(image_config, _branch_layout=True)
        validate_config(video_config, _branch_layout=True)
        validate_artifact_paths(config)
        return

    if _branch_layout:
        validate_branch_artifact_paths(config)
    else:
        validate_artifact_paths(config)
    validate_precision_device_pair(config.runtime.precision, config.runtime.device)
    _validate_video_workflow(config)
    _validate_model_workflow(config)
    _validate_preprocessing_workflow(config)
    _validate_visualization_workflow(config, applicable)
    _validate_output_workflow(config)

    if "cmap" in applicable["visualization"]:
        _validate_colormap(config.visualization.cmap)
    if (
        "background" in applicable["visualization"]
        and config.visualization.background is not None
    ):
        _validate_color(config.visualization.background, "visualization.background")


def _validate_video_workflow(config: VisionLensConfig) -> None:
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


def _validate_model_workflow(config: VisionLensConfig) -> None:
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

    if isinstance(config.analysis, AttentionAnalysisConfig | RolloutAnalysisConfig):
        _validate_known_attention_constraints(config)


def _validate_preprocessing_workflow(config: VisionLensConfig) -> None:
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


def _validate_visualization_workflow(
    config: VisionLensConfig,
    applicable: dict[str, set[str]],
) -> None:
    if (
        "interpolation" in applicable["visualization"]
        and config.visualization.anyup_query_chunk_size is not None
        and config.visualization.interpolation not in ANYUP_INTERPOLATIONS
    ):
        raise ValueError(
            "visualization.anyup_query_chunk_size requires "
            "visualization.interpolation to be 'anyup', 'anyup_mask', or "
            "'anyup_soft', or 'anyup_soft_mask'."
        )
    if (
        "interpolation" in applicable["visualization"]
        and config.visualization.interpolation in SOFT_ANYUP_INTERPOLATIONS
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


def _validate_output_workflow(config: VisionLensConfig) -> None:
    save_projection = (
        config.analysis.save_projection
        if isinstance(config.analysis, PatchPCAAnalysisConfig)
        else None
    )
    if not any(
        (
            config.output.heatmaps,
            config.output.overlays,
            config.output.transparent_overlays,
            config.output.grids,
            config.output.raw_arrays,
            save_projection is not None,
        )
    ):
        raise ValueError("At least one output type must be enabled.")

    rendered = _renders_output(config)
    if config.visualization.output_size == "match" and not rendered:
        raise ValueError(
            "visualization.output_size='match' requires a rendered heatmap, "
            "overlay, or grid output."
        )

    if isinstance(config.analysis, PatchPCAAnalysisConfig):
        if config.output.overlays:
            raise ValueError("output.overlays is not supported for patch PCA.")
        if config.output.transparent_overlays:
            raise ValueError(
                "output.transparent_overlays is not supported for patch PCA."
            )
        if config.analysis.projection == "load":
            if config.analysis.projection_path is None:
                raise ValueError(
                    "analysis.projection_path is required when projection='load'."
                )
            if not config.analysis.projection_path.is_file():
                raise ValueError(
                    "PCA projection file does not exist: "
                    f"{config.analysis.projection_path}."
                )


def _validate_resolved_values(config: VisionLensConfig) -> None:
    if not isinstance(config, VisionLensConfig):
        raise ValueError("config must be a VisionLensConfig instance.")
    _require_instance(config.input, InputConfig, "input")
    _require_instance(config.model, ModelConfig, "model")
    _require_instance(config.preprocessing, PreprocessingConfig, "preprocessing")
    _require_instance(config.runtime, RuntimeConfig, "runtime")
    _require_instance(config.visualization, VisualizationConfig, "visualization")
    _require_instance(config.output, OutputConfig, "output")
    if config.video is not None:
        _require_instance(config.video, VideoConfig, "video")

    analysis_method = getattr(config.analysis, "method", None)
    expected_analysis_type = ANALYSIS_CONFIG_TYPES.get(analysis_method)
    if expected_analysis_type is None or not isinstance(
        config.analysis, expected_analysis_type
    ):
        expected = (
            "a known method-specific analysis config"
            if expected_analysis_type is None
            else expected_analysis_type.__name__
        )
        raise ValueError(
            f"analysis must be {expected}; got {type(config.analysis).__name__}."
        )

    _validate_input_values(config.input)
    _validate_model_values(config.model)
    _validate_preprocessing_values(config.preprocessing)
    _validate_analysis_values(config)
    _validate_runtime_values(config.runtime)
    _validate_visualization_values(config.visualization)
    _validate_output_values(config.output)
    if config.video is not None:
        _validate_video_values(config.video)


def _validate_input_values(input_config: InputConfig) -> None:
    if not isinstance(input_config.paths, tuple) or not input_config.paths:
        raise ValueError("input.paths must contain at least one file.")
    for path in input_config.paths:
        _require_path(path, "input.paths")
        if not path.is_file():
            raise ValueError(f"Input file does not exist: {path}.")
    image_paths, video_paths = partition_media_paths(input_config.paths)
    if len(image_paths) + len(video_paths) != len(input_config.paths):
        raise ValueError("input.paths contains an unsupported media type.")


def _validate_model_values(model: ModelConfig) -> None:
    _require_string(model.architecture, "model.architecture")
    _require_string(model.backend, "model.backend")
    _require_string(model.name, "model.name")
    _require_bool(model.pretrained, "model.pretrained")
    if model.options is not None:
        if not isinstance(model.options, dict) or not all(
            isinstance(key, str) for key in model.options
        ):
            raise ValueError("model.options must be a mapping with string keys.")


def _validate_preprocessing_values(preprocessing: PreprocessingConfig) -> None:
    _require_int(preprocessing.image_size, "preprocessing.image_size", minimum=1)
    _require_choice(
        preprocessing.resize,
        "preprocessing.resize",
        RESIZE_CHOICES,
    )
    _require_choice(preprocessing.crop, "preprocessing.crop", CROP_CHOICES)
    _require_choice(preprocessing.pad, "preprocessing.pad", PAD_CHOICES)
    if preprocessing.interpolation is not None:
        _require_choice(
            preprocessing.interpolation,
            "preprocessing.interpolation",
            PREPROCESSING_INTERPOLATION_CHOICES,
        )
    _require_bool(preprocessing.normalize, "preprocessing.normalize")
    _require_triplet(preprocessing.mean, "preprocessing.mean")
    _require_triplet(preprocessing.std, "preprocessing.std", positive=True)


def _validate_analysis_values(config: VisionLensConfig) -> None:
    analysis = config.analysis
    if isinstance(analysis, AttentionAnalysisConfig | RolloutAnalysisConfig):
        _validate_attention_analysis_values(analysis)
    elif isinstance(analysis, GradCAMAnalysisConfig):
        _validate_gradcam_analysis_values(analysis)
    else:
        assert isinstance(analysis, PatchPCAAnalysisConfig)
        image_paths, video_paths = partition_media_paths(config.input.paths)
        _validate_patch_pca_analysis_values(
            analysis,
            is_video=bool(video_paths and not image_paths),
        )


def _validate_attention_analysis_values(
    analysis: AttentionAnalysisConfig | RolloutAnalysisConfig,
) -> None:
    _require_indices_or_all(analysis.layers, "analysis.layers")
    _require_optional_indices(analysis.heads, "analysis.heads")
    _require_choice(analysis.head_fusion, "analysis.head_fusion", HEAD_FUSION_CHOICES)


def _validate_gradcam_analysis_values(analysis: GradCAMAnalysisConfig) -> None:
    if analysis.target_layer is not None:
        _require_string(analysis.target_layer, "analysis.target_layer")
    if analysis.target_class is not None:
        _require_int(analysis.target_class, "analysis.target_class", minimum=0)


def _validate_patch_pca_analysis_values(
    analysis: PatchPCAAnalysisConfig,
    *,
    is_video: bool,
) -> None:
    if analysis.foreground_separation is not None:
        _require_bool(
            analysis.foreground_separation,
            "analysis.foreground_separation",
        )
    if (
        analysis.foreground_threshold is not None
        and analysis.foreground_threshold != "auto"
    ):
        _require_number(
            analysis.foreground_threshold,
            "analysis.foreground_threshold",
            minimum=0,
            maximum=1,
        )
    if analysis.foreground_side is not None:
        _require_choice(
            analysis.foreground_side,
            "analysis.foreground_side",
            FOREGROUND_SIDE_CHOICES,
        )
    if analysis.rgb_fit_scope is not None:
        _require_choice(
            analysis.rgb_fit_scope,
            "analysis.rgb_fit_scope",
            RGB_FIT_SCOPE_CHOICES,
        )
    _require_choice(analysis.projection, "analysis.projection", PROJECTION_CHOICES)
    if analysis.projection_path is not None:
        _require_path(analysis.projection_path, "analysis.projection_path")
    if analysis.save_projection is not None:
        _require_path(analysis.save_projection, "analysis.save_projection")
    if (
        not is_video
        and analysis.projection == "fit"
        and analysis.foreground_separation is None
    ):
        raise ValueError("Image patch PCA requires analysis.foreground_separation.")
    if analysis.foreground_separation is True and any(
        value is None
        for value in (
            analysis.foreground_threshold,
            analysis.foreground_side,
            analysis.rgb_fit_scope,
        )
    ):
        raise ValueError(
            "Foreground-separated patch PCA requires threshold, side, and "
            "RGB fit scope settings."
        )


def _validate_runtime_values(runtime: RuntimeConfig) -> None:
    _require_choice(runtime.device, "runtime.device", DEVICE_CHOICES)
    _require_int(runtime.batch_size, "runtime.batch_size", minimum=1)
    _require_int(runtime.workers, "runtime.workers", minimum=0)
    _require_choice(
        runtime.precision,
        "runtime.precision",
        PRECISION_CHOICES,
    )
    if runtime.seed is not None:
        _require_int(runtime.seed, "runtime.seed", minimum=0, maximum=2**32 - 1)


def _validate_visualization_values(visualization: VisualizationConfig) -> None:
    _validate_visualization_layout_values(visualization)
    _validate_visualization_rendering_values(visualization)


def _validate_visualization_layout_values(
    visualization: VisualizationConfig,
) -> None:
    _require_optional_size(visualization.tile_size, "visualization.tile_size")
    for key in ("columns", "items_per_grid", "dpi", "anyup_query_chunk_size"):
        value = getattr(visualization, key)
        if value is not None:
            _require_int(value, f"visualization.{key}", minimum=1)
    for key in ("spacing", "padding"):
        value = getattr(visualization, key)
        if value is not None:
            _require_int(value, f"visualization.{key}", minimum=0)
    if visualization.labels is not None:
        _require_bool(visualization.labels, "visualization.labels")
    if visualization.background is not None:
        _require_string(visualization.background, "visualization.background")
    if visualization.output_size != "match":
        _require_optional_size(visualization.output_size, "visualization.output_size")


def _validate_visualization_rendering_values(
    visualization: VisualizationConfig,
) -> None:
    _require_choice(
        visualization.interpolation,
        "visualization.interpolation",
        VISUALIZATION_INTERPOLATION_CHOICES,
    )
    _require_number(
        visualization.overlay_alpha,
        "visualization.overlay_alpha",
        minimum=0,
        maximum=1,
    )
    _require_string(visualization.cmap, "visualization.cmap")
    if visualization.cmap_black is not None:
        if (
            not isinstance(visualization.cmap_black, tuple)
            or len(visualization.cmap_black) != 3
        ):
            raise ValueError("visualization.cmap_black must be a three-item tuple.")
        threshold, blend_width, transparent = visualization.cmap_black
        _require_int(
            threshold, "visualization.cmap_black.threshold", minimum=0, maximum=254
        )
        _require_int(
            blend_width,
            "visualization.cmap_black.blend_width",
            minimum=1,
            maximum=255 - threshold,
        )
        _require_bool(transparent, "visualization.cmap_black.transparent")
    _require_choice(
        visualization.grid_format,
        "visualization.grid_format",
        GRID_FORMAT_CHOICES,
    )
    _require_choice(
        visualization.rollout_grid,
        "visualization.rollout_grid",
        ROLLOUT_GRID_CHOICES,
    )
    _require_choice(
        visualization.normalization,
        "visualization.normalization",
        NORMALIZATION_CHOICES,
    )
    _require_range(
        visualization.normalization_range,
        "visualization.normalization_range",
    )
    if visualization.overlay_alpha_curve is not None:
        _require_instance(
            visualization.overlay_alpha_curve,
            OverlayAlphaCurveSpec,
            "visualization.overlay_alpha_curve",
        )
        _require_number(
            visualization.overlay_alpha_curve.steepness,
            "visualization.overlay_alpha_curve.steepness",
            minimum=0,
            minimum_inclusive=False,
        )
        _require_number(
            visualization.overlay_alpha_curve.midpoint,
            "visualization.overlay_alpha_curve.midpoint",
            minimum=0,
            maximum=1,
        )


def _validate_output_values(output: OutputConfig) -> None:
    _require_path(output.directory, "output.directory")
    for key in (
        "heatmaps",
        "overlays",
        "transparent_overlays",
        "grids",
        "raw_arrays",
    ):
        _require_bool(getattr(output, key), f"output.{key}")
    _require_choice(
        output.image_format,
        "output.image_format",
        IMAGE_FORMAT_CHOICES,
    )
    _require_choice(output.raw_format, "output.raw_format", RAW_FORMAT_CHOICES)
    _require_choice(output.overwrite, "output.overwrite", OVERWRITE_CHOICES)


def _validate_video_values(video: VideoConfig) -> None:
    _require_number(video.start_time, "video.start_time", minimum=0)
    if video.end_time is not None:
        _require_number(video.end_time, "video.end_time", minimum=0)
    if video.sampling_rate != "auto":
        _require_number(
            video.sampling_rate,
            "video.sampling_rate",
            minimum=0,
            minimum_inclusive=False,
        )
    if video.frame_limit is not None:
        _require_int(video.frame_limit, "video.frame_limit", minimum=1)
    _require_int(video.pca_fit_frames, "video.pca_fit_frames", minimum=1)
    _require_number(
        video.temporal_smoothing,
        "video.temporal_smoothing",
        minimum=0,
        maximum=1,
    )
    _require_string(video.codec, "video.codec")
    _require_choice(
        video.alpha_format,
        "video.alpha_format",
        ALPHA_FORMAT_CHOICES,
    )


def _require_instance(value: Any, expected: type[Any], field_name: str) -> None:
    if not isinstance(value, expected):
        raise ValueError(f"{field_name} must be {expected.__name__}.")


def _require_path(value: Any, field_name: str) -> None:
    if not isinstance(value, Path):
        raise ValueError(f"{field_name} must be a Path.")


def _require_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _require_bool(value: Any, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")


def _require_int(
    value: Any,
    field_name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}.")


def finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a finite number.")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a finite number.") from error
    if not isfinite(number):
        raise ValueError(f"{field_name} must be a finite number.")
    return number


def _require_number(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> None:
    number = finite_number(value, field_name)
    if minimum is not None and (
        number < minimum if minimum_inclusive else number <= minimum
    ):
        qualifier = "at least" if minimum_inclusive else "greater than"
        raise ValueError(f"{field_name} must be {qualifier} {minimum}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}.")


def _require_choice(value: Any, field_name: str, choices: Any) -> None:
    if not isinstance(value, str) or value not in choices:
        options = ", ".join(sorted(choices))
        raise ValueError(f"{field_name} must be one of: {options}.")


def _require_triplet(
    value: Any,
    field_name: str,
    *,
    positive: bool = False,
) -> None:
    if value is None:
        return
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{field_name} must be a three-item tuple or None.")
    for item in value:
        _require_number(
            item,
            field_name,
            minimum=0 if positive else None,
            minimum_inclusive=not positive,
        )


def _require_indices_or_all(value: Any, field_name: str) -> None:
    if value == "all":
        return
    _require_optional_indices(value, field_name, allow_none=False)


def _require_optional_indices(
    value: Any,
    field_name: str,
    *,
    allow_none: bool = True,
) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field_name} must be a non-empty tuple of indices.")
    for item in value:
        _require_int(item, field_name, minimum=0)
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} must not contain duplicates.")


def _require_optional_size(value: Any, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-item tuple or None.")
    for item in value:
        _require_int(item, field_name, minimum=1)


def _require_range(value: Any, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-item tuple or None.")
    for item in value:
        _require_number(item, field_name)
    if value[1] <= value[0]:
        raise ValueError(f"{field_name} maximum must exceed its minimum.")


def validate_precision_device_pair(precision: str, device: str) -> None:
    if precision == "float16" and device == "cpu":
        raise ValueError(
            "runtime.precision='float16' is not supported on CPU; use "
            "float32 or bfloat16."
        )
    if precision == "bfloat16" and device == "mps":
        raise ValueError(
            "runtime.precision='bfloat16' is not supported on MPS; use "
            "float32 or float16."
        )


def applicable_setting_keys(config: VisionLensConfig) -> dict[str, set[str]]:
    image_paths, video_paths = partition_media_paths(config.input.paths)
    if image_paths and video_paths:
        image_config, video_config = split_media_configs(config)
        assert image_config is not None and video_config is not None
        image_applicable = applicable_setting_keys(image_config)
        video_applicable = applicable_setting_keys(video_config)
        return {
            section: image_applicable[section] | video_applicable[section]
            for section in image_applicable
        }

    is_video = bool(video_paths)
    runtime = {"batch_size", "device", "precision", "seed"}
    if not is_video:
        runtime.add("workers")
    return {
        "input": set(SECTION_KEYS["input"]),
        "model": set(SECTION_KEYS["model"]),
        "preprocessing": _applicable_preprocessing_keys(config, is_video=is_video),
        "analysis": _applicable_analysis_keys(config, is_video=is_video),
        "runtime": runtime,
        "visualization": _applicable_visualization_keys(config, is_video=is_video),
        "output": _applicable_output_keys(config, is_video=is_video),
        "video": _applicable_video_keys(config, is_video=is_video),
    }


def _applicable_preprocessing_keys(
    config: VisionLensConfig,
    *,
    is_video: bool,
) -> set[str]:
    preprocessing = {"image_size", "interpolation", "normalize"}
    if not is_video:
        preprocessing.update({"resize", "crop", "pad"})
    if config.preprocessing.normalize:
        preprocessing.update({"mean", "std"})
    return preprocessing


def _applicable_analysis_keys(
    config: VisionLensConfig,
    *,
    is_video: bool,
) -> set[str]:
    method = config.analysis.method
    analysis = set(ANALYSIS_KEYS[method])
    if method == "rollout" and (
        is_video
        or not config.output.grids
        or config.visualization.rollout_grid != "comparison"
    ):
        analysis.difference_update({"heads", "head_fusion"})
    elif isinstance(config.analysis, PatchPCAAnalysisConfig):
        if config.analysis.projection == "load":
            analysis.intersection_update({"method", "projection", "projection_path"})
        else:
            analysis.discard("projection_path")
            if is_video:
                analysis.difference_update(
                    {
                        "foreground_separation",
                        "foreground_threshold",
                        "foreground_side",
                        "rgb_fit_scope",
                    }
                )
            elif not config.analysis.foreground_separation:
                analysis.difference_update(
                    {"foreground_threshold", "foreground_side", "rgb_fit_scope"}
                )
    return analysis


def _applicable_visualization_keys(
    config: VisionLensConfig,
    *,
    is_video: bool,
) -> set[str]:
    method = config.analysis.method
    rendered = _renders_output(config)
    spatial_output = rendered or (config.output.raw_arrays and method != "patch_pca")
    visualization: set[str] = set()
    if spatial_output:
        visualization.update({"output_size", "interpolation"})
        if config.visualization.interpolation in ANYUP_INTERPOLATIONS:
            visualization.add("anyup_query_chunk_size")
    if method != "patch_pca" and rendered:
        visualization.update({"normalization", "cmap", "cmap_black"})
        if config.visualization.normalization == "fixed":
            visualization.add("normalization_range")
        if (
            config.output.overlays
            or config.output.transparent_overlays
            or config.output.grids
        ):
            visualization.update({"overlay_alpha", "overlay_alpha_curve"})
    if not is_video and config.output.grids:
        visualization.update(GRID_VISUALIZATION_KEYS)
        if method == "rollout":
            visualization.add("rollout_grid")
        if method == "patch_pca":
            visualization.discard("labels")
    return visualization


def _applicable_output_keys(
    config: VisionLensConfig,
    *,
    is_video: bool,
) -> set[str]:
    method = config.analysis.method
    output = {"directory", "heatmaps", "raw_arrays", "overwrite"}
    if method != "patch_pca":
        output.update({"overlays", "transparent_overlays"})
    if not is_video:
        output.add("grids")
    if not is_video and (config.output.heatmaps or config.output.overlays):
        output.add("image_format")
    if config.output.raw_arrays:
        output.add("raw_format")
    return output


def _applicable_video_keys(
    config: VisionLensConfig,
    *,
    is_video: bool,
) -> set[str]:
    method = config.analysis.method
    video: set[str] = set()
    if is_video:
        video.update({"start_time", "end_time", "sampling_rate", "frame_limit"})
        if (
            isinstance(config.analysis, PatchPCAAnalysisConfig)
            and config.analysis.projection == "fit"
        ):
            video.add("pca_fit_frames")
        if method != "patch_pca" or config.output.heatmaps:
            video.add("temporal_smoothing")
        if config.output.heatmaps or config.output.overlays:
            video.add("codec")
        if config.output.transparent_overlays:
            video.add("alpha_format")
    return video


def validate_canonical_settings(
    config: VisionLensConfig,
    applicable: dict[str, set[str]] | None = None,
) -> None:
    applicable = applicable or applicable_setting_keys(config)
    sections = {
        "model": config.model,
        "preprocessing": config.preprocessing,
        "analysis": config.analysis,
        "runtime": config.runtime,
        "visualization": config.visualization,
        "output": config.output,
    }
    if config.video is not None:
        sections["video"] = config.video

    defaults = {
        **SECTION_DEFAULTS,
        "analysis": ANALYSIS_DEFAULTS[config.analysis.method],
    }
    output_defaults = dict(defaults["output"])
    if config.video is not None:
        output_defaults["grids"] = False
    if isinstance(config.analysis, PatchPCAAnalysisConfig):
        output_defaults["overlays"] = False
    defaults["output"] = output_defaults

    for section_name, section in sections.items():
        for key, expected in defaults[section_name].items():
            if key in applicable[section_name]:
                continue
            actual = getattr(section, key)
            if actual != expected:
                raise ValueError(
                    f"Resolved setting {section_name}.{key} is not applicable to "
                    f"this workflow; expected its canonical value {expected!r}, "
                    f"got {actual!r}."
                )


def _renders_output(config: VisionLensConfig) -> bool:
    return (
        config.output.heatmaps
        or config.output.overlays
        or config.output.transparent_overlays
        or config.output.grids
    )


def validate_applicable_settings(
    raw_config: dict[str, Any],
    config: VisionLensConfig,
) -> None:
    input_section = config_section(raw_config, "input")
    if not input_section.get("folders"):
        _reject_present_keys(
            "input",
            input_section,
            {"patterns", "recursive"},
            "input.patterns and input.recursive require input.folders",
        )

    applicable = applicable_setting_keys(config)
    analysis_section = config_section(raw_config, "analysis")
    _reject_present_keys(
        "analysis",
        analysis_section,
        set(analysis_section) - applicable["analysis"],
        "the settings do not affect this workflow",
    )
    for section_name in SECTION_KEYS:
        section = config_section(raw_config, section_name)
        present_but_unused = set(section) - applicable[section_name]
        if present_but_unused:
            _reject_present_keys(
                section_name,
                section,
                present_but_unused,
                "the settings do not affect this workflow",
            )


def _validate_colormap(name: str) -> None:
    from matplotlib import colormaps

    if name not in colormaps:
        raise ValueError(
            f"visualization.cmap is not a known Matplotlib colormap: {name!r}."
        )


def _validate_color(value: str, field_name: str) -> None:
    from matplotlib.colors import is_color_like

    if not is_color_like(value):
        raise ValueError(f"{field_name} is not a valid Matplotlib color: {value!r}.")


def _validate_known_attention_constraints(config: VisionLensConfig) -> None:
    analysis = config.analysis
    assert isinstance(analysis, AttentionAnalysisConfig | RolloutAnalysisConfig)
    depth = KNOWN_VIT_DEPTHS.get(config.model.name)
    if depth is not None and analysis.layers != "all":
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
