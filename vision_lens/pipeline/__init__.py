"""Dispatch image and video configurations to their respective pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from vision_lens.config import VisionLensConfig, load_config, validate_config
from vision_lens.config.media import split_media_configs
from vision_lens.errors import ConfigurationError, PipelineError
from vision_lens.output.artifacts import (
    check_artifact_overwrite,
    prepare_output_directory,
    run_manifest_path,
    unique_input_labels,
)
from vision_lens.output.manifest import write_run_manifest
from vision_lens.pipeline import image as _image_pipeline
from vision_lens.pipeline import video as _video_pipeline
from vision_lens.pipeline.progress import progress_output, status

_ImagePipelineResult = (
    _image_pipeline.PipelineResult
    | _image_pipeline.GradCamPipelineResult
    | _image_pipeline.PatchPCAPipelineResult
)
_VideoPipelineResult = (
    _video_pipeline.VideoPipelineResult | _video_pipeline.VideoBatchPipelineResult
)
_SinglePipelineResult = _ImagePipelineResult | _VideoPipelineResult


@dataclass(frozen=True)
class MixedPipelineResult:
    """Combined outputs from the image and video branches of a mixed run."""

    config: VisionLensConfig
    image: _ImagePipelineResult
    video: _VideoPipelineResult
    output_paths: tuple[Path, ...]


_PipelineResult = _SinglePipelineResult | MixedPipelineResult

__all__ = ["run_pipeline", "run_pipeline_from_config"]


def run_pipeline(
    config_path: str | Path,
    *,
    show_progress: bool = True,
) -> _PipelineResult:
    config = load_config(config_path)
    return run_pipeline_from_config(config, show_progress=show_progress)


def run_pipeline_from_config(
    config: VisionLensConfig,
    *,
    show_progress: bool = True,
) -> _PipelineResult:
    if not isinstance(show_progress, bool):
        raise PipelineError("show_progress must be a boolean.")
    try:
        with progress_output(show_progress):
            return _dispatch_pipeline(config)
    except ConfigurationError:
        raise
    except PipelineError:
        raise
    except Exception as error:
        raise PipelineError(str(error)) from error


def _dispatch_pipeline(config: VisionLensConfig) -> _PipelineResult:
    started_at = datetime.now(UTC)
    image_config, video_config = split_media_configs(config)
    _preflight_pipeline(config, video_config)
    prepare_output_directory(config)
    if image_config is not None and video_config is not None:
        status(
            f"Detected {len(image_config.input.paths)} image(s) and "
            f"{len(video_config.input.paths)} video(s)"
        )
        image_result = _dispatch_image_pipeline(image_config)
        video_result = _video_pipeline.run_video_from_config(video_config)
        result: _PipelineResult = MixedPipelineResult(
            config=config,
            image=image_result,
            video=video_result,
            output_paths=image_result.output_paths + video_result.output_paths,
        )
    elif video_config is not None:
        result = _video_pipeline.run_video_from_config(video_config)
    elif image_config is not None:
        result = _dispatch_image_pipeline(image_config)
    else:
        raise ValueError("Configuration does not contain any supported media inputs.")
    _write_root_manifest(config, result, started_at=started_at)
    return result


def _preflight_pipeline(
    config: VisionLensConfig,
    video_config: VisionLensConfig | None,
) -> None:
    validate_config(config)
    check_artifact_overwrite(config)
    if video_config is not None:
        _video_pipeline.preflight_video_dependencies(video_config)


def _write_root_manifest(
    config: VisionLensConfig,
    result: _PipelineResult,
    *,
    started_at: datetime,
) -> None:
    if isinstance(result, MixedPipelineResult):
        loaded_model = result.image.loaded_model
        manifests = (
            run_manifest_path(result.image.config.output.directory),
            *_video_manifest_paths(result.video),
        )
        media = "mixed"
    elif isinstance(result, _video_pipeline.VideoBatchPipelineResult):
        loaded_model = result.videos[0].loaded_model
        manifests = _video_manifest_paths(result)
        media = "video"
    elif isinstance(result, _video_pipeline.VideoPipelineResult):
        loaded_model = result.loaded_model
        manifests = _video_manifest_paths(result)
        media = "video"
    else:
        loaded_model = result.loaded_model
        manifests = (run_manifest_path(result.config.output.directory),)
        media = "image"
    write_run_manifest(
        config,
        loaded_model,
        unique_input_labels(config.input.paths),
        result.output_paths,
        started_at=started_at,
        run_details={
            "media": media,
            "manifests": [str(path.resolve()) for path in manifests],
        },
    )


def _video_manifest_paths(
    result: _VideoPipelineResult,
) -> tuple[Path, ...]:
    if isinstance(result, _video_pipeline.VideoBatchPipelineResult):
        return tuple(
            run_manifest_path(video.config.output.directory) for video in result.videos
        )
    return (run_manifest_path(result.config.output.directory),)


def _dispatch_image_pipeline(
    config: VisionLensConfig,
) -> _ImagePipelineResult:
    method = config.analysis.method
    if method == "attention":
        return _image_pipeline.run_vit_attention_from_config(config)
    if method == "rollout":
        return _image_pipeline.run_vit_rollout_comparison_from_config(config)
    if method == "gradcam":
        return _image_pipeline.run_gradcam_from_config(config)
    if method == "patch_pca":
        return _image_pipeline.run_patch_pca_from_config(config)

    raise ValueError(f"Unsupported analysis method: {method}")
