"""Dispatch image and video configurations to their respective pipelines."""

from __future__ import annotations

from pathlib import Path

from vision_lens.config import VisionLensConfig, load_config
from vision_lens.errors import ConfigurationError, PipelineError
from vision_lens.pipeline import image as _image_pipeline
from vision_lens.pipeline import video as _video_pipeline
from vision_lens.pipeline.progress import progress_output

_PipelineResult = (
    _image_pipeline.PipelineResult
    | _image_pipeline.GradCamPipelineResult
    | _image_pipeline.PatchPCAPipelineResult
    | _video_pipeline.VideoPipelineResult
    | _video_pipeline.VideoBatchPipelineResult
)

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
    if config.video is not None:
        return _video_pipeline.run_video_from_config(config)
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
