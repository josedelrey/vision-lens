"""Dispatch image and video configurations to their respective pipelines."""

from __future__ import annotations

from pathlib import Path

from vision_lens.config import VisionLensConfig, load_config
from vision_lens.pipeline.image import (
    GradCamPipelineResult,
    PatchPCAPipelineResult,
    PipelineResult,
    export_attention_outputs,
    export_gradcam_outputs,
    export_patch_pca_outputs,
    export_rollout_comparison_outputs,
    run_gradcam,
    run_gradcam_from_config,
    run_patch_pca,
    run_patch_pca_from_config,
    run_vit_attention,
    run_vit_attention_from_config,
    run_vit_rollout_comparison,
    run_vit_rollout_comparison_from_config,
)
from vision_lens.pipeline.video import (
    VideoBatchPipelineResult,
    VideoPipelineResult,
    run_video_from_config,
)

__all__ = [
    "GradCamPipelineResult",
    "PatchPCAPipelineResult",
    "PipelineResult",
    "VideoBatchPipelineResult",
    "VideoPipelineResult",
    "export_attention_outputs",
    "export_gradcam_outputs",
    "export_patch_pca_outputs",
    "export_rollout_comparison_outputs",
    "run_gradcam",
    "run_gradcam_from_config",
    "run_patch_pca",
    "run_patch_pca_from_config",
    "run_pipeline",
    "run_pipeline_from_config",
    "run_video",
    "run_video_from_config",
    "run_vit_attention",
    "run_vit_attention_from_config",
    "run_vit_rollout_comparison",
    "run_vit_rollout_comparison_from_config",
]


def run_pipeline(
    config_path: str | Path,
) -> (
    PipelineResult
    | GradCamPipelineResult
    | PatchPCAPipelineResult
    | VideoPipelineResult
    | VideoBatchPipelineResult
):
    config = load_config(config_path)
    return run_pipeline_from_config(config)


def run_pipeline_from_config(
    config: VisionLensConfig,
) -> (
    PipelineResult
    | GradCamPipelineResult
    | PatchPCAPipelineResult
    | VideoPipelineResult
    | VideoBatchPipelineResult
):
    if config.video is not None:
        return run_video_from_config(config)
    method = config.analysis.method
    if method == "attention":
        return run_vit_attention_from_config(config)
    if method == "rollout":
        return run_vit_rollout_comparison_from_config(config)
    if method == "gradcam":
        return run_gradcam_from_config(config)
    if method == "patch_pca":
        return run_patch_pca_from_config(config)

    raise ValueError(f"Unsupported analysis method: {method}")


def run_video(
    config_path: str | Path,
) -> VideoPipelineResult | VideoBatchPipelineResult:
    config = load_config(config_path)
    return run_video_from_config(config)
