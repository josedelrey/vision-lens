"""Supported public API for Vision Lens."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from vision_lens.config import (
    VisionLensConfig,
    config_to_dict,
    load_config,
    parse_config,
    resolved_config_yaml,
    validate_config,
)
from vision_lens.errors import ConfigurationError, PipelineError, VisionLensError

__all__ = [
    "ConfigurationError",
    "PipelineError",
    "RunResult",
    "VisionLensConfig",
    "VisionLensError",
    "config_to_dict",
    "load_config",
    "parse_config",
    "resolved_config_yaml",
    "run_pipeline",
    "run_pipeline_from_config",
    "validate_config",
]


@runtime_checkable
class RunResult(Protocol):
    """Fields shared by every completed pipeline result."""

    @property
    def config(self) -> VisionLensConfig: ...

    @property
    def output_paths(self) -> tuple[Path, ...]: ...


def run_pipeline(
    config_path: str | Path,
    *,
    show_progress: bool = True,
) -> RunResult:
    """Load a configuration and run its selected workflow."""
    from vision_lens.pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(config_path, show_progress=show_progress)


def run_pipeline_from_config(
    config: VisionLensConfig,
    *,
    show_progress: bool = True,
) -> RunResult:
    """Run an already parsed configuration."""
    from vision_lens.pipeline import run_pipeline_from_config as _run_pipeline

    return _run_pipeline(config, show_progress=show_progress)
