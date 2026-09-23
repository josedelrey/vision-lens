"""Stable public API for configuration loading, validation, and serialization.

Implementation details live in three focused modules: schema definitions,
external representation conversion, and validation.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from vision_lens.config import parsing as _parsing
from vision_lens.config import validation as _validation
from vision_lens.config.schema import (
    AlphaFormat,
    AnalysisConfig,
    AnalysisMethod,
    AttentionAnalysisConfig,
    AttentionLayers,
    ColormapSpec,
    Device,
    GradCAMAnalysisConfig,
    HeadFusion,
    InputConfig,
    ModelConfig,
    NormalizationMode,
    OutputConfig,
    OverlayAlphaCurveSpec,
    PatchPCAAnalysisConfig,
    Precision,
    PreprocessingConfig,
    RolloutAnalysisConfig,
    RolloutGrid,
    RuntimeConfig,
    VideoConfig,
    VisionLensConfig,
    VisualizationConfig,
    VisualizationInterpolation,
    VisualizationOutputSize,
)
from vision_lens.errors import ConfigurationError

__all__ = [
    "AlphaFormat",
    "AnalysisConfig",
    "AnalysisMethod",
    "AttentionAnalysisConfig",
    "AttentionLayers",
    "ColormapSpec",
    "Device",
    "GradCAMAnalysisConfig",
    "HeadFusion",
    "InputConfig",
    "ModelConfig",
    "NormalizationMode",
    "OutputConfig",
    "OverlayAlphaCurveSpec",
    "PatchPCAAnalysisConfig",
    "Precision",
    "PreprocessingConfig",
    "RolloutAnalysisConfig",
    "RolloutGrid",
    "RuntimeConfig",
    "VideoConfig",
    "VisionLensConfig",
    "VisualizationConfig",
    "VisualizationInterpolation",
    "VisualizationOutputSize",
    "config_to_dict",
    "load_config",
    "parse_config",
    "resolved_config_yaml",
    "validate_config",
    "validate_precision_device_pair",
]


def load_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> VisionLensConfig:
    """Load and validate a YAML configuration."""
    return _configuration_call(_parsing.load_config, path, overrides=overrides)


def parse_config(
    raw_config: Mapping[str, Any],
    base_dir: Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> VisionLensConfig:
    """Parse and validate a configuration mapping."""
    return _configuration_call(
        _parsing.parse_config,
        raw_config,
        base_dir=base_dir,
        overrides=overrides,
    )


def validate_config(config: VisionLensConfig) -> None:
    """Validate a resolved configuration object."""
    _configuration_call(_validation.validate_config, config)


def config_to_dict(config: VisionLensConfig) -> dict[str, Any]:
    """Serialize a resolved configuration to its public mapping form."""
    return _configuration_call(_parsing.config_to_dict, config)


def resolved_config_yaml(config: VisionLensConfig) -> str:
    """Serialize a resolved configuration as YAML."""
    return _configuration_call(_parsing.resolved_config_yaml, config)


def validate_precision_device_pair(precision: str, device: str) -> None:
    """Reject statically incompatible precision and device selections."""
    _configuration_call(_validation.validate_precision_device_pair, precision, device)


def _configuration_call[**P, R](
    operation: Callable[P, R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    try:
        return operation(*args, **kwargs)
    except ConfigurationError:
        raise
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ConfigurationError(str(error)) from error
