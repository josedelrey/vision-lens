"""Stable public API for configuration loading, validation, and serialization.

Implementation details live in three focused modules: schema definitions,
external representation conversion, and validation.
"""

from vision_lens.config.parsing import (
    config_to_dict,
    load_config,
    parse_config,
    resolved_config_yaml,
)
from vision_lens.config.schema import (
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
    RuntimeConfig,
    VideoConfig,
    VisionLensConfig,
    VisualizationConfig,
    VisualizationInterpolation,
    VisualizationOutputSize,
)
from vision_lens.config.validation import (
    validate_config,
    validate_precision_device_pair,
)

__all__ = [
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
    "RuntimeConfig",
    "RolloutAnalysisConfig",
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
