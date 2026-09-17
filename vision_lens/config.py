"""Public configuration API.

The implementation is split by responsibility so schema types, parsing,
validation, and serialization can evolve independently without duplicating the
public import surface.
"""

from vision_lens.config_parsing import load_config, parse_config
from vision_lens.config_serialization import config_to_dict, resolved_config_yaml
from vision_lens.config_types import (
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
from vision_lens.config_validation import (
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
