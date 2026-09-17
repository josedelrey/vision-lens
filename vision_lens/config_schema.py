from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Any

from vision_lens.config_types import (
    AttentionAnalysisConfig,
    GradCAMAnalysisConfig,
    ModelConfig,
    OutputConfig,
    PatchPCAAnalysisConfig,
    PreprocessingConfig,
    RolloutAnalysisConfig,
    RuntimeConfig,
    VideoConfig,
    VisualizationConfig,
)

DEFAULT_INPUT_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.webp")
DEFAULT_ANALYSIS_METHOD = "attention"
PATCH_PCA_IMAGE_FIT_DEFAULTS = {
    "foreground_separation": True,
    "foreground_threshold": 0.5,
    "foreground_side": "high",
    "rgb_fit_scope": "foreground",
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

SECTION_CONFIG_TYPES = {
    "model": ModelConfig,
    "preprocessing": PreprocessingConfig,
    "runtime": RuntimeConfig,
    "visualization": VisualizationConfig,
    "output": OutputConfig,
    "video": VideoConfig,
}
SECTION_KEYS = {
    "input": {"files", "folders", "patterns", "recursive", "limit"},
    **{
        section: {field.name for field in fields(config_type)}
        for section, config_type in SECTION_CONFIG_TYPES.items()
    },
}
TOP_LEVEL_KEYS = set(SECTION_KEYS) | {"analysis"}
VIDEO_OUTPUT_KEYS = SECTION_KEYS["output"] - {"grids", "image_format"}
GRID_VISUALIZATION_KEYS = {
    "tile_size",
    "columns",
    "items_per_grid",
    "spacing",
    "padding",
    "labels",
    "background",
    "dpi",
    "grid_format",
}
ANALYSIS_CONFIG_TYPES = {
    "attention": AttentionAnalysisConfig,
    "rollout": RolloutAnalysisConfig,
    "gradcam": GradCAMAnalysisConfig,
    "patch_pca": PatchPCAAnalysisConfig,
}
ANALYSIS_KEYS = {
    method: {field.name for field in fields(config_type)}
    for method, config_type in ANALYSIS_CONFIG_TYPES.items()
}


def _dataclass_defaults(config_type: type[Any]) -> dict[str, Any]:
    defaults = {}
    for field in fields(config_type):
        if field.default is not MISSING:
            defaults[field.name] = field.default
        elif field.default_factory is not MISSING:
            defaults[field.name] = field.default_factory()
    return defaults


SECTION_DEFAULTS = {
    section: _dataclass_defaults(config_type)
    for section, config_type in SECTION_CONFIG_TYPES.items()
}
ANALYSIS_DEFAULTS = {
    method: _dataclass_defaults(config_type)
    for method, config_type in ANALYSIS_CONFIG_TYPES.items()
}


def config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be a mapping.")
    return section
