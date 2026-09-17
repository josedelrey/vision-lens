from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from vision_lens.config_types import VisionLensConfig, VisualizationConfig
from vision_lens.config_validation import (
    applicable_setting_keys,
    validate_config,
)


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
