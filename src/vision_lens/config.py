from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from vision_lens.presets import get_preset

HeadFusion = Literal["mean", "max", "none"]
Device = Literal["auto", "cpu", "cuda", "mps"]
AttentionLayers = Literal["all"] | tuple[int, ...]


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    backend: str
    name: str
    pretrained: bool = True
    options: dict[str, Any] | None = None


@dataclass(frozen=True)
class ImageConfig:
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class OutputConfig:
    directory: Path


@dataclass(frozen=True)
class AttentionConfig:
    layers: AttentionLayers
    heads: tuple[int, ...] | None = None
    head_fusion: HeadFusion = "mean"


@dataclass(frozen=True)
class RuntimeConfig:
    device: Device = "auto"
    image_size: int = 672


@dataclass(frozen=True)
class VisualizationConfig:
    overlay_alpha: float = 0.45
    cmap: str = "viridis"
    grid_format: str = "png"


@dataclass(frozen=True)
class PatchPCAConfig:
    foreground_threshold: float = 0.5
    foreground_side: Literal["high", "low"] = "high"


@dataclass(frozen=True)
class VisionLensConfig:
    task: str
    model: ModelConfig
    images: ImageConfig
    output: OutputConfig
    attention: AttentionConfig | None
    runtime: RuntimeConfig
    visualization: VisualizationConfig
    patch_pca: PatchPCAConfig | None = None
    preset: str | None = None


def load_config(
    path: str | Path,
    *,
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> VisionLensConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    if not isinstance(raw_config, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")

    if config_path.parent.name == "configs":
        base_dir = config_path.parent.parent
    else:
        base_dir = config_path.parent
    return parse_config(
        raw_config,
        base_dir=base_dir,
        preset=preset,
        overrides=overrides,
    )


def load_preset(
    name: str,
    *,
    base_dir: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> VisionLensConfig:
    return parse_config(
        {},
        base_dir=base_dir,
        preset=name,
        overrides=overrides,
    )


def parse_config(
    raw_config: dict[str, Any],
    base_dir: Path | None = None,
    *,
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> VisionLensConfig:
    base = Path.cwd() if base_dir is None else base_dir
    preset_name = _preset_name(raw_config, preset)
    resolved = get_preset(preset_name) if preset_name is not None else {}
    user_config = {key: value for key, value in raw_config.items() if key != "preset"}
    resolved = _deep_merge(resolved, user_config)
    resolved = _deep_merge(resolved, overrides or {})

    model = _section(resolved, "model")
    images = _section(resolved, "images")
    output = _section(resolved, "output")
    task = _optional_str(resolved.get("task", "vit_attention"), "task")
    attention = _optional_section(resolved, "attention")
    runtime = _section(resolved, "runtime")
    visualization = _section(resolved, "visualization")
    patch_pca = _optional_section(resolved, "patch_pca")

    return VisionLensConfig(
        task=task,
        model=ModelConfig(
            architecture=_required_str(model, "architecture", "model"),
            backend=_required_str(model, "backend", "model"),
            name=_required_str(model, "name", "model"),
            pretrained=_bool(model.get("pretrained", True), "model.pretrained"),
            options=_optional_mapping(model.get("options"), "model.options"),
        ),
        images=ImageConfig(
            paths=tuple(
                _resolve_path(path, base)
                for path in _required_list(images, "paths", "images")
            ),
        ),
        output=OutputConfig(
            directory=_resolve_path(
                _required_str(output, "directory", "output"),
                base,
            ),
        ),
        attention=_parse_attention(attention, task),
        runtime=RuntimeConfig(
            device=_device(runtime.get("device", "auto")),
            image_size=_positive_int(
                runtime.get("image_size", 672),
                "runtime.image_size",
            ),
        ),
        visualization=VisualizationConfig(
            overlay_alpha=_alpha(visualization.get("overlay_alpha", 0.45)),
            cmap=_optional_str(
                visualization.get("cmap", "viridis"),
                "visualization.cmap",
            ),
            grid_format=_grid_format(visualization.get("grid_format", "png")),
        ),
        patch_pca=_parse_patch_pca(patch_pca, task),
        preset=preset_name,
    )


def _preset_name(raw_config: dict[str, Any], selected: str | None) -> str | None:
    configured = raw_config.get("preset")
    if configured is not None and (
        not isinstance(configured, str) or not configured.strip()
    ):
        raise ValueError("preset must be a non-empty string.")
    if selected is not None and (not isinstance(selected, str) or not selected.strip()):
        raise ValueError("preset must be a non-empty string.")
    return selected if selected is not None else configured


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be a mapping.")
    return section


def _optional_section(config: dict[str, Any], name: str) -> dict[str, Any] | None:
    section = config.get(name)
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be a mapping.")
    return section


def _parse_attention(
    section: dict[str, Any] | None,
    task: str,
) -> AttentionConfig | None:
    if section is None:
        if task in {"vit_attention", "vit_rollout"}:
            raise ValueError(f"attention is required when task is {task}.")
        return None

    return AttentionConfig(
        layers=_attention_layers(section.get("layers")),
        heads=_optional_non_negative_ints(
            section.get("heads"),
            "attention.heads",
        ),
        head_fusion=_head_fusion(section.get("head_fusion", "mean")),
    )


def _parse_patch_pca(
    section: dict[str, Any] | None,
    task: str,
) -> PatchPCAConfig | None:
    if section is None:
        return PatchPCAConfig() if task == "patch_pca" else None

    return PatchPCAConfig(
        foreground_threshold=_alpha(
            section.get("foreground_threshold", 0.5),
            field_name="patch_pca.foreground_threshold",
        ),
        foreground_side=_foreground_side(section.get("foreground_side", "high")),
    )


def _optional_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _required_str(section: dict[str, Any], key: str, section_name: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section_name}.{key} must be a non-empty string.")
    return value


def _required_list(section: dict[str, Any], key: str, section_name: str) -> list[Any]:
    value = section.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{section_name}.{key} must be a non-empty list.")
    return value


def _attention_layers(value: Any) -> AttentionLayers:
    if value == "all":
        return "all"
    if not isinstance(value, list) or not value:
        raise ValueError("attention.layers must be `all` or a non-empty list.")
    return tuple(_non_negative_ints(value, "attention.layers"))


def _optional_mapping(value: Any, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping or null.")
    return dict(value)


def _resolve_path(path: Any, base_dir: Path) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Paths must be non-empty strings.")

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved


def _non_negative_ints(values: list[Any], field_name: str) -> list[int]:
    parsed = [_non_negative_int(value, field_name) for value in values]
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return parsed


def _optional_non_negative_ints(value: Any, field_name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list or null.")
    return tuple(_non_negative_ints(value, field_name))


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} values must be non-negative integers.")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be true or false.")
    return value


def _head_fusion(value: Any) -> HeadFusion:
    allowed = {"mean", "max", "none"}
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"attention.head_fusion must be one of: {options}.")
    return value


def _foreground_side(value: Any) -> Literal["high", "low"]:
    allowed = {"high", "low"}
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"patch_pca.foreground_side must be one of: {options}.")
    return value


def _device(value: Any) -> Device:
    allowed = {"auto", "cpu", "cuda", "mps"}
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"runtime.device must be one of: {options}.")
    return value


def _alpha(
    value: Any,
    field_name: str = "visualization.overlay_alpha",
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return float(value)


def _grid_format(value: Any) -> str:
    allowed = {"pdf", "png", "svg"}
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"visualization.grid_format must be one of: {options}.")
    return value
