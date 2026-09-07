from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from vision_lens.presets import get_preset

HeadFusion = Literal["mean", "max", "none"]
Device = Literal["auto", "cpu", "cuda", "mps"]
AttentionLayers = Literal["all"] | tuple[int, ...]
AnalysisMethod = Literal["attention", "rollout", "gradcam", "patch_pca"]

TOP_LEVEL_KEYS = {
    "preset",
    "input",
    "model",
    "preprocessing",
    "analysis",
    "runtime",
    "visualization",
    "output",
}
SECTION_KEYS = {
    "input": {"paths"},
    "model": {"architecture", "backend", "name", "pretrained", "options"},
    "preprocessing": {"image_size"},
    "runtime": {"device"},
    "visualization": {"overlay_alpha", "cmap", "grid_format"},
    "output": {"directory"},
}
ANALYSIS_KEYS = {
    "attention": {"method", "layers", "heads", "head_fusion"},
    "rollout": {"method", "layers", "heads", "head_fusion"},
    "gradcam": {"method", "target_layer"},
    "patch_pca": {"method", "foreground_threshold", "foreground_side"},
}
METHOD_TASKS = {
    "attention": "vit_attention",
    "rollout": "vit_rollout",
    "gradcam": "gradcam",
    "patch_pca": "patch_pca",
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


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    backend: str
    name: str
    pretrained: bool = True
    options: dict[str, Any] | None = None


@dataclass(frozen=True)
class InputConfig:
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
class PatchPCAConfig:
    foreground_threshold: float = 0.5
    foreground_side: Literal["high", "low"] = "high"


@dataclass(frozen=True)
class AnalysisConfig:
    method: AnalysisMethod
    layers: AttentionLayers | None = None
    heads: tuple[int, ...] | None = None
    head_fusion: HeadFusion | None = None
    target_layer: str | None = None
    foreground_threshold: float | None = None
    foreground_side: Literal["high", "low"] | None = None


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: int = 672


@dataclass(frozen=True)
class RuntimeConfig:
    device: Device = "auto"


@dataclass(frozen=True)
class VisualizationConfig:
    overlay_alpha: float = 0.45
    cmap: str = "viridis"
    grid_format: str = "png"


@dataclass(frozen=True)
class VisionLensConfig:
    input: InputConfig
    model: ModelConfig
    preprocessing: PreprocessingConfig
    analysis: AnalysisConfig
    runtime: RuntimeConfig
    visualization: VisualizationConfig
    output: OutputConfig
    preset: str | None = None

    @property
    def task(self) -> str:
        return METHOD_TASKS[self.analysis.method]

    @property
    def images(self) -> InputConfig:
        """Compatibility alias for the pre-release Python API."""
        return self.input

    @property
    def attention(self) -> AttentionConfig | None:
        if self.analysis.method not in {"attention", "rollout"}:
            return None
        assert self.analysis.layers is not None
        assert self.analysis.head_fusion is not None
        return AttentionConfig(
            layers=self.analysis.layers,
            heads=self.analysis.heads,
            head_fusion=self.analysis.head_fusion,
        )

    @property
    def patch_pca(self) -> PatchPCAConfig | None:
        if self.analysis.method != "patch_pca":
            return None
        assert self.analysis.foreground_threshold is not None
        assert self.analysis.foreground_side is not None
        return PatchPCAConfig(
            foreground_threshold=self.analysis.foreground_threshold,
            foreground_side=self.analysis.foreground_side,
        )


def load_config(
    path: str | Path,
    *,
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> VisionLensConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    if not isinstance(raw_config, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")

    return parse_config(
        raw_config,
        base_dir=config_path.parent,
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
    base = Path.cwd().resolve() if base_dir is None else Path(base_dir).resolve()
    preset_name = _preset_name(raw_config, preset)
    resolved = get_preset(preset_name) if preset_name is not None else {}
    user_config = {key: value for key, value in raw_config.items() if key != "preset"}
    resolved = _deep_merge(resolved, user_config)
    resolved = _deep_merge(resolved, overrides or {})
    _validate_keys(resolved)

    input_section = _section(resolved, "input")
    model_section = _section(resolved, "model")
    preprocessing_section = _section(resolved, "preprocessing")
    analysis_section = _section(resolved, "analysis")
    runtime_section = _section(resolved, "runtime")
    visualization_section = _section(resolved, "visualization")
    output_section = _section(resolved, "output")
    method = _analysis_method(analysis_section.get("method", "attention"))
    _reject_unknown_keys("analysis", analysis_section, ANALYSIS_KEYS[method])

    config = VisionLensConfig(
        input=InputConfig(
            paths=tuple(
                _resolve_path(path, base)
                for path in _required_list(input_section, "paths", "input")
            )
        ),
        model=ModelConfig(
            architecture=_required_str(model_section, "architecture", "model"),
            backend=_required_str(model_section, "backend", "model"),
            name=_required_str(model_section, "name", "model"),
            pretrained=_bool(
                model_section.get("pretrained", True),
                "model.pretrained",
            ),
            options=_optional_mapping(
                model_section.get("options"),
                "model.options",
            ),
        ),
        preprocessing=PreprocessingConfig(
            image_size=_positive_int(
                preprocessing_section.get("image_size", 672),
                "preprocessing.image_size",
            )
        ),
        analysis=_parse_analysis(analysis_section, method),
        runtime=RuntimeConfig(
            device=_device(runtime_section.get("device", "auto")),
        ),
        visualization=VisualizationConfig(
            overlay_alpha=_unit_interval(
                visualization_section.get("overlay_alpha", 0.45),
                "visualization.overlay_alpha",
            ),
            cmap=_optional_str(
                visualization_section.get("cmap", "viridis"),
                "visualization.cmap",
            ),
            grid_format=_grid_format(visualization_section.get("grid_format", "png")),
        ),
        output=OutputConfig(
            directory=_resolve_path(
                _required_str(output_section, "directory", "output"),
                base,
            )
        ),
        preset=preset_name,
    )
    validate_config(config)
    return config


def validate_config(config: VisionLensConfig) -> None:
    missing_inputs = [path for path in config.input.paths if not path.is_file()]
    if missing_inputs:
        paths = ", ".join(str(path) for path in missing_inputs)
        raise ValueError(f"Input file(s) do not exist: {paths}.")

    method = config.analysis.method
    actual_pair = (config.model.architecture, config.model.backend)
    expected_pair = ("cnn", "torchvision") if method == "gradcam" else ("vit", "timm")
    if actual_pair != expected_pair:
        raise ValueError(
            f"analysis.method={method!r} requires model.architecture="
            f"{expected_pair[0]!r} and model.backend={expected_pair[1]!r}; got "
            f"architecture={actual_pair[0]!r}, backend={actual_pair[1]!r}."
        )

    options = config.model.options or {}
    if "img_size" in options:
        raise ValueError(
            "model.options.img_size is not allowed; use "
            "preprocessing.image_size as the authoritative input size."
        )

    fixed_size = KNOWN_FIXED_IMAGE_SIZES.get((config.model.backend, config.model.name))
    if fixed_size is not None and config.preprocessing.image_size != fixed_size:
        raise ValueError(
            f"model {config.model.name!r} requires preprocessing.image_size="
            f"{fixed_size}; got {config.preprocessing.image_size}. The full image "
            "will be resized to this model size without cropping."
        )

    if method in {"attention", "rollout"}:
        _validate_known_attention_constraints(config)


def config_to_dict(config: VisionLensConfig) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    if config.preset is not None:
        resolved["preset"] = config.preset
    resolved["input"] = {"paths": [str(path) for path in config.input.paths]}
    resolved["model"] = {
        "architecture": config.model.architecture,
        "backend": config.model.backend,
        "name": config.model.name,
        "pretrained": config.model.pretrained,
        "options": config.model.options,
    }
    resolved["preprocessing"] = {"image_size": config.preprocessing.image_size}
    resolved["analysis"] = _analysis_to_dict(config.analysis)
    resolved["runtime"] = {"device": config.runtime.device}
    resolved["visualization"] = {
        "overlay_alpha": config.visualization.overlay_alpha,
        "cmap": config.visualization.cmap,
        "grid_format": config.visualization.grid_format,
    }
    resolved["output"] = {"directory": str(config.output.directory)}
    return resolved


def resolved_config_yaml(config: VisionLensConfig) -> str:
    return yaml.safe_dump(config_to_dict(config), sort_keys=False)


def _parse_analysis(
    section: dict[str, Any],
    method: AnalysisMethod,
) -> AnalysisConfig:
    if method in {"attention", "rollout"}:
        return AnalysisConfig(
            method=method,
            layers=_attention_layers(section.get("layers")),
            heads=_optional_non_negative_ints(
                section.get("heads"),
                "analysis.heads",
            ),
            head_fusion=_head_fusion(section.get("head_fusion", "mean")),
        )
    if method == "gradcam":
        target_layer = section.get("target_layer")
        return AnalysisConfig(
            method=method,
            target_layer=(
                None
                if target_layer is None
                else _optional_str(target_layer, "analysis.target_layer")
            ),
        )
    return AnalysisConfig(
        method=method,
        foreground_threshold=_unit_interval(
            section.get("foreground_threshold", 0.5),
            "analysis.foreground_threshold",
        ),
        foreground_side=_foreground_side(section.get("foreground_side", "high")),
    )


def _analysis_to_dict(analysis: AnalysisConfig) -> dict[str, Any]:
    resolved: dict[str, Any] = {"method": analysis.method}
    if analysis.method in {"attention", "rollout"}:
        resolved["layers"] = (
            analysis.layers if analysis.layers == "all" else list(analysis.layers or ())
        )
        resolved["heads"] = None if analysis.heads is None else list(analysis.heads)
        resolved["head_fusion"] = analysis.head_fusion
    elif analysis.method == "gradcam":
        resolved["target_layer"] = analysis.target_layer
    else:
        resolved["foreground_threshold"] = analysis.foreground_threshold
        resolved["foreground_side"] = analysis.foreground_side
    return resolved


def _validate_keys(config: dict[str, Any]) -> None:
    _reject_unknown_keys("top level", config, TOP_LEVEL_KEYS)
    for section_name, allowed in SECTION_KEYS.items():
        _reject_unknown_keys(section_name, _section(config, section_name), allowed)


def _reject_unknown_keys(
    section_name: str,
    section: dict[str, Any],
    allowed: set[str],
) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        keys = ", ".join(unknown)
        allowed_keys = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unknown key(s) in {section_name}: {keys}. Allowed keys: {allowed_keys}."
        )


def _validate_known_attention_constraints(config: VisionLensConfig) -> None:
    analysis = config.analysis
    depth = KNOWN_VIT_DEPTHS.get(config.model.name)
    if depth is not None and analysis.layers != "all":
        assert analysis.layers is not None
        invalid_layers = [layer for layer in analysis.layers if layer >= depth]
        if invalid_layers:
            raise ValueError(
                f"analysis.layers contains {invalid_layers}; model "
                f"{config.model.name!r} has layers 0 through {depth - 1}."
            )

    head_count = KNOWN_VIT_HEADS.get(config.model.name)
    if head_count is not None and analysis.heads is not None:
        invalid_heads = [head for head in analysis.heads if head >= head_count]
        if invalid_heads:
            raise ValueError(
                f"analysis.heads contains {invalid_heads}; model "
                f"{config.model.name!r} has heads 0 through {head_count - 1}."
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
        raise ValueError("analysis.layers must be `all` or a non-empty list.")
    return tuple(_non_negative_ints(value, "analysis.layers"))


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
    return resolved.resolve()


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


def _analysis_method(value: Any) -> AnalysisMethod:
    allowed = set(ANALYSIS_KEYS)
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"analysis.method must be one of: {options}.")
    return value


def _head_fusion(value: Any) -> HeadFusion:
    allowed = {"mean", "max", "none"}
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"analysis.head_fusion must be one of: {options}.")
    return value


def _foreground_side(value: Any) -> Literal["high", "low"]:
    allowed = {"high", "low"}
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"analysis.foreground_side must be one of: {options}.")
    return value


def _device(value: Any) -> Device:
    allowed = {"auto", "cpu", "cuda", "mps"}
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"runtime.device must be one of: {options}.")
    return value


def _unit_interval(value: Any, field_name: str) -> float:
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
