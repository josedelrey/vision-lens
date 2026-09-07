from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import timm
import torch
from timm.data import ImageNetInfo, resolve_model_data_config
from torchvision.models import get_model, get_model_weights

from vision_lens.config import ModelConfig, RuntimeConfig, VisionLensConfig


@dataclass(frozen=True)
class ModelMetadata:
    architecture: str
    backend: str
    name: str
    pretrained: bool
    device: str
    input_size: tuple[int, int, int]
    image_size: tuple[int, int]
    patch_size: tuple[int, int] | None
    num_classes: int | None
    data_config: dict[str, Any]
    class_labels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    metadata: ModelMetadata


def load_model(config: VisionLensConfig) -> LoadedModel:
    return load_configured_model(config.model, config.runtime)


def load_configured_model(
    model_config: ModelConfig,
    runtime_config: RuntimeConfig,
) -> LoadedModel:
    if model_config.architecture == "vit" and model_config.backend == "timm":
        return load_timm_vit(model_config, runtime_config)
    if model_config.architecture == "cnn" and model_config.backend == "torchvision":
        return load_torchvision_cnn(model_config, runtime_config)

    raise ValueError(
        "Unsupported model configuration: "
        f"architecture={model_config.architecture!r}, "
        f"backend={model_config.backend!r}."
    )


def load_torchvision_cnn(
    model_config: ModelConfig,
    runtime_config: RuntimeConfig,
) -> LoadedModel:
    device = resolve_device(runtime_config.device)
    model_options = model_config.options or {}
    model_options = {
        key: value
        for key, value in model_options.items()
        if key != "gradcam_target_layer"
    }
    weights = None
    class_labels = _imagenet_labels()
    if model_config.pretrained:
        weights = get_model_weights(model_config.name).DEFAULT
        class_labels = tuple(weights.meta.get("categories", class_labels or ()))

    model = get_model(model_config.name, weights=weights, **model_options)
    model.eval()
    model.to(torch.device(device))

    data_config = {
        "input_size": (3, runtime_config.image_size, runtime_config.image_size),
        "interpolation": "bilinear",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "crop_pct": 1.0,
        "crop_mode": "none",
    }

    metadata = ModelMetadata(
        architecture=model_config.architecture,
        backend=model_config.backend,
        name=model_config.name,
        pretrained=model_config.pretrained,
        device=device,
        input_size=_input_size(data_config),
        image_size=(runtime_config.image_size, runtime_config.image_size),
        patch_size=None,
        num_classes=_num_classes(model),
        data_config=data_config,
        class_labels=class_labels,
    )
    return LoadedModel(model=model, metadata=metadata)


def load_timm_vit(
    model_config: ModelConfig,
    runtime_config: RuntimeConfig,
) -> LoadedModel:
    device = resolve_device(runtime_config.device)
    model_options = dict(model_config.options or {})
    model = timm.create_model(
        model_config.name,
        pretrained=model_config.pretrained,
        **model_options,
    )
    model.eval()
    model.to(torch.device(device))

    image_size = _accepted_timm_image_size(model, runtime_config.image_size)
    data_config = dict(resolve_model_data_config(model))
    data_config["input_size"] = (
        3,
        image_size[0],
        image_size[1],
    )
    data_config["crop_pct"] = 1.0
    data_config["crop_mode"] = "none"

    metadata = ModelMetadata(
        architecture=model_config.architecture,
        backend=model_config.backend,
        name=model_config.name,
        pretrained=model_config.pretrained,
        device=device,
        input_size=_input_size(data_config),
        image_size=image_size,
        patch_size=_patch_size(model),
        num_classes=_num_classes(model),
        data_config=data_config,
        class_labels=_imagenet_labels(),
    )
    return LoadedModel(model=model, metadata=metadata)


def _accepted_timm_image_size(
    model: Any,
    requested_image_size: int,
) -> tuple[int, int]:
    requested = (requested_image_size, requested_image_size)
    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is None or not getattr(patch_embed, "strict_img_size", False):
        return requested

    fixed_size = getattr(patch_embed, "img_size", None)
    if isinstance(fixed_size, int):
        return (fixed_size, fixed_size)
    if (
        isinstance(fixed_size, (list, tuple))
        and len(fixed_size) == 2
        and all(isinstance(value, int) for value in fixed_size)
    ):
        return (fixed_size[0], fixed_size[1])
    return requested


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _input_size(data_config: dict[str, Any]) -> tuple[int, int, int]:
    input_size = data_config.get("input_size")
    if (
        not isinstance(input_size, tuple)
        or len(input_size) != 3
        or not all(isinstance(value, int) for value in input_size)
    ):
        raise ValueError("Model data config must include a 3-value input_size tuple.")
    return input_size


def _patch_size(model: Any) -> tuple[int, int] | None:
    patch_embed = getattr(model, "patch_embed", None)
    patch_size = getattr(patch_embed, "patch_size", None)

    if isinstance(patch_size, int):
        return (patch_size, patch_size)
    if (
        isinstance(patch_size, tuple)
        and len(patch_size) == 2
        and all(isinstance(value, int) for value in patch_size)
    ):
        return patch_size
    return None


def _num_classes(model: Any) -> int | None:
    num_classes = getattr(model, "num_classes", None)
    if isinstance(num_classes, int):
        return num_classes

    pretrained_cfg = getattr(model, "pretrained_cfg", {})
    if isinstance(pretrained_cfg, dict):
        cfg_num_classes = pretrained_cfg.get("num_classes")
        if isinstance(cfg_num_classes, int):
            return cfg_num_classes
    return None


def _imagenet_labels() -> tuple[str, ...] | None:
    return tuple(ImageNetInfo().label_descriptions())
