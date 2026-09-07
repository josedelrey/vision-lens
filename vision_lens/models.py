from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import timm
import torch
from timm.data import ImageNetInfo, resolve_model_data_config
from torchvision.models import get_model, get_model_weights

from vision_lens.config import (
    ModelConfig,
    PreprocessingConfig,
    RuntimeConfig,
    VisionLensConfig,
)


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
    return load_configured_model(
        config.model,
        config.preprocessing,
        config.runtime,
    )


def load_configured_model(
    model_config: ModelConfig,
    preprocessing_config: PreprocessingConfig,
    runtime_config: RuntimeConfig,
) -> LoadedModel:
    if model_config.architecture == "vit" and model_config.backend == "timm":
        return load_timm_vit(model_config, preprocessing_config, runtime_config)
    if model_config.architecture == "cnn" and model_config.backend == "torchvision":
        return load_torchvision_cnn(
            model_config,
            preprocessing_config,
            runtime_config,
        )

    raise ValueError(
        "Unsupported model configuration: "
        f"architecture={model_config.architecture!r}, "
        f"backend={model_config.backend!r}."
    )


def load_torchvision_cnn(
    model_config: ModelConfig,
    preprocessing_config: PreprocessingConfig,
    runtime_config: RuntimeConfig,
) -> LoadedModel:
    device = resolve_device(runtime_config.device)
    _validate_precision(runtime_config.precision, device)
    model_options = model_config.options or {}
    weights = None
    class_labels = _imagenet_labels()
    weights_transform = None
    if model_config.pretrained:
        weights = get_model_weights(model_config.name).DEFAULT
        class_labels = tuple(weights.meta.get("categories", class_labels or ()))
        weights_transform = weights.transforms()

    model = get_model(model_config.name, weights=weights, **model_options)
    model.eval()
    _move_model(model, device, runtime_config.precision)

    data_config = {
        "input_size": (
            3,
            preprocessing_config.image_size,
            preprocessing_config.image_size,
        ),
        "interpolation": (
            weights_transform.interpolation.value
            if weights_transform is not None
            else "bilinear"
        ),
        "mean": (
            tuple(weights_transform.mean)
            if weights_transform is not None
            else (0.485, 0.456, 0.406)
        ),
        "std": (
            tuple(weights_transform.std)
            if weights_transform is not None
            else (0.229, 0.224, 0.225)
        ),
        "crop_pct": 1.0,
        "crop_mode": "none",
    }
    _apply_preprocessing_overrides(data_config, preprocessing_config)

    metadata = ModelMetadata(
        architecture=model_config.architecture,
        backend=model_config.backend,
        name=model_config.name,
        pretrained=model_config.pretrained,
        device=device,
        input_size=_input_size(data_config),
        image_size=(
            preprocessing_config.image_size,
            preprocessing_config.image_size,
        ),
        patch_size=None,
        num_classes=_num_classes(model),
        data_config=data_config,
        class_labels=class_labels,
    )
    return LoadedModel(model=model, metadata=metadata)


def load_timm_vit(
    model_config: ModelConfig,
    preprocessing_config: PreprocessingConfig,
    runtime_config: RuntimeConfig,
) -> LoadedModel:
    device = resolve_device(runtime_config.device)
    _validate_precision(runtime_config.precision, device)
    model_options = dict(model_config.options or {})
    model_options["img_size"] = preprocessing_config.image_size
    try:
        model = timm.create_model(
            model_config.name,
            pretrained=model_config.pretrained,
            **model_options,
        )
    except TypeError as error:
        if "img_size" not in str(error):
            raise
        raise ValueError(
            f"model {model_config.name!r} does not accept "
            f"preprocessing.image_size={preprocessing_config.image_size}."
        ) from error
    model.eval()
    _move_model(model, device, runtime_config.precision)

    image_size = _accepted_timm_image_size(
        model,
        preprocessing_config.image_size,
    )
    expected_image_size = (
        preprocessing_config.image_size,
        preprocessing_config.image_size,
    )
    if image_size != expected_image_size:
        raise ValueError(
            f"model {model_config.name!r} accepted image size {image_size}, not "
            f"configured preprocessing.image_size={preprocessing_config.image_size}."
        )
    data_config = dict(resolve_model_data_config(model))
    data_config["input_size"] = (
        3,
        image_size[0],
        image_size[1],
    )
    data_config["crop_pct"] = 1.0
    data_config["crop_mode"] = "none"
    _apply_preprocessing_overrides(data_config, preprocessing_config)

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
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "runtime.device='cuda' was requested, but CUDA is unavailable."
        )
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("runtime.device='mps' was requested, but MPS is unavailable.")
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _move_model(model: Any, device: str, precision: str) -> None:
    model.to(torch.device(device))
    if precision == "float16":
        model.to(dtype=torch.float16)
    elif precision == "bfloat16":
        model.to(dtype=torch.bfloat16)


def _validate_precision(precision: str, device: str) -> None:
    if precision == "float16" and device == "cpu":
        raise ValueError(
            "runtime.precision='float16' is not supported on CPU; use "
            "float32 or bfloat16."
        )
    if precision == "bfloat16" and device == "mps":
        raise ValueError(
            "runtime.precision='bfloat16' is not supported on MPS; use "
            "float32 or float16."
        )


def _apply_preprocessing_overrides(
    data_config: dict[str, Any],
    config: PreprocessingConfig,
) -> None:
    if config.interpolation is not None:
        data_config["interpolation"] = config.interpolation
    if not config.normalize:
        data_config["mean"] = (0.0, 0.0, 0.0)
        data_config["std"] = (1.0, 1.0, 1.0)
    else:
        if config.mean is not None:
            data_config["mean"] = config.mean
        if config.std is not None:
            data_config["std"] = config.std


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
