from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from timm.data import resolve_model_data_config
from timm.data.transforms import str_to_interp_mode
from torchvision import transforms


def example_image_paths(example_dir: str | Path = "data/examples") -> tuple[Path, ...]:
    directory = Path(example_dir)
    return tuple(
        sorted(
            path
            for path in directory.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
    )


def load_image(path: str | Path):
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    with Image.open(image_path) as image:
        return image.convert("RGB")


def load_images(paths: Iterable[str | Path]) -> list:
    return [load_image(path) for path in paths]


def build_timm_preprocess(
    model: Any,
    image_size: int | tuple[int, int] | None = None,
):
    data_config = resolve_model_data_config(model)
    size = _image_size(image_size, data_config.get("input_size"), default=672)
    interpolation = str_to_interp_mode(data_config.get("interpolation", "bilinear"))
    return _resize_and_normalize(
        size,
        interpolation=interpolation,
        mean=data_config.get("mean", (0.485, 0.456, 0.406)),
        std=data_config.get("std", (0.229, 0.224, 0.225)),
    )


def build_preprocess(
    model: Any,
    backend: str,
    image_size: int | tuple[int, int] | None = None,
):
    if backend == "timm":
        return build_timm_preprocess(model, image_size=image_size)
    if backend == "torchvision":
        return build_torchvision_preprocess(image_size=image_size)

    raise ValueError(f"Unsupported preprocessing backend: {backend}")


def build_torchvision_preprocess(
    image_size: int | tuple[int, int] | None = None,
):
    size = _image_size(image_size, default=672)
    return _resize_and_normalize(
        size,
        interpolation=transforms.InterpolationMode.BILINEAR,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def _resize_and_normalize(
    size: tuple[int, int],
    *,
    interpolation: transforms.InterpolationMode,
    mean: Iterable[float],
    std: Iterable[float],
):
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=interpolation, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )


def _image_size(
    requested: int | tuple[int, int] | None,
    model_input_size: Any = None,
    *,
    default: int,
) -> tuple[int, int]:
    if isinstance(requested, int):
        return (requested, requested)
    if (
        isinstance(requested, tuple)
        and len(requested) == 2
        and all(isinstance(value, int) for value in requested)
    ):
        return requested
    if (
        isinstance(model_input_size, (list, tuple))
        and len(model_input_size) == 3
        and all(isinstance(value, int) for value in model_input_size)
    ):
        return (model_input_size[1], model_input_size[2])
    return (default, default)


def preprocess_image(image: Any, transform: Any):
    return transform(image).unsqueeze(0)


def preprocess_images(images: Iterable[Any], transform: Any):
    return torch.cat([preprocess_image(image, transform) for image in images], dim=0)


def tensor_to_display_image(
    tensor: Any,
    mean: Iterable[float] = (0.485, 0.456, 0.406),
    std: Iterable[float] = (0.229, 0.224, 0.225),
):
    if tensor.ndim != 3:
        raise ValueError("tensor must have shape (channels, height, width).")

    if tensor.shape[0] != 3:
        raise ValueError("tensor must have 3 channels.")

    image = tensor.detach().cpu()

    mean_tensor = torch.tensor(
        tuple(mean),
        dtype=image.dtype,
        device=image.device,
    )[:, None, None]
    std_tensor = torch.tensor(
        tuple(std),
        dtype=image.dtype,
        device=image.device,
    )[:, None, None]

    image = image * std_tensor + mean_tensor
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).round().astype(np.uint8)

    return Image.fromarray(image).convert("RGB")


def tensors_to_display_images(
    tensors: Any,
    data_config: dict[str, Any] | None = None,
) -> list:
    if tensors.ndim != 4:
        raise ValueError("tensors must have shape (batch, channels, height, width).")

    config = {} if data_config is None else data_config
    mean = config.get("mean", (0.485, 0.456, 0.406))
    std = config.get("std", (0.229, 0.224, 0.225))

    return [tensor_to_display_image(tensor, mean=mean, std=std) for tensor in tensors]
