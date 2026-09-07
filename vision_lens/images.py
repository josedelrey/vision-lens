from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from timm.data import resolve_model_data_config
from timm.data.transforms import str_to_interp_mode
from torchvision import transforms

from vision_lens.config import PreprocessingConfig


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


def load_images(paths: Iterable[str | Path], workers: int = 0) -> list:
    image_paths = tuple(paths)
    if workers <= 0:
        return [load_image(path) for path in image_paths]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(load_image, image_paths))


def build_timm_preprocess(
    model: Any,
    image_size: int | tuple[int, int] | None = None,
    config: PreprocessingConfig | None = None,
    data_config: dict[str, Any] | None = None,
):
    resolved_data_config = data_config or resolve_model_data_config(model)
    size = _image_size(
        image_size,
        resolved_data_config.get("input_size"),
        default=672,
    )
    interpolation = str_to_interp_mode(
        resolved_data_config.get("interpolation", "bilinear")
    )
    return _resize_and_normalize(
        size,
        interpolation=interpolation,
        mean=resolved_data_config.get("mean", (0.485, 0.456, 0.406)),
        std=resolved_data_config.get("std", (0.229, 0.224, 0.225)),
        config=config,
    )


def build_preprocess(
    model: Any,
    backend: str,
    image_size: int | tuple[int, int] | None = None,
    config: PreprocessingConfig | None = None,
    data_config: dict[str, Any] | None = None,
):
    if backend == "timm":
        return build_timm_preprocess(
            model,
            image_size=image_size,
            config=config,
            data_config=data_config,
        )
    if backend == "torchvision":
        return build_torchvision_preprocess(
            image_size=image_size,
            config=config,
            data_config=data_config,
        )

    raise ValueError(f"Unsupported preprocessing backend: {backend}")


def build_torchvision_preprocess(
    image_size: int | tuple[int, int] | None = None,
    config: PreprocessingConfig | None = None,
    data_config: dict[str, Any] | None = None,
):
    size = _image_size(image_size, default=672)
    resolved_data_config = data_config or {}
    return _resize_and_normalize(
        size,
        interpolation=str_to_interp_mode(
            resolved_data_config.get("interpolation", "bilinear")
        ),
        mean=resolved_data_config.get("mean", (0.485, 0.456, 0.406)),
        std=resolved_data_config.get("std", (0.229, 0.224, 0.225)),
        config=config,
    )


def _resize_and_normalize(
    size: tuple[int, int],
    *,
    interpolation: transforms.InterpolationMode,
    mean: Iterable[float],
    std: Iterable[float],
    config: PreprocessingConfig | None = None,
):
    settings = config or PreprocessingConfig(image_size=size[0])
    if settings.interpolation is not None:
        interpolation = str_to_interp_mode(settings.interpolation)

    operations = []
    if settings.resize == "stretch":
        operations.append(
            transforms.Resize(size, interpolation=interpolation, antialias=True)
        )
    elif settings.resize == "shortest":
        operations.append(
            transforms.Resize(min(size), interpolation=interpolation, antialias=True)
        )
    elif settings.resize == "longest":
        operations.append(
            transforms.Lambda(lambda image: _resize_longest(image, size, interpolation))
        )

    if settings.crop == "center":
        operations.append(transforms.CenterCrop(size))
    if settings.pad == "center":
        operations.append(transforms.Lambda(lambda image: _pad_to_size(image, size)))

    operations.append(transforms.ToTensor())
    if settings.normalize:
        operations.append(
            transforms.Normalize(
                mean=tuple(settings.mean or mean),
                std=tuple(settings.std or std),
            )
        )
    return transforms.Compose(operations)


def _resize_longest(
    image: Image.Image,
    size: tuple[int, int],
    interpolation: transforms.InterpolationMode,
) -> Image.Image:
    target_height, target_width = size
    scale = min(target_width / image.width, target_height / image.height)
    resized = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(resized, resample=_pil_resampling(interpolation))


def _pad_to_size(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_height, target_width = size
    if image.width > target_width or image.height > target_height:
        raise ValueError(
            "preprocessing.pad='center' cannot shrink an image; use "
            "resize='longest' first."
        )
    result = Image.new("RGB", (target_width, target_height), "black")
    result.paste(
        image,
        ((target_width - image.width) // 2, (target_height - image.height) // 2),
    )
    return result


def _pil_resampling(mode: transforms.InterpolationMode) -> Image.Resampling:
    mapping = {
        transforms.InterpolationMode.NEAREST: Image.Resampling.NEAREST,
        transforms.InterpolationMode.BILINEAR: Image.Resampling.BILINEAR,
        transforms.InterpolationMode.BICUBIC: Image.Resampling.BICUBIC,
        transforms.InterpolationMode.LANCZOS: Image.Resampling.LANCZOS,
    }
    return mapping[mode]


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
