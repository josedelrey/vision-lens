from __future__ import annotations

from pathlib import Path
from typing import Iterable


def sample_image_paths(sample_dir: str | Path = "data/samples") -> tuple[Path, ...]:
    directory = Path(sample_dir)
    return tuple(
        sorted(
            path
            for path in directory.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
    )


def load_image(path: str | Path):
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required to load images. "
            'Install the project with `python -m pip install -e ".[dev]"`.'
        ) from error

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    with Image.open(image_path) as image:
        return image.convert("RGB")


def load_images(paths: Iterable[str | Path]) -> list:
    return [load_image(path) for path in paths]


def build_timm_preprocess(model, image_size: int | None = None):
    try:
        from timm.data import create_transform, resolve_model_data_config
    except ImportError as error:
        raise RuntimeError(
            "timm is required to build preprocessing for timm models. "
            'Install the project with `python -m pip install -e ".[dev]"`.'
        ) from error

    data_config = resolve_model_data_config(model)
    if image_size is not None:
        data_config["input_size"] = (3, image_size, image_size)

    return create_transform(**data_config, is_training=False)


def build_preprocess(model, backend: str, image_size: int | None = None):
    if backend == "timm":
        return build_timm_preprocess(model, image_size=image_size)
    if backend == "torchvision":
        return build_torchvision_preprocess(image_size=image_size)

    raise ValueError(f"Unsupported preprocessing backend: {backend}")


def build_torchvision_preprocess(image_size: int | None = None):
    try:
        from PIL import Image as _Image  # noqa: F401
        from torchvision import transforms
    except ImportError as error:
        raise RuntimeError(
            "torchvision is required to build preprocessing for torchvision "
            "models. Install the project dependencies in the cv environment."
        ) from error

    size = 224 if image_size is None else image_size
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def preprocess_image(image, transform):
    return transform(image).unsqueeze(0)


def preprocess_images(images: Iterable, transform):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to batch preprocessed images. "
            'Install the project with `python -m pip install -e ".[dev]"`.'
        ) from error

    return torch.cat([preprocess_image(image, transform) for image in images], dim=0)
