import numpy as np
import torch
from PIL import Image

from vision_lens.config import PreprocessingConfig
from vision_lens.images import build_timm_preprocess, build_torchvision_preprocess


def test_torchvision_preprocess_resizes_rectangular_images_without_cropping():
    image = _horizontal_gradient(width=16, height=8)

    tensor = build_torchvision_preprocess(image_size=8)(image)
    restored_red = tensor[0] * 0.229 + 0.485

    assert tensor.shape == (3, 8, 8)
    assert restored_red[:, 0].mean() < 0.1
    assert restored_red[:, -1].mean() > 0.9


def test_torchvision_preprocess_keeps_672_square_input_dimensions():
    image = Image.new("RGB", (672, 672), "white")

    tensor = build_torchvision_preprocess()(image)

    assert tensor.shape == (3, 672, 672)


def test_timm_preprocess_uses_model_normalization_without_cropping(monkeypatch):
    from vision_lens import images

    monkeypatch.setattr(
        images,
        "resolve_model_data_config",
        lambda _model: {
            "input_size": (3, 224, 224),
            "interpolation": "bilinear",
            "mean": (0.0, 0.0, 0.0),
            "std": (1.0, 1.0, 1.0),
            "crop_pct": 0.9,
            "crop_mode": "center",
        },
    )

    tensor = build_timm_preprocess(object(), image_size=(8, 8))(
        _horizontal_gradient(width=16, height=8)
    )

    assert tensor.shape == (3, 8, 8)
    assert tensor[0, :, 0].mean() < 0.1
    assert tensor[0, :, -1].mean() > 0.9


def test_longest_resize_with_center_padding_preserves_aspect_ratio():
    image = Image.new("RGB", (16, 8), "white")
    config = PreprocessingConfig(
        image_size=8,
        resize="longest",
        pad="center",
        normalize=False,
    )

    tensor = build_torchvision_preprocess(image_size=8, config=config)(image)

    assert tensor.shape == (3, 8, 8)
    assert tensor[:, :2].max() == 0
    assert tensor[:, 2:6].min() == 1
    assert tensor[:, 6:].max() == 0


def test_custom_normalization_overrides_model_defaults():
    config = PreprocessingConfig(
        image_size=4,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    tensor = build_torchvision_preprocess(image_size=4, config=config)(
        Image.new("RGB", (4, 4), (128, 128, 128))
    )

    assert torch.allclose(tensor, torch.full_like(tensor, 1 / 255), atol=1e-6)


def _horizontal_gradient(width: int, height: int) -> Image.Image:
    values = torch.linspace(0, 255, width, dtype=torch.uint8).numpy()
    red = np.tile(values, (height, 1))
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :, 0] = red
    return Image.fromarray(array, mode="RGB")
