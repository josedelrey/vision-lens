from types import SimpleNamespace

from vision_lens.config import ModelConfig, PreprocessingConfig, RuntimeConfig
from vision_lens.models import _accepted_timm_image_size, load_timm_vit


def test_fixed_timm_model_uses_the_size_it_actually_accepts():
    model = SimpleNamespace(
        patch_embed=SimpleNamespace(strict_img_size=True, img_size=(224, 224))
    )

    assert _accepted_timm_image_size(model, 672) == (224, 224)


def test_dynamic_timm_model_uses_the_requested_672_size():
    model = SimpleNamespace(
        patch_embed=SimpleNamespace(strict_img_size=False, img_size=(224, 224))
    )

    assert _accepted_timm_image_size(model, 672) == (672, 672)


def test_timm_loader_uses_authoritative_preprocessing_size(monkeypatch):
    from vision_lens import models

    calls = []
    model = SimpleNamespace(
        patch_embed=SimpleNamespace(
            strict_img_size=True,
            img_size=(672, 672),
            patch_size=(14, 14),
        ),
        num_classes=1000,
        eval=lambda: None,
        to=lambda _device: None,
    )
    monkeypatch.setattr(
        models.timm,
        "create_model",
        lambda name, **kwargs: calls.append((name, kwargs)) or model,
    )
    monkeypatch.setattr(
        models,
        "resolve_model_data_config",
        lambda _model: {
            "input_size": (3, 518, 518),
            "interpolation": "bicubic",
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        },
    )
    monkeypatch.setattr(models, "_imagenet_labels", lambda: None)

    loaded = load_timm_vit(
        ModelConfig("vit", "timm", "mock_vit", pretrained=False),
        PreprocessingConfig(image_size=672),
        RuntimeConfig(device="cpu"),
    )

    assert calls == [
        ("mock_vit", {"pretrained": False, "img_size": 672}),
    ]
    assert loaded.metadata.input_size == (3, 672, 672)
    assert loaded.metadata.image_size == (672, 672)
    assert loaded.metadata.data_config["crop_mode"] == "none"
