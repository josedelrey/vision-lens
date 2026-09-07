from __future__ import annotations

from copy import deepcopy
from typing import Any

_EXAMPLE_IMAGES = [f"data/examples/{index}.jpg" for index in range(1, 9)]


PRESETS: dict[str, dict[str, Any]] = {
    "dino-vits8-attention": {
        "task": "vit_attention",
        "model": {
            "architecture": "vit",
            "backend": "timm",
            "name": "vit_small_patch8_224.dino",
            "pretrained": True,
            "options": {},
        },
        "images": {"paths": _EXAMPLE_IMAGES},
        "output": {"directory": "outputs/dino_vits8_attention"},
        "attention": {
            "layers": [2, 5, 8, 11],
            "heads": None,
            "head_fusion": "mean",
        },
        "runtime": {"device": "auto", "image_size": 224},
        "visualization": {
            "overlay_alpha": 0.8,
            "cmap": "viridis",
            "grid_format": "pdf",
        },
    },
    "dinov2-reg4-attention": {
        "task": "vit_attention",
        "model": {
            "architecture": "vit",
            "backend": "timm",
            "name": "hf_hub:timm/vit_small_patch14_reg4_dinov2.lvd142m",
            "pretrained": True,
            "options": {"img_size": 672},
        },
        "images": {"paths": _EXAMPLE_IMAGES},
        "output": {"directory": "outputs/dinov2_vits14_reg4_attention"},
        "attention": {
            "layers": [2, 5, 8, 11],
            "heads": None,
            "head_fusion": "mean",
        },
        "runtime": {"device": "auto", "image_size": 672},
        "visualization": {
            "overlay_alpha": 0.8,
            "cmap": "viridis",
            "grid_format": "pdf",
        },
    },
    "dinov2-reg4-rollout": {
        "task": "vit_rollout",
        "model": {
            "architecture": "vit",
            "backend": "timm",
            "name": "hf_hub:timm/vit_small_patch14_reg4_dinov2.lvd142m",
            "pretrained": True,
            "options": {"img_size": 672},
        },
        "images": {"paths": _EXAMPLE_IMAGES},
        "output": {"directory": "outputs/dinov2_reg4_attention/rollout"},
        "attention": {
            "layers": [2, 5, 8, 11],
            "heads": None,
            "head_fusion": "mean",
        },
        "runtime": {"device": "auto", "image_size": 672},
        "visualization": {
            "overlay_alpha": 0.8,
            "cmap": "viridis",
            "grid_format": "pdf",
        },
    },
    "resnet50-gradcam": {
        "task": "gradcam",
        "model": {
            "architecture": "cnn",
            "backend": "torchvision",
            "name": "resnet50",
            "pretrained": True,
            "options": {"gradcam_target_layer": "layer4"},
        },
        "images": {"paths": _EXAMPLE_IMAGES},
        "output": {"directory": "outputs/gradcam"},
        "runtime": {"device": "auto", "image_size": 224},
        "visualization": {
            "overlay_alpha": 0.8,
            "cmap": "viridis",
            "grid_format": "pdf",
        },
    },
    "dinov2-pca": {
        "task": "patch_pca",
        "model": {
            "architecture": "vit",
            "backend": "timm",
            "name": "hf_hub:timm/vit_base_patch14_dinov2.lvd142m",
            "pretrained": True,
            "options": {"img_size": 672},
        },
        "images": {"paths": ["data/examples/5.jpg", "data/examples/6.jpg"]},
        "output": {"directory": "outputs/dinov2_patch_pca"},
        "runtime": {"device": "auto", "image_size": 672},
        "patch_pca": {
            "foreground_threshold": 0.5,
            "foreground_side": "low",
        },
        "visualization": {},
    },
}


def available_presets() -> tuple[str, ...]:
    return tuple(PRESETS)


def get_preset(name: str) -> dict[str, Any]:
    try:
        return deepcopy(PRESETS[name])
    except KeyError as error:
        choices = ", ".join(available_presets())
        raise ValueError(
            f"Unknown preset {name!r}. Available presets: {choices}."
        ) from error
