from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from vision_lens.attention import infer_patch_grid_from_image
from vision_lens.models import ModelMetadata


@dataclass(frozen=True)
class PatchPCAResult:
    patch_embeddings: Any
    foreground_mask: Any
    images: tuple[Image.Image, ...]
    patch_grid: tuple[int, int]
    image_size: tuple[int, int]


def extract_patch_pca(
    model: Any,
    inputs: Any,
    metadata: ModelMetadata,
    foreground_threshold: float = 0.5,
    foreground_side: Literal["high", "low"] = "high",
) -> PatchPCAResult:
    """Extract ViT patch tokens and render their shared PCA projection as RGB."""
    if metadata.patch_size is None:
        raise ValueError("Patch PCA requires a model with a known patch size.")

    image_size = (int(inputs.shape[-2]), int(inputs.shape[-1]))
    patch_grid = infer_patch_grid_from_image(image_size, metadata.patch_size)

    with torch.no_grad():
        model_inputs = inputs.to(_model_device(model))
        features = model.forward_features(model_inputs)

    patch_embeddings = _patch_tokens_from_features(
        features,
        patch_count=patch_grid[0] * patch_grid[1],
    ).detach().float().cpu()
    return project_patch_embeddings(
        patch_embeddings,
        patch_grid=patch_grid,
        image_size=image_size,
        foreground_threshold=foreground_threshold,
        foreground_side=foreground_side,
    )


def project_patch_embeddings(
    patch_embeddings: Any,
    patch_grid: tuple[int, int],
    image_size: tuple[int, int],
    foreground_threshold: float = 0.5,
    foreground_side: Literal["high", "low"] = "high",
) -> PatchPCAResult:
    """Project a batch of patch embeddings into one shared RGB PCA space."""
    if not 0 <= foreground_threshold <= 1:
        raise ValueError("foreground_threshold must be between 0 and 1.")
    if foreground_side not in {"high", "low"}:
        raise ValueError("foreground_side must be one of: high, low.")

    embeddings = torch.as_tensor(patch_embeddings).detach().float().cpu()
    if embeddings.ndim != 3:
        raise ValueError(
            "patch_embeddings must have shape (batch, patches, features)."
        )

    batch_size, patch_count, feature_count = embeddings.shape
    expected_patch_count = patch_grid[0] * patch_grid[1]
    if patch_count != expected_patch_count:
        raise ValueError(
            f"patch_grid describes {expected_patch_count} patches, "
            f"but patch_embeddings contains {patch_count}."
        )
    if batch_size == 0 or feature_count == 0:
        raise ValueError("patch_embeddings must not be empty.")

    flattened = embeddings.reshape(batch_size * patch_count, feature_count)
    first_component = _pca_projection(flattened, components=1)
    first_component = _minmax_normalize(first_component)
    if foreground_side == "high":
        foreground_mask = first_component[:, 0] > foreground_threshold
    else:
        foreground_mask = first_component[:, 0] < foreground_threshold

    rgb_patches = torch.zeros(
        (batch_size * patch_count, 3),
        dtype=flattened.dtype,
    )
    foreground_embeddings = flattened[foreground_mask]
    if foreground_embeddings.numel():
        foreground_rgb = _pca_projection(foreground_embeddings, components=3)
        foreground_rgb = _minmax_normalize(foreground_rgb)
        if foreground_rgb.shape[1] < 3:
            foreground_rgb = functional.pad(
                foreground_rgb,
                (0, 3 - foreground_rgb.shape[1]),
            )
        rgb_patches[foreground_mask] = foreground_rgb[:, :3]

    patch_images = rgb_patches.reshape(
        batch_size,
        patch_grid[0],
        patch_grid[1],
        3,
    ).permute(0, 3, 1, 2)
    rendered = functional.interpolate(
        patch_images,
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )
    rendered = rendered.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    images = tuple(
        Image.fromarray((array * 255).round().astype(np.uint8), mode="RGB")
        for array in rendered
    )

    return PatchPCAResult(
        patch_embeddings=embeddings,
        foreground_mask=foreground_mask.reshape(batch_size, patch_count),
        images=images,
        patch_grid=patch_grid,
        image_size=image_size,
    )


def _patch_tokens_from_features(features: Any, patch_count: int) -> Any:
    tokens = features
    contains_only_patches = False
    if isinstance(features, dict):
        if "x_norm_patchtokens" in features:
            tokens = features["x_norm_patchtokens"]
            contains_only_patches = True
        else:
            for key in ("x_prenorm", "last_hidden_state", "x"):
                if key in features:
                    tokens = features[key]
                    break
            else:
                available = ", ".join(sorted(str(key) for key in features))
                raise ValueError(
                    "Could not find patch tokens in model features. "
                    f"Available keys: {available}."
                )

    if not hasattr(tokens, "ndim"):
        raise ValueError("Model forward_features did not return a tensor.")
    if tokens.ndim == 4:
        tokens = tokens.flatten(2).transpose(1, 2)
        contains_only_patches = True
    if tokens.ndim != 3:
        raise ValueError(
            "Model features must have shape (batch, tokens, features) or "
            "(batch, features, height, width)."
        )

    token_count = int(tokens.shape[1])
    if contains_only_patches:
        if token_count != patch_count:
            raise ValueError(
                f"Model returned {token_count} patch tokens; expected {patch_count}."
            )
        return tokens

    prefix_token_count = token_count - patch_count
    if prefix_token_count < 0:
        raise ValueError(
            f"Model returned {token_count} tokens; expected at least {patch_count}."
        )
    return tokens[:, prefix_token_count:]


def _pca_projection(values: Any, components: int) -> Any:
    component_count = min(components, int(values.shape[0]), int(values.shape[1]))
    if component_count == 0:
        return values.new_zeros((values.shape[0], 0))
    if values.shape[0] == 1:
        return values[:, :component_count]

    with torch.random.fork_rng():
        torch.manual_seed(0)
        _u, _s, vectors = torch.pca_lowrank(
            values,
            q=component_count,
            center=True,
        )
    return values @ vectors[:, :component_count]


def _minmax_normalize(values: Any) -> Any:
    minimum = values.min(dim=0).values
    maximum = values.max(dim=0).values
    span = maximum - minimum
    safe_span = torch.where(span > 0, span, torch.ones_like(span))
    normalized = (values - minimum) / safe_span
    return torch.where(span > 0, normalized, torch.zeros_like(normalized))


def _model_device(model: Any) -> Any:
    return next(model.parameters()).device
