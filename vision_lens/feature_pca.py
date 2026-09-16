from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from vision_lens.anyup import (
    is_anyup_interpolation,
    upsample_features,
    upsample_values_streaming,
)
from vision_lens.attention import infer_patch_grid_from_image
from vision_lens.models import ModelMetadata

ForegroundThreshold = float | Literal["auto"]
RGBFitScope = Literal["foreground", "all"]
Interpolation = Literal[
    "nearest",
    "bilinear",
    "bilinear_mask",
    "anyup",
    "anyup_mask",
    "anyup_soft",
    "anyup_soft_mask",
]
_OTSU_BINS = 256


@dataclass(frozen=True)
class PatchPCAResult:
    patch_embeddings: Any
    foreground_mask: Any
    images: tuple[Image.Image, ...]
    patch_grid: tuple[int, int]
    image_size: tuple[int, int]
    projection: PatchPCAProjection | None = None
    rgb_patches: Any | None = None
    interpolation: Interpolation = "bilinear"


@dataclass(frozen=True)
class PatchPCAProjection:
    foreground_components: Any
    foreground_minimum: Any
    foreground_maximum: Any
    rgb_components: Any
    rgb_minimum: Any
    rgb_maximum: Any
    foreground_threshold: float
    foreground_side: Literal["high", "low"]
    rgb_fit_scope: RGBFitScope
    foreground_separation: bool


def extract_patch_pca(
    model: Any,
    inputs: Any,
    metadata: ModelMetadata,
    foreground_separation: bool | None = None,
    foreground_threshold: ForegroundThreshold = 0.5,
    foreground_side: Literal["high", "low"] = "high",
    rgb_fit_scope: RGBFitScope = "foreground",
    projection: PatchPCAProjection | None = None,
    interpolation: Interpolation = "bilinear",
    guidance_image: Any | None = None,
    output_size: tuple[int, int] | None = None,
    anyup_query_chunk_size: int | None = None,
) -> PatchPCAResult:
    """Extract ViT patch tokens and render their shared PCA projection as RGB."""
    if metadata.patch_size is None:
        raise ValueError("Patch PCA requires a model with a known patch size.")

    image_size = (int(inputs.shape[-2]), int(inputs.shape[-1]))
    patch_grid = infer_patch_grid_from_image(image_size, metadata.patch_size)

    patch_embeddings = extract_patch_embeddings(model, inputs, patch_grid)
    return project_patch_embeddings(
        patch_embeddings,
        patch_grid=patch_grid,
        image_size=output_size or image_size,
        foreground_separation=foreground_separation,
        foreground_threshold=foreground_threshold,
        foreground_side=foreground_side,
        rgb_fit_scope=rgb_fit_scope,
        projection=projection,
        interpolation=interpolation,
        guidance_image=inputs if guidance_image is None else guidance_image,
        anyup_query_chunk_size=anyup_query_chunk_size,
    )


def extract_patch_embeddings(
    model: Any,
    inputs: Any,
    patch_grid: tuple[int, int],
) -> Any:
    with torch.no_grad():
        parameter = next(model.parameters())
        model_inputs = inputs.to(device=parameter.device, dtype=parameter.dtype)
        features = model.forward_features(model_inputs)
    return (
        _patch_tokens_from_features(
            features,
            patch_count=patch_grid[0] * patch_grid[1],
        )
        .detach()
        .float()
        .cpu()
    )


def project_patch_embeddings(
    patch_embeddings: Any,
    patch_grid: tuple[int, int],
    image_size: tuple[int, int],
    foreground_separation: bool | None = None,
    foreground_threshold: ForegroundThreshold = 0.5,
    foreground_side: Literal["high", "low"] = "high",
    rgb_fit_scope: RGBFitScope = "foreground",
    projection: PatchPCAProjection | None = None,
    interpolation: Interpolation = "bilinear",
    guidance_image: Any | None = None,
    anyup_query_chunk_size: int | None = None,
) -> PatchPCAResult:
    """Project a batch of patch embeddings into one shared RGB PCA space."""
    if foreground_separation is not None and not isinstance(
        foreground_separation, bool
    ):
        raise ValueError("foreground_separation must be a boolean or None.")
    _validate_foreground_threshold(foreground_threshold)
    if foreground_side not in {"high", "low"}:
        raise ValueError("foreground_side must be one of: high, low.")
    if rgb_fit_scope not in {"foreground", "all"}:
        raise ValueError("rgb_fit_scope must be one of: foreground, all.")
    choices = {
        "nearest",
        "bilinear",
        "bilinear_mask",
        "anyup",
        "anyup_mask",
        "anyup_soft",
        "anyup_soft_mask",
    }
    if interpolation not in choices:
        raise ValueError(
            "interpolation must be one of: nearest, bilinear, bilinear_mask, "
            "anyup, anyup_mask, anyup_soft, anyup_soft_mask."
        )
    if is_anyup_interpolation(interpolation) and guidance_image is None:
        raise ValueError(
            f"guidance_image is required for interpolation={interpolation!r}."
        )
    if (
        interpolation in {"anyup_soft", "anyup_soft_mask"}
        and anyup_query_chunk_size is None
    ):
        raise ValueError("soft AnyUp interpolation requires anyup_query_chunk_size.")

    embeddings = torch.as_tensor(patch_embeddings).detach().float().cpu()
    if embeddings.ndim != 3:
        raise ValueError("patch_embeddings must have shape (batch, patches, features).")

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
    if projection is None:
        resolved_foreground_separation = (
            True if foreground_separation is None else foreground_separation
        )
        projection, first_component = _fit_projection(
            flattened,
            resolved_foreground_separation,
            foreground_threshold,
            foreground_side,
            rgb_fit_scope,
        )
    else:
        _validate_projection(projection, feature_count)
        resolved_foreground_separation = (
            projection.foreground_separation
            if foreground_separation is None
            else foreground_separation
        )
        first_component = (
            _apply_projection(
                flattened,
                projection.foreground_components,
                projection.foreground_minimum,
                projection.foreground_maximum,
            )
            if resolved_foreground_separation
            else None
        )

    if resolved_foreground_separation:
        assert first_component is not None
        if projection.foreground_side == "high":
            foreground_mask = (
                first_component[:, 0] > projection.foreground_threshold
            )
        else:
            foreground_mask = (
                first_component[:, 0] < projection.foreground_threshold
            )
    else:
        foreground_mask = torch.ones(
            batch_size * patch_count,
            dtype=torch.bool,
        )

    rgb_patches = torch.zeros(
        (batch_size * patch_count, 3),
        dtype=flattened.dtype,
    )
    projected_mask = (
        torch.ones_like(foreground_mask)
        if (
            not resolved_foreground_separation
            or interpolation == "bilinear_mask"
            or is_anyup_interpolation(interpolation)
        )
        else foreground_mask
    )
    projected_embeddings = flattened[projected_mask]
    if projected_embeddings.numel():
        projected_rgb = _apply_projection(
            projected_embeddings,
            projection.rgb_components,
            projection.rgb_minimum,
            projection.rgb_maximum,
        )
        rgb_patches[projected_mask] = projected_rgb[:, :3]

    batched_rgb_patches = rgb_patches.reshape(batch_size, patch_count, 3)
    batched_foreground_mask = foreground_mask.reshape(batch_size, patch_count)
    if is_anyup_interpolation(interpolation):
        images, output_foreground_mask = _render_anyup_pca_images(
            embeddings,
            batched_foreground_mask,
            patch_grid,
            image_size,
            projection,
            resolved_foreground_separation,
            interpolation,
            guidance_image,
            anyup_query_chunk_size,
        )
    else:
        images = render_patch_pca_images(
            batched_rgb_patches,
            batched_foreground_mask,
            patch_grid,
            image_size,
            interpolation,
        )
        output_foreground_mask = batched_foreground_mask

    return PatchPCAResult(
        patch_embeddings=embeddings,
        foreground_mask=output_foreground_mask,
        images=images,
        patch_grid=patch_grid,
        image_size=image_size,
        projection=projection,
        rgb_patches=batched_rgb_patches,
        interpolation=interpolation,
    )


def render_patch_pca_images(
    rgb_patches: Any,
    foreground_mask: Any,
    patch_grid: tuple[int, int],
    image_size: tuple[int, int],
    interpolation: Interpolation,
) -> tuple[Image.Image, ...]:
    """Render projected patch colors, applying masked edges after interpolation."""
    colors = torch.as_tensor(rgb_patches).detach().float().cpu()
    masks = torch.as_tensor(foreground_mask).detach().bool().cpu()
    batch_size, patch_count, channels = colors.shape
    if channels != 3 or patch_count != patch_grid[0] * patch_grid[1]:
        raise ValueError("rgb_patches shape does not match the patch grid.")
    if tuple(masks.shape) != (batch_size, patch_count):
        raise ValueError("foreground_mask shape does not match rgb_patches.")
    if interpolation not in {"nearest", "bilinear", "bilinear_mask"}:
        raise ValueError(
            "interpolation must be one of: nearest, bilinear, bilinear_mask."
        )

    patch_images = colors.reshape(batch_size, patch_grid[0], patch_grid[1], 3).permute(
        0, 3, 1, 2
    )
    mode = "nearest" if interpolation == "nearest" else "bilinear"
    options = {} if mode == "nearest" else {"align_corners": False}
    rendered = functional.interpolate(
        patch_images, size=image_size, mode=mode, **options
    )
    if interpolation == "bilinear_mask":
        patch_masks = masks.reshape(batch_size, 1, *patch_grid).float()
        sharp_mask = functional.interpolate(
            patch_masks, size=image_size, mode="nearest"
        )
        rendered = rendered * sharp_mask

    arrays = rendered.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return tuple(
        Image.fromarray((array * 255).round().astype(np.uint8), mode="RGB")
        for array in arrays
    )


def _render_anyup_pca_images(
    embeddings: Any,
    foreground_mask: Any,
    patch_grid: tuple[int, int],
    image_size: tuple[int, int],
    projection: PatchPCAProjection,
    foreground_separation: bool,
    interpolation: Interpolation,
    guidance_image: Any,
    anyup_query_chunk_size: int | None,
) -> tuple[tuple[Image.Image, ...], Any]:
    batch_size, _patch_count, feature_count = embeddings.shape
    dense_mask = foreground_separation and interpolation in {"anyup", "anyup_soft"}
    soft_attention = interpolation in {"anyup_soft", "anyup_soft_mask"}
    feature_map = embeddings.reshape(
        batch_size,
        patch_grid[0],
        patch_grid[1],
        feature_count,
    ).permute(0, 3, 1, 2)
    if anyup_query_chunk_size is None:
        upsampled = upsample_features(
            guidance_image,
            feature_map,
            image_size,
        )
        flattened = upsampled.permute(0, 2, 3, 1).reshape(-1, feature_count)
        rendered = _apply_projection(
            flattened,
            projection.rgb_components,
            projection.rgb_minimum,
            projection.rgb_maximum,
        )[:, :3]
        if dense_mask:
            dense_first_component = _apply_projection(
                flattened,
                projection.foreground_components,
                projection.foreground_minimum,
                projection.foreground_maximum,
            )[:, 0]
    else:
        value_components = projection.rgb_components
        if dense_mask:
            value_components = torch.cat(
                (value_components, projection.foreground_components),
                dim=1,
            )
        projected_values = (
            (embeddings @ value_components)
            .reshape(
                batch_size,
                patch_grid[0],
                patch_grid[1],
                value_components.shape[1],
            )
            .permute(0, 3, 1, 2)
        )
        upsampled_projection = upsample_values_streaming(
            guidance_image,
            feature_map,
            image_size,
            values=projected_values,
            q_chunk_size=anyup_query_chunk_size,
            attention_mode="soft" if soft_attention else "hard",
        )
        flattened = upsampled_projection.permute(0, 2, 3, 1).reshape(
            -1,
            value_components.shape[1],
        )
        rendered = _normalize_with_bounds(
            flattened[:, :3],
            projection.rgb_minimum,
            projection.rgb_maximum,
        )
        if dense_mask:
            dense_first_component = _normalize_with_bounds(
                flattened[:, 3:4],
                projection.foreground_minimum,
                projection.foreground_maximum,
            )[:, 0]
    rendered = rendered.reshape(batch_size, *image_size, 3).permute(0, 3, 1, 2)

    if not foreground_separation:
        upsampled_mask = torch.ones(
            (batch_size, 1, *image_size),
            dtype=rendered.dtype,
        )
    elif dense_mask:
        if projection.foreground_side == "high":
            dense_foreground = dense_first_component > projection.foreground_threshold
        else:
            dense_foreground = dense_first_component < projection.foreground_threshold
        upsampled_mask = dense_foreground.reshape(
            batch_size,
            1,
            *image_size,
        ).float()
    else:
        patch_masks = foreground_mask.reshape(batch_size, 1, *patch_grid).float()
        upsampled_mask = functional.interpolate(
            patch_masks,
            size=image_size,
            mode="nearest",
        )
    rendered = rendered * upsampled_mask

    arrays = rendered.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    images = tuple(
        Image.fromarray((array * 255).round().astype(np.uint8), mode="RGB")
        for array in arrays
    )
    return images, upsampled_mask[:, 0].bool().cpu()


def fit_patch_pca_projection_batches(
    batch_factory: Callable[[], Iterable[Any]],
    *,
    foreground_separation: bool = True,
    foreground_threshold: ForegroundThreshold = 0.5,
    foreground_side: Literal["high", "low"] = "high",
    rgb_fit_scope: RGBFitScope = "foreground",
    rgb_percentile_bounds: tuple[float, float] | None = None,
) -> PatchPCAProjection:
    """Fit one approximate PCA projection from all embedding batches."""
    if not isinstance(foreground_separation, bool):
        raise ValueError("foreground_separation must be a boolean.")
    _validate_foreground_threshold(foreground_threshold)
    if foreground_side not in {"high", "low"}:
        raise ValueError("foreground_side must be one of: high, low.")
    if rgb_fit_scope not in {"foreground", "all"}:
        raise ValueError("rgb_fit_scope must be one of: foreground, all.")
    _validate_percentile_bounds(rgb_percentile_bounds)

    if foreground_separation:
        foreground_components = _fit_batched_components(batch_factory, components=1)
        foreground_minimum, foreground_maximum = _streaming_projected_bounds(
            batch_factory,
            foreground_components,
        )
        if foreground_threshold == "auto":
            histogram = np.zeros(_OTSU_BINS, dtype=np.int64)
            for embeddings in batch_factory():
                flattened = _flatten_embedding_batch(embeddings)
                normalized = _apply_projection(
                    flattened,
                    foreground_components,
                    foreground_minimum,
                    foreground_maximum,
                )
                histogram += _foreground_histogram(normalized[:, 0])
            foreground_threshold = _otsu_threshold(histogram)
    else:
        feature_count = _batch_feature_count(batch_factory)
        foreground_components = torch.zeros((feature_count, 1), dtype=torch.float32)
        foreground_minimum = torch.zeros(1, dtype=torch.float32)
        foreground_maximum = torch.zeros(1, dtype=torch.float32)
        foreground_threshold = 0.5

    def foreground_batches() -> Iterable[Any]:
        for embeddings in batch_factory():
            flattened = _flatten_embedding_batch(embeddings)
            first_component = _apply_projection(
                flattened,
                foreground_components,
                foreground_minimum,
                foreground_maximum,
            )
            if foreground_side == "high":
                mask = first_component[:, 0] > foreground_threshold
            else:
                mask = first_component[:, 0] < foreground_threshold
            if mask.any():
                yield flattened[mask]

    resolved_rgb_fit_scope: RGBFitScope = (
        "all" if not foreground_separation else rgb_fit_scope
    )
    rgb_batches = (
        batch_factory if resolved_rgb_fit_scope == "all" else foreground_batches
    )
    rgb_components = _fit_batched_components(
        rgb_batches,
        components=3,
        allow_empty=True,
        feature_count=int(foreground_components.shape[0]),
    )
    if rgb_components.shape[1] < 3:
        rgb_components = functional.pad(
            rgb_components,
            (0, 3 - rgb_components.shape[1]),
        )
    if rgb_components.numel():
        if rgb_percentile_bounds is None:
            rgb_minimum, rgb_maximum = _streaming_projected_bounds(
                rgb_batches,
                rgb_components,
                allow_empty=True,
            )
        else:
            rgb_minimum, rgb_maximum = _projected_percentile_bounds(
                rgb_batches,
                rgb_components,
                rgb_percentile_bounds,
                allow_empty=True,
            )
    else:
        rgb_minimum = torch.zeros(3)
        rgb_maximum = torch.zeros(3)

    return PatchPCAProjection(
        foreground_components=foreground_components,
        foreground_minimum=foreground_minimum,
        foreground_maximum=foreground_maximum,
        rgb_components=rgb_components,
        rgb_minimum=rgb_minimum,
        rgb_maximum=rgb_maximum,
        foreground_threshold=foreground_threshold,
        foreground_side=foreground_side,
        rgb_fit_scope=resolved_rgb_fit_scope,
        foreground_separation=foreground_separation,
    )


def save_patch_pca_projection(
    projection: PatchPCAProjection,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        np.savez_compressed(
            file,
            foreground_components=_as_numpy(projection.foreground_components),
            foreground_minimum=_as_numpy(projection.foreground_minimum),
            foreground_maximum=_as_numpy(projection.foreground_maximum),
            rgb_components=_as_numpy(projection.rgb_components),
            rgb_minimum=_as_numpy(projection.rgb_minimum),
            rgb_maximum=_as_numpy(projection.rgb_maximum),
            foreground_threshold=projection.foreground_threshold,
            foreground_side=projection.foreground_side,
            rgb_fit_scope=projection.rgb_fit_scope,
            foreground_separation=projection.foreground_separation,
        )
    return output


def load_patch_pca_projection(path: str | Path) -> PatchPCAProjection:
    with np.load(Path(path), allow_pickle=False) as values:
        required = {
            "foreground_components",
            "foreground_minimum",
            "foreground_maximum",
            "rgb_components",
            "rgb_minimum",
            "rgb_maximum",
            "foreground_threshold",
            "foreground_side",
            "rgb_fit_scope",
            "foreground_separation",
        }
        missing = sorted(required - set(values.files))
        if missing:
            raise ValueError(
                "PCA projection is missing required value(s): "
                f"{', '.join(missing)}."
            )
        return PatchPCAProjection(
            foreground_components=torch.from_numpy(values["foreground_components"]),
            foreground_minimum=torch.from_numpy(values["foreground_minimum"]),
            foreground_maximum=torch.from_numpy(values["foreground_maximum"]),
            rgb_components=torch.from_numpy(values["rgb_components"]),
            rgb_minimum=torch.from_numpy(values["rgb_minimum"]),
            rgb_maximum=torch.from_numpy(values["rgb_maximum"]),
            foreground_threshold=float(values["foreground_threshold"]),
            foreground_side=str(values["foreground_side"]),
            rgb_fit_scope=str(values["rgb_fit_scope"]),
            foreground_separation=bool(values["foreground_separation"]),
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


def _fit_projection(
    values: Any,
    foreground_separation: bool,
    foreground_threshold: ForegroundThreshold,
    foreground_side: Literal["high", "low"],
    rgb_fit_scope: RGBFitScope,
) -> tuple[PatchPCAProjection, Any]:
    if foreground_separation:
        first_projected, first_components = _fit_pca_projection(values, components=1)
        first_minimum, first_maximum = _value_bounds(first_projected)
        normalized_first = _normalize_with_bounds(
            first_projected,
            first_minimum,
            first_maximum,
        )
        if foreground_threshold == "auto":
            foreground_threshold = _otsu_threshold(
                _foreground_histogram(normalized_first[:, 0])
            )
        if foreground_side == "high":
            foreground_mask = normalized_first[:, 0] > foreground_threshold
        else:
            foreground_mask = normalized_first[:, 0] < foreground_threshold
    else:
        first_components = values.new_zeros((values.shape[1], 1))
        first_minimum = values.new_zeros(1)
        first_maximum = values.new_zeros(1)
        normalized_first = values.new_zeros((values.shape[0], 1))
        foreground_mask = torch.ones(values.shape[0], dtype=torch.bool)
        foreground_threshold = 0.5

    resolved_rgb_fit_scope: RGBFitScope = (
        "all" if not foreground_separation else rgb_fit_scope
    )
    rgb_values = (
        values if resolved_rgb_fit_scope == "all" else values[foreground_mask]
    )
    if rgb_values.numel():
        rgb_projected, rgb_components = _fit_pca_projection(rgb_values, components=3)
        rgb_minimum, rgb_maximum = _value_bounds(rgb_projected)
    else:
        rgb_components = values.new_zeros((values.shape[1], 3))
        rgb_minimum = values.new_zeros(3)
        rgb_maximum = values.new_zeros(3)

    if rgb_components.shape[1] < 3:
        missing = 3 - rgb_components.shape[1]
        rgb_components = functional.pad(rgb_components, (0, missing))
        rgb_minimum = functional.pad(rgb_minimum, (0, missing))
        rgb_maximum = functional.pad(rgb_maximum, (0, missing))

    projection = PatchPCAProjection(
        foreground_components=first_components,
        foreground_minimum=first_minimum,
        foreground_maximum=first_maximum,
        rgb_components=rgb_components,
        rgb_minimum=rgb_minimum,
        rgb_maximum=rgb_maximum,
        foreground_threshold=foreground_threshold,
        foreground_side=foreground_side,
        rgb_fit_scope=resolved_rgb_fit_scope,
        foreground_separation=foreground_separation,
    )
    return projection, normalized_first


def _validate_foreground_threshold(value: ForegroundThreshold) -> None:
    if value == "auto":
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0 <= value <= 1
    ):
        raise ValueError("foreground_threshold must be between 0 and 1 or 'auto'.")


def _foreground_histogram(values: Any) -> Any:
    histogram, _ = np.histogram(_as_numpy(values), bins=_OTSU_BINS, range=(0.0, 1.0))
    return histogram


def _otsu_threshold(histogram: Any) -> float:
    occupied = np.flatnonzero(histogram)
    if len(occupied) < 2:
        return 0.5

    weights = histogram.astype(np.float64)
    counts = np.cumsum(weights)[:-1]
    weighted = np.cumsum(weights * np.arange(_OTSU_BINS))[:-1]
    total = weights.sum()
    total_weighted = np.dot(weights, np.arange(_OTSU_BINS))
    valid = (counts > 0) & (counts < total)
    between = np.zeros(_OTSU_BINS - 1)
    between[valid] = (total_weighted * counts[valid] - total * weighted[valid]) ** 2 / (
        counts[valid] * (total - counts[valid])
    )
    best = np.flatnonzero(np.isclose(between, between.max(), rtol=1e-12, atol=0))
    return float((best[0] + best[-1] + 2) / (2 * _OTSU_BINS))


def _fit_pca_projection(values: Any, components: int) -> tuple[Any, Any]:
    component_count = min(components, int(values.shape[0]), int(values.shape[1]))
    if component_count == 0:
        return (
            values.new_zeros((values.shape[0], 0)),
            values.new_zeros((values.shape[1], 0)),
        )
    if values.shape[0] == 1:
        components_matrix = values.new_zeros((values.shape[1], component_count))
        indices = torch.arange(component_count)
        components_matrix[indices, indices] = 1
        return values[:, :component_count], components_matrix

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        _u, _s, vectors = torch.pca_lowrank(
            values,
            q=component_count,
            center=True,
        )
    components_matrix = vectors[:, :component_count]
    return values @ components_matrix, components_matrix


def _fit_batched_components(
    batch_factory: Callable[[], Iterable[Any]],
    *,
    components: int,
    allow_empty: bool = False,
    feature_count: int | None = None,
) -> Any:
    batches = []
    for batch in batch_factory():
        values = _flatten_embedding_batch(batch)
        if feature_count is None:
            feature_count = int(values.shape[1])
        elif int(values.shape[1]) != feature_count:
            raise ValueError("PCA embedding batches have inconsistent feature counts.")
        batches.append(values)

    if not batches or feature_count is None or not any(len(batch) for batch in batches):
        if allow_empty and feature_count is not None:
            return torch.zeros((feature_count, components), dtype=torch.float32)
        raise ValueError("Cannot fit PCA from empty embedding batches.")
    values = torch.cat(batches)
    batches.clear()
    _, result = _fit_pca_projection(values, components)
    return result


def _streaming_projected_bounds(
    batch_factory: Callable[[], Iterable[Any]],
    components: Any,
    *,
    allow_empty: bool = False,
) -> tuple[Any, Any]:
    minimum = None
    maximum = None
    for batch in batch_factory():
        projected = _flatten_embedding_batch(batch) @ components
        batch_minimum, batch_maximum = _value_bounds(projected)
        minimum = (
            batch_minimum if minimum is None else torch.minimum(minimum, batch_minimum)
        )
        maximum = (
            batch_maximum if maximum is None else torch.maximum(maximum, batch_maximum)
        )
    if minimum is None or maximum is None:
        if allow_empty:
            zeros = torch.zeros(int(components.shape[1]), dtype=torch.float32)
            return zeros, zeros.clone()
        raise ValueError("Cannot calculate PCA bounds from empty embedding batches.")
    return minimum, maximum


def _projected_percentile_bounds(
    batch_factory: Callable[[], Iterable[Any]],
    components: Any,
    percentiles: tuple[float, float],
    *,
    allow_empty: bool = False,
) -> tuple[Any, Any]:
    projected_batches = []
    for batch in batch_factory():
        projected = _flatten_embedding_batch(batch) @ components
        if len(projected):
            projected_batches.append(projected)
    if not projected_batches:
        if allow_empty:
            zeros = torch.zeros(int(components.shape[1]), dtype=torch.float32)
            return zeros, zeros.clone()
        raise ValueError("Cannot calculate PCA bounds from empty embedding batches.")
    projected = torch.cat(projected_batches)
    lower, upper = percentiles
    return (
        torch.quantile(projected, lower, dim=0),
        torch.quantile(projected, upper, dim=0),
    )


def _batch_feature_count(batch_factory: Callable[[], Iterable[Any]]) -> int:
    feature_count = None
    found_values = False
    for batch in batch_factory():
        values = _flatten_embedding_batch(batch)
        found_values = found_values or len(values) > 0
        if feature_count is None:
            feature_count = int(values.shape[1])
        elif int(values.shape[1]) != feature_count:
            raise ValueError("PCA embedding batches have inconsistent feature counts.")
    if feature_count is None or not found_values:
        raise ValueError("Cannot fit PCA from empty embedding batches.")
    return feature_count


def _flatten_embedding_batch(values: Any) -> Any:
    tensor = torch.as_tensor(values).detach().float().cpu()
    if tensor.ndim == 3:
        return tensor.reshape(-1, tensor.shape[-1])
    if tensor.ndim == 2:
        return tensor
    raise ValueError("PCA embedding batches must have two or three dimensions.")


def _validate_percentile_bounds(value: tuple[float, float] | None) -> None:
    if value is None:
        return
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int | float)
            for item in value
        )
        or not 0 <= value[0] < value[1] <= 1
    ):
        raise ValueError(
            "rgb_percentile_bounds must contain two increasing values between 0 and 1."
        )


def _minmax_normalize(values: Any) -> Any:
    minimum, maximum = _value_bounds(values)
    return _normalize_with_bounds(values, minimum, maximum)


def _value_bounds(values: Any) -> tuple[Any, Any]:
    return values.min(dim=0).values, values.max(dim=0).values


def _normalize_with_bounds(values: Any, minimum: Any, maximum: Any) -> Any:
    span = maximum - minimum
    safe_span = torch.where(span > 0, span, torch.ones_like(span))
    normalized = (values - minimum) / safe_span
    normalized = torch.where(span > 0, normalized, torch.zeros_like(normalized))
    return normalized.clamp(0, 1)


def _apply_projection(
    values: Any,
    components: Any,
    minimum: Any,
    maximum: Any,
) -> Any:
    return _normalize_with_bounds(values @ components, minimum, maximum)


def _validate_projection(projection: PatchPCAProjection, feature_count: int) -> None:
    if tuple(projection.foreground_components.shape) != (feature_count, 1):
        raise ValueError(
            "PCA projection feature count does not match model patch embeddings."
        )
    if tuple(projection.rgb_components.shape) != (feature_count, 3):
        raise ValueError(
            "PCA RGB projection feature count does not match model patch embeddings."
        )


def _as_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
