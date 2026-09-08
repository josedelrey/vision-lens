from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
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
    projection: PatchPCAProjection | None = None


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


def extract_patch_pca(
    model: Any,
    inputs: Any,
    metadata: ModelMetadata,
    foreground_threshold: float = 0.5,
    foreground_side: Literal["high", "low"] = "high",
    projection: PatchPCAProjection | None = None,
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
        image_size=image_size,
        foreground_threshold=foreground_threshold,
        foreground_side=foreground_side,
        projection=projection,
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
    foreground_threshold: float = 0.5,
    foreground_side: Literal["high", "low"] = "high",
    projection: PatchPCAProjection | None = None,
) -> PatchPCAResult:
    """Project a batch of patch embeddings into one shared RGB PCA space."""
    if not 0 <= foreground_threshold <= 1:
        raise ValueError("foreground_threshold must be between 0 and 1.")
    if foreground_side not in {"high", "low"}:
        raise ValueError("foreground_side must be one of: high, low.")

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
        projection, first_component = _fit_projection(
            flattened,
            foreground_threshold,
            foreground_side,
        )
    else:
        _validate_projection(projection, feature_count)
        first_component = _apply_projection(
            flattened,
            projection.foreground_components,
            projection.foreground_minimum,
            projection.foreground_maximum,
        )
        foreground_threshold = projection.foreground_threshold
        foreground_side = projection.foreground_side

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
        foreground_rgb = _apply_projection(
            foreground_embeddings,
            projection.rgb_components,
            projection.rgb_minimum,
            projection.rgb_maximum,
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
        projection=projection,
    )


def fit_patch_pca_projection_batches(
    batch_factory: Callable[[], Iterable[Any]],
    *,
    foreground_threshold: float = 0.5,
    foreground_side: Literal["high", "low"] = "high",
) -> PatchPCAProjection:
    """Fit one PCA projection without retaining every embedding batch in memory."""
    if not 0 <= foreground_threshold <= 1:
        raise ValueError("foreground_threshold must be between 0 and 1.")
    if foreground_side not in {"high", "low"}:
        raise ValueError("foreground_side must be one of: high, low.")

    foreground_components = _fit_streaming_components(batch_factory, components=1)
    foreground_minimum, foreground_maximum = _streaming_projected_bounds(
        batch_factory,
        foreground_components,
    )

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

    rgb_components = _fit_streaming_components(
        foreground_batches,
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
        rgb_minimum, rgb_maximum = _streaming_projected_bounds(
            foreground_batches,
            rgb_components,
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
        )
    return output


def load_patch_pca_projection(path: str | Path) -> PatchPCAProjection:
    with np.load(Path(path), allow_pickle=False) as values:
        return PatchPCAProjection(
            foreground_components=torch.from_numpy(values["foreground_components"]),
            foreground_minimum=torch.from_numpy(values["foreground_minimum"]),
            foreground_maximum=torch.from_numpy(values["foreground_maximum"]),
            rgb_components=torch.from_numpy(values["rgb_components"]),
            rgb_minimum=torch.from_numpy(values["rgb_minimum"]),
            rgb_maximum=torch.from_numpy(values["rgb_maximum"]),
            foreground_threshold=float(values["foreground_threshold"]),
            foreground_side=str(values["foreground_side"]),
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
    foreground_threshold: float,
    foreground_side: Literal["high", "low"],
) -> tuple[PatchPCAProjection, Any]:
    first_projected, first_components = _fit_pca_projection(values, components=1)
    first_minimum, first_maximum = _value_bounds(first_projected)
    normalized_first = _normalize_with_bounds(
        first_projected,
        first_minimum,
        first_maximum,
    )
    if foreground_side == "high":
        foreground_mask = normalized_first[:, 0] > foreground_threshold
    else:
        foreground_mask = normalized_first[:, 0] < foreground_threshold

    foreground = values[foreground_mask]
    if foreground.numel():
        rgb_projected, rgb_components = _fit_pca_projection(foreground, components=3)
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
    )
    return projection, normalized_first


def _fit_pca_projection(values: Any, components: int) -> tuple[Any, Any]:
    components_matrix = _fit_streaming_components(
        lambda: (values,),
        components=components,
    )
    return values @ components_matrix, components_matrix


def _canonicalize_component_signs(components: Any) -> Any:
    result = components.clone()
    for index in range(int(result.shape[1])):
        column = result[:, index]
        pivot = int(column.abs().argmax())
        if column[pivot] < 0:
            result[:, index] = -column
    return result


def _fit_streaming_components(
    batch_factory: Callable[[], Iterable[Any]],
    *,
    components: int,
    allow_empty: bool = False,
    feature_count: int | None = None,
) -> Any:
    count = 0
    value_sum = None
    for batch in batch_factory():
        values = _flatten_embedding_batch(batch).double()
        if feature_count is None:
            feature_count = int(values.shape[1])
        elif int(values.shape[1]) != feature_count:
            raise ValueError("PCA embedding batches have inconsistent feature counts.")
        count += int(values.shape[0])
        batch_sum = values.sum(dim=0)
        value_sum = batch_sum if value_sum is None else value_sum + batch_sum

    if count == 0 or feature_count is None:
        if allow_empty and feature_count is not None:
            return torch.zeros((feature_count, components), dtype=torch.float32)
        raise ValueError("Cannot fit PCA from empty embedding batches.")

    component_count = min(components, count, feature_count)
    if count == 1:
        return torch.zeros((feature_count, component_count), dtype=torch.float32)

    component_count = min(component_count, count - 1)

    mean = value_sum / count
    covariance = torch.zeros((feature_count, feature_count), dtype=torch.float64)
    for batch in batch_factory():
        centered = _flatten_embedding_batch(batch).double() - mean
        covariance += centered.transpose(0, 1) @ centered

    _eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    result = eigenvectors[:, -component_count:].flip(dims=(1,))
    return _canonicalize_component_signs(result).float()


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


def _flatten_embedding_batch(values: Any) -> Any:
    tensor = torch.as_tensor(values).detach().float().cpu()
    if tensor.ndim == 3:
        return tensor.reshape(-1, tensor.shape[-1])
    if tensor.ndim == 2:
        return tensor
    raise ValueError("PCA embedding batches must have two or three dimensions.")


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
