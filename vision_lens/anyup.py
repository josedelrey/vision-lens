from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from typing import Any

import torch

ANYUP_REPOSITORY = "wimmerth/anyup:checkpoint_v2"
ANYUP_MODEL = "anyup_multi_backbone"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

__all__ = [
    "ANYUP_MODEL",
    "ANYUP_REPOSITORY",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "is_anyup_interpolation",
    "load_anyup",
    "prepare_anyup_image",
    "upsample_features",
]


def is_anyup_interpolation(interpolation: str) -> bool:
    """Return whether an interpolation mode uses AnyUp."""
    return interpolation in {"anyup", "anyup_mask"}


def prepare_anyup_image(
    image: Any,
    *,
    source_mean: Iterable[float] | None = None,
    source_std: Iterable[float] | None = None,
) -> Any:
    """Convert a BCHW image tensor to the ImageNet normalization AnyUp expects.

    Pass the normalization used to create ``image`` as ``source_mean`` and
    ``source_std``. Leave both unset when the input tensor contains RGB values in
    the 0-1 range.
    """
    tensor = torch.as_tensor(image)
    _validate_image(tensor)
    if (source_mean is None) != (source_std is None):
        raise ValueError("source_mean and source_std must be provided together.")

    if source_mean is not None and source_std is not None:
        source_mean_tensor = _channel_values(
            source_mean,
            "source_mean",
            tensor,
        )
        source_std_tensor = _channel_values(
            source_std,
            "source_std",
            tensor,
        )
        tensor = tensor * source_std_tensor + source_mean_tensor

    target_mean = _channel_values(IMAGENET_MEAN, "IMAGENET_MEAN", tensor)
    target_std = _channel_values(IMAGENET_STD, "IMAGENET_STD", tensor)
    return (tensor - target_mean) / target_std


@cache
def load_anyup(device: str = "cpu") -> Any:
    """Load and cache the official multi-backbone AnyUp model for one device."""
    try:
        model = torch.hub.load(
            ANYUP_REPOSITORY,
            ANYUP_MODEL,
            pretrained=True,
            use_natten=False,
            device=device,
            trust_repo=True,
        )
    except Exception as error:
        raise RuntimeError(
            "Could not load AnyUp. The first use requires network access to "
            f"download {ANYUP_REPOSITORY}/{ANYUP_MODEL} and its checkpoint."
        ) from error
    return model.eval()


def upsample_features(
    image: Any,
    features: Any,
    output_size: tuple[int, int],
    *,
    model: Any | None = None,
    q_chunk_size: int | None = None,
) -> Any:
    """Upsample BCHW features with AnyUp using an ImageNet-normalized image."""
    image_tensor = torch.as_tensor(image)
    feature_tensor = torch.as_tensor(features)
    _validate_image(image_tensor)
    _validate_features(feature_tensor)
    if image_tensor.shape[0] != feature_tensor.shape[0]:
        raise ValueError("image and features must have the same batch size.")
    if (
        not isinstance(output_size, tuple)
        or len(output_size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in output_size
        )
    ):
        raise ValueError("output_size must contain two positive integers.")
    if q_chunk_size is not None and (
        isinstance(q_chunk_size, bool)
        or not isinstance(q_chunk_size, int)
        or q_chunk_size <= 0
    ):
        raise ValueError("q_chunk_size must be a positive integer or None.")

    original_device = feature_tensor.device
    original_dtype = feature_tensor.dtype
    execution_device = image_tensor.device
    upsampler = model if model is not None else load_anyup(str(execution_device))
    parameter = next(iter(upsampler.parameters()), None)
    if parameter is not None:
        execution_device = parameter.device
        execution_dtype = parameter.dtype
    else:
        execution_dtype = torch.float32

    prepared_image = image_tensor.to(
        device=execution_device,
        dtype=execution_dtype,
    )
    prepared_features = feature_tensor.to(
        device=execution_device,
        dtype=execution_dtype,
    )
    with torch.inference_mode():
        upsampled = upsampler(
            prepared_image,
            prepared_features,
            output_size=output_size,
            q_chunk_size=q_chunk_size,
        )
    expected_shape = (
        feature_tensor.shape[0],
        feature_tensor.shape[1],
        *output_size,
    )
    if tuple(upsampled.shape) != expected_shape:
        raise RuntimeError(
            "AnyUp returned an unexpected shape: "
            f"expected {expected_shape}, got {tuple(upsampled.shape)}."
        )
    return upsampled.detach().to(device=original_device, dtype=original_dtype)


def _validate_image(image: Any) -> None:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must have shape (batch, 3, height, width).")
    if not image.is_floating_point():
        raise ValueError("image must be a floating-point tensor.")


def _validate_features(features: Any) -> None:
    if features.ndim != 4:
        raise ValueError("features must have shape (batch, channels, height, width).")
    if not features.is_floating_point():
        raise ValueError("features must be a floating-point tensor.")


def _channel_values(values: Iterable[float], name: str, tensor: Any) -> Any:
    resolved = tuple(values)
    if len(resolved) != 3:
        raise ValueError(f"{name} must contain three values.")
    return torch.tensor(resolved, dtype=tensor.dtype, device=tensor.device)[
        None, :, None, None
    ]
