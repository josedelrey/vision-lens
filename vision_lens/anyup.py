from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from typing import Any

import torch
import torch.nn.functional as functional

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
    "upsample_values_streaming",
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
    if q_chunk_size is not None:
        return upsample_values_streaming(
            image_tensor,
            feature_tensor,
            output_size,
            model=upsampler,
            q_chunk_size=q_chunk_size,
        )
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


def upsample_values_streaming(
    image: Any,
    features: Any,
    output_size: tuple[int, int],
    *,
    values: Any | None = None,
    model: Any | None = None,
    q_chunk_size: int,
) -> Any:
    """Run AnyUp in row chunks and aggregate a compact value tensor.

    ``features`` still determines AnyUp's keys. ``values`` may use fewer channels
    when a linear projection can be applied before attention. Query features and
    locality masks are generated per chunk, and completed chunks are transferred
    to the value tensor's original device immediately.
    """
    image_tensor = torch.as_tensor(image)
    feature_tensor = torch.as_tensor(features)
    value_tensor = feature_tensor if values is None else torch.as_tensor(values)
    _validate_image(image_tensor)
    _validate_features(feature_tensor)
    _validate_features(value_tensor)
    _validate_output_size(output_size)
    _validate_query_chunk_size(q_chunk_size)
    if image_tensor.shape[0] != feature_tensor.shape[0]:
        raise ValueError("image and features must have the same batch size.")
    if value_tensor.shape[0] != feature_tensor.shape[0]:
        raise ValueError("values and features must have the same batch size.")
    if value_tensor.shape[-2:] != feature_tensor.shape[-2:]:
        raise ValueError("values and features must have the same spatial size.")

    original_device = value_tensor.device
    original_dtype = value_tensor.dtype
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
    prepared_values = value_tensor.to(
        device=execution_device,
        dtype=execution_dtype,
    )
    with torch.inference_mode():
        upsampled = _stream_anyup_values(
            upsampler,
            prepared_image,
            prepared_features,
            prepared_values,
            output_size,
            q_chunk_size,
            output_device=original_device,
            output_dtype=original_dtype,
        )
    expected_shape = (
        value_tensor.shape[0],
        value_tensor.shape[1],
        *output_size,
    )
    if tuple(upsampled.shape) != expected_shape:
        raise RuntimeError(
            "Streaming AnyUp returned an unexpected shape: "
            f"expected {expected_shape}, got {tuple(upsampled.shape)}."
        )
    return upsampled


def _stream_anyup_values(
    upsampler: Any,
    image: Any,
    features: Any,
    values: Any,
    output_size: tuple[int, int],
    q_chunk_size: int,
    *,
    output_device: Any,
    output_dtype: Any,
) -> Any:
    try:
        decoder = upsampler.cross_decode
        cross_attention = decoder.cross_attn
        query_convolution = decoder.conv2d
    except AttributeError as error:
        raise RuntimeError(
            "Streaming AnyUp requires the pinned standard cross-attention model."
        ) from error

    encoded = upsampler.image_encoder(image)
    encoded_height, encoded_width = encoded.shape[-2:]
    coordinates = _coordinates(
        encoded_height,
        encoded_width,
        device=encoded.device,
        dtype=encoded.dtype,
    )
    encoded = encoded.permute(0, 2, 3, 1).reshape(
        encoded.shape[0],
        encoded_height * encoded_width,
        encoded.shape[1],
    )
    encoded = upsampler.rope(encoded, coordinates)
    encoded = encoded.reshape(
        encoded.shape[0],
        encoded_height,
        encoded_width,
        encoded.shape[-1],
    ).permute(0, 3, 1, 2)

    query_source = upsampler.query_encoder(encoded)
    feature_height, feature_width = features.shape[-2:]
    keys = functional.adaptive_avg_pool2d(
        upsampler.key_encoder(encoded),
        output_size=(feature_height, feature_width),
    )
    feature_keys = upsampler.key_features_encoder(functional.normalize(features, dim=1))
    keys = upsampler.aggregation(torch.cat((keys, feature_keys), dim=1))
    key_sequence = keys.permute(0, 2, 3, 1).reshape(
        keys.shape[0],
        feature_height * feature_width,
        keys.shape[1],
    )
    normalized_keys = cross_attention.norm_k(key_sequence)
    value_sequence = values.permute(0, 2, 3, 1).reshape(
        values.shape[0],
        feature_height * feature_width,
        values.shape[1],
    )

    output_height, output_width = output_size
    rows_per_chunk = max(1, q_chunk_size // output_width)
    chunks = []
    for row_start in range(0, output_height, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, output_height)
        halo_start = max(0, row_start - 1)
        halo_end = min(output_height, row_end + 1)
        pooled_queries = _adaptive_pool_rows(
            query_source,
            output_size,
            halo_start,
            halo_end,
        )
        convolved_queries = query_convolution(pooled_queries)
        chunk_offset = row_start - halo_start
        convolved_queries = convolved_queries[
            :,
            :,
            chunk_offset : chunk_offset + row_end - row_start,
            :,
        ]
        query_sequence = convolved_queries.permute(0, 2, 3, 1).reshape(
            convolved_queries.shape[0],
            (row_end - row_start) * output_width,
            convolved_queries.shape[1],
        )
        normalized_queries = cross_attention.norm_q(query_sequence)
        attention_mask = _attention_mask_rows(
            output_size,
            (feature_height, feature_width),
            row_start,
            row_end,
            float(decoder.window_ratio),
            device=normalized_queries.device,
        )
        _ignored, attention = cross_attention.attention(
            normalized_queries,
            normalized_keys,
            key_sequence,
            average_attn_weights=True,
            attn_mask=attention_mask,
        )
        output = torch.einsum("bij,bjd->bid", attention, value_sequence)
        output = output.reshape(
            output.shape[0],
            row_end - row_start,
            output_width,
            output.shape[-1],
        ).permute(0, 3, 1, 2)
        chunks.append(output.to(device=output_device, dtype=output_dtype))
    return torch.cat(chunks, dim=2)


def _adaptive_pool_rows(
    source: Any,
    output_size: tuple[int, int],
    row_start: int,
    row_end: int,
) -> Any:
    input_height = source.shape[-2]
    output_height, output_width = output_size
    rows = []
    for output_row in range(row_start, row_end):
        input_start = output_row * input_height // output_height
        input_end = -(-(output_row + 1) * input_height // output_height)
        pooled_row = source[:, :, input_start:input_end, :].mean(
            dim=2,
            keepdim=True,
        )
        rows.append(
            functional.adaptive_avg_pool2d(
                pooled_row,
                output_size=(1, output_width),
            )
        )
    return torch.cat(rows, dim=2)


def _attention_mask_rows(
    output_size: tuple[int, int],
    feature_size: tuple[int, int],
    row_start: int,
    row_end: int,
    window_ratio: float,
    *,
    device: Any,
) -> Any | None:
    if window_ratio <= 0:
        return None
    output_height, output_width = output_size
    feature_height, feature_width = feature_size
    output_rows = torch.arange(row_start, row_end, device=device)
    output_columns = torch.arange(output_width, device=device)
    row_positions = (output_rows.float() + 0.5) / output_height
    column_positions = (output_columns.float() + 0.5) / output_width
    row_positions, column_positions = torch.meshgrid(
        row_positions,
        column_positions,
        indexing="ij",
    )
    row_positions = row_positions.reshape(-1, 1)
    column_positions = column_positions.reshape(-1, 1)
    row_minimum = ((row_positions - window_ratio).clamp(0, 1) * feature_height).floor()
    row_maximum = ((row_positions + window_ratio).clamp(0, 1) * feature_height).ceil()
    column_minimum = (
        (column_positions - window_ratio).clamp(0, 1) * feature_width
    ).floor()
    column_maximum = (
        (column_positions + window_ratio).clamp(0, 1) * feature_width
    ).ceil()
    feature_rows = torch.arange(feature_height, device=device)
    feature_columns = torch.arange(feature_width, device=device)
    row_allowed = (feature_rows >= row_minimum) & (feature_rows < row_maximum)
    column_allowed = (feature_columns >= column_minimum) & (
        feature_columns < column_maximum
    )
    allowed = row_allowed.unsqueeze(2) & column_allowed.unsqueeze(1)
    return ~allowed.reshape(-1, feature_height * feature_width)


def _coordinates(
    height: int,
    width: int,
    *,
    device: Any,
    dtype: Any,
) -> Any:
    rows = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    columns = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    rows, columns = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack((rows, columns), dim=-1).reshape(1, height * width, 2)


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


def _validate_output_size(output_size: tuple[int, int]) -> None:
    if (
        not isinstance(output_size, tuple)
        or len(output_size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in output_size
        )
    ):
        raise ValueError("output_size must contain two positive integers.")


def _validate_query_chunk_size(q_chunk_size: int | None) -> None:
    if q_chunk_size is not None and (
        isinstance(q_chunk_size, bool)
        or not isinstance(q_chunk_size, int)
        or q_chunk_size <= 0
    ):
        raise ValueError("q_chunk_size must be a positive integer or None.")


def _channel_values(values: Iterable[float], name: str, tensor: Any) -> Any:
    resolved = tuple(values)
    if len(resolved) != 3:
        raise ValueError(f"{name} must contain three values.")
    return torch.tensor(resolved, dtype=tensor.dtype, device=tensor.device)[
        None, :, None, None
    ]
