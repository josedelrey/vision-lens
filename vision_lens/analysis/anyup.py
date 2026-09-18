from __future__ import annotations

import math
from collections.abc import Iterable
from functools import cache
from typing import Any, Literal

import torch
import torch.nn.functional as functional

ANYUP_REPOSITORY = "wimmerth/anyup:1551eaa16b61600de78093c510005ae59ece8866"
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
    return interpolation in {
        "anyup",
        "anyup_mask",
        "anyup_soft",
        "anyup_soft_mask",
    }


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
    attention_mode: Literal["hard", "soft"] = "hard",
) -> Any:
    """Upsample BCHW features with AnyUp using an ImageNet-normalized image."""
    image_tensor = torch.as_tensor(image)
    feature_tensor = torch.as_tensor(features)
    _validate_image(image_tensor)
    _validate_features(feature_tensor)
    _validate_attention_mode(attention_mode)
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
    if attention_mode == "soft" and q_chunk_size is None:
        raise ValueError("soft AnyUp attention requires q_chunk_size.")
    if q_chunk_size is not None:
        return upsample_values_streaming(
            image_tensor,
            feature_tensor,
            output_size,
            model=upsampler,
            q_chunk_size=q_chunk_size,
            attention_mode=attention_mode,
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
    attention_mode: Literal["hard", "soft"] = "hard",
) -> Any:
    """Run AnyUp in sparse 2D tiles and aggregate a compact value tensor.

    ``features`` still determines AnyUp's keys. ``values`` may use fewer channels
    when a linear projection can be applied before attention. Query features and
    locality masks are generated per tile, and only keys within each tile's local
    neighborhood participate in attention. Completed tiles are transferred to the
    value tensor's original device immediately. ``attention_mode="soft"`` replaces
    AnyUp's hard spatial cutoff with a cosine-tapered local bias.
    """
    image_tensor = torch.as_tensor(image)
    feature_tensor = torch.as_tensor(features)
    value_tensor = feature_tensor if values is None else torch.as_tensor(values)
    _validate_image(image_tensor)
    _validate_features(feature_tensor)
    _validate_features(value_tensor)
    _validate_output_size(output_size)
    _validate_query_chunk_size(q_chunk_size)
    _validate_attention_mode(attention_mode)
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
            attention_mode,
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
    attention_mode: Literal["hard", "soft"],
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
    normalized_key_grid = cross_attention.norm_k(key_sequence).reshape(
        keys.shape[0],
        feature_height,
        feature_width,
        keys.shape[1],
    )
    value_grid = values.permute(0, 2, 3, 1)

    output_height, output_width = output_size
    tile_height, tile_width = _query_tile_size(output_size, q_chunk_size)
    upsampled = torch.empty(
        (values.shape[0], values.shape[1], output_height, output_width),
        device=output_device,
        dtype=output_dtype,
    )
    for row_start in range(0, output_height, tile_height):
        row_end = min(row_start + tile_height, output_height)
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
        for column_start in range(0, output_width, tile_width):
            column_end = min(column_start + tile_width, output_width)
            query_tile = convolved_queries[:, :, :, column_start:column_end]
            query_sequence = query_tile.permute(0, 2, 3, 1).reshape(
                query_tile.shape[0],
                (row_end - row_start) * (column_end - column_start),
                query_tile.shape[1],
            )
            normalized_queries = cross_attention.norm_q(query_sequence)
            key_bounds = _local_key_bounds(
                output_size,
                (feature_height, feature_width),
                row_start,
                row_end,
                column_start,
                column_end,
                float(decoder.window_ratio),
            )
            key_row_start, key_row_end, key_column_start, key_column_end = key_bounds
            local_keys = normalized_key_grid[
                :,
                key_row_start:key_row_end,
                key_column_start:key_column_end,
                :,
            ].reshape(normalized_key_grid.shape[0], -1, normalized_key_grid.shape[-1])
            local_values = value_grid[
                :,
                key_row_start:key_row_end,
                key_column_start:key_column_end,
                :,
            ].reshape(value_grid.shape[0], -1, value_grid.shape[-1])
            mask_arguments = (
                output_size,
                (feature_height, feature_width),
                row_start,
                row_end,
                column_start,
                column_end,
                key_row_start,
                key_row_end,
                key_column_start,
                key_column_end,
                float(decoder.window_ratio),
            )
            if attention_mode == "soft":
                attention_mask = _soft_attention_bias_tile(
                    *mask_arguments,
                    device=normalized_queries.device,
                    dtype=normalized_queries.dtype,
                )
            else:
                attention_mask = _attention_mask_tile(
                    *mask_arguments,
                    device=normalized_queries.device,
                )
            output = _attention_weighted_values(
                cross_attention,
                normalized_queries,
                local_keys,
                local_values,
                attention_mask,
            )
            output = output.reshape(
                output.shape[0],
                row_end - row_start,
                column_end - column_start,
                output.shape[-1],
            ).permute(0, 3, 1, 2)
            upsampled[:, :, row_start:row_end, column_start:column_end].copy_(
                output.to(device=output_device, dtype=output_dtype)
            )
    return upsampled


def _attention_weighted_values(
    cross_attention: Any,
    queries: Any,
    keys: Any,
    values: Any,
    attention_mask: Any | None,
) -> Any:
    """Apply AnyUp attention to custom values without storing attention weights.

    The official implementation asks ``MultiheadAttention`` for its averaged
    attention matrix, discards the module's projected output, then multiplies
    that matrix by the original feature values. Scaled dot-product attention
    can perform the same weighted reduction directly, avoiding both the unused
    value/output projections and the large query-by-key attention tensor.
    """
    attention = cross_attention.attention
    if not isinstance(attention, torch.nn.MultiheadAttention):
        return _materialized_attention_values(
            attention,
            queries,
            keys,
            values,
            attention_mask,
        )
    if (
        attention.dropout != 0
        or attention.add_zero_attn
        or attention.bias_k is not None
        or attention.bias_v is not None
    ):
        return _materialized_attention_values(
            attention,
            queries,
            keys,
            values,
            attention_mask,
        )

    embed_dim = attention.embed_dim
    head_count = attention.num_heads
    head_dim = embed_dim // head_count
    projection_bias = attention.in_proj_bias
    query_bias = None if projection_bias is None else projection_bias[:embed_dim]
    key_bias = (
        None if projection_bias is None else projection_bias[embed_dim : 2 * embed_dim]
    )
    if attention._qkv_same_embed_dim:
        query_weight = attention.in_proj_weight[:embed_dim]
        key_weight = attention.in_proj_weight[embed_dim : 2 * embed_dim]
    else:
        query_weight = attention.q_proj_weight
        key_weight = attention.k_proj_weight

    projected_queries = functional.linear(queries, query_weight, query_bias)
    projected_keys = functional.linear(keys, key_weight, key_bias)
    projected_queries = projected_queries.reshape(
        projected_queries.shape[0],
        projected_queries.shape[1],
        head_count,
        head_dim,
    ).transpose(1, 2)
    projected_keys = projected_keys.reshape(
        projected_keys.shape[0],
        projected_keys.shape[1],
        head_count,
        head_dim,
    ).transpose(1, 2)
    head_values = values.unsqueeze(1).expand(-1, head_count, -1, -1)
    if attention_mask is not None:
        attention_mask = attention_mask[None, None]
        if attention_mask.dtype == torch.bool:
            # MultiheadAttention uses True for blocked positions, whereas SDPA
            # uses True for positions that participate in attention.
            attention_mask = ~attention_mask

    attended = functional.scaled_dot_product_attention(
        projected_queries,
        projected_keys,
        head_values,
        attn_mask=attention_mask,
        dropout_p=0.0,
    )
    return attended.mean(dim=1)


def _materialized_attention_values(
    attention: Any,
    queries: Any,
    keys: Any,
    values: Any,
    attention_mask: Any | None,
) -> Any:
    _ignored, weights = attention(
        queries,
        keys,
        keys,
        average_attn_weights=True,
        attn_mask=attention_mask,
    )
    return torch.einsum("bij,bjd->bid", weights, values)


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


def _query_tile_size(
    output_size: tuple[int, int],
    q_chunk_size: int,
) -> tuple[int, int]:
    """Choose compact 2D query tiles without exceeding the chunk budget."""
    output_height, output_width = output_size
    aspect_adjusted_height = math.sqrt(q_chunk_size * output_height / output_width)
    tile_height = min(
        output_height,
        q_chunk_size,
        max(1, int(aspect_adjusted_height)),
    )
    tile_width = min(output_width, max(1, q_chunk_size // tile_height))
    return tile_height, tile_width


def _local_key_bounds(
    output_size: tuple[int, int],
    feature_size: tuple[int, int],
    row_start: int,
    row_end: int,
    column_start: int,
    column_end: int,
    window_ratio: float,
) -> tuple[int, int, int, int]:
    """Return a conservative key rectangle containing a query tile's support."""
    output_height, output_width = output_size
    feature_height, feature_width = feature_size
    if window_ratio <= 0:
        return 0, feature_height, 0, feature_width

    first_row = (row_start + 0.5) / output_height
    last_row = (row_end - 0.5) / output_height
    first_column = (column_start + 0.5) / output_width
    last_column = (column_end - 0.5) / output_width
    key_row_start = max(
        0,
        math.floor((first_row - window_ratio) * feature_height) - 1,
    )
    key_row_end = min(
        feature_height,
        math.ceil((last_row + window_ratio) * feature_height) + 1,
    )
    key_column_start = max(
        0,
        math.floor((first_column - window_ratio) * feature_width) - 1,
    )
    key_column_end = min(
        feature_width,
        math.ceil((last_column + window_ratio) * feature_width) + 1,
    )
    return key_row_start, key_row_end, key_column_start, key_column_end


def _attention_mask_rows(
    output_size: tuple[int, int],
    feature_size: tuple[int, int],
    row_start: int,
    row_end: int,
    window_ratio: float,
    *,
    device: Any,
) -> Any | None:
    return _attention_mask_tile(
        output_size,
        feature_size,
        row_start,
        row_end,
        0,
        output_size[1],
        0,
        feature_size[0],
        0,
        feature_size[1],
        window_ratio,
        device=device,
    )


def _attention_mask_tile(
    output_size: tuple[int, int],
    feature_size: tuple[int, int],
    row_start: int,
    row_end: int,
    column_start: int,
    column_end: int,
    key_row_start: int,
    key_row_end: int,
    key_column_start: int,
    key_column_end: int,
    window_ratio: float,
    *,
    device: Any,
) -> Any | None:
    if window_ratio <= 0:
        return None
    output_height, output_width = output_size
    feature_height, feature_width = feature_size
    output_rows = torch.arange(row_start, row_end, device=device)
    output_columns = torch.arange(column_start, column_end, device=device)
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
    feature_rows = torch.arange(key_row_start, key_row_end, device=device)
    feature_columns = torch.arange(key_column_start, key_column_end, device=device)
    row_allowed = (feature_rows >= row_minimum) & (feature_rows < row_maximum)
    column_allowed = (feature_columns >= column_minimum) & (
        feature_columns < column_maximum
    )
    allowed = row_allowed.unsqueeze(2) & column_allowed.unsqueeze(1)
    return ~allowed.reshape(
        -1,
        (key_row_end - key_row_start) * (key_column_end - key_column_start),
    )


def _soft_attention_bias_rows(
    output_size: tuple[int, int],
    feature_size: tuple[int, int],
    row_start: int,
    row_end: int,
    window_ratio: float,
    *,
    device: Any,
    dtype: Any,
) -> Any | None:
    """Create a continuous local-attention bias for a range of output rows."""
    return _soft_attention_bias_tile(
        output_size,
        feature_size,
        row_start,
        row_end,
        0,
        output_size[1],
        0,
        feature_size[0],
        0,
        feature_size[1],
        window_ratio,
        device=device,
        dtype=dtype,
    )


def _soft_attention_bias_tile(
    output_size: tuple[int, int],
    feature_size: tuple[int, int],
    row_start: int,
    row_end: int,
    column_start: int,
    column_end: int,
    key_row_start: int,
    key_row_end: int,
    key_column_start: int,
    key_column_end: int,
    window_ratio: float,
    *,
    device: Any,
    dtype: Any,
) -> Any | None:
    """Create a continuous local-attention bias for one query/key tile pair."""
    if window_ratio <= 0:
        return None
    output_height, output_width = output_size
    feature_height, feature_width = feature_size
    output_rows = torch.arange(row_start, row_end, device=device, dtype=torch.float32)
    output_columns = torch.arange(
        column_start,
        column_end,
        device=device,
        dtype=torch.float32,
    )
    query_rows = (output_rows + 0.5) / output_height
    query_columns = (output_columns + 0.5) / output_width
    query_rows, query_columns = torch.meshgrid(
        query_rows,
        query_columns,
        indexing="ij",
    )
    query_rows = query_rows.reshape(-1, 1)
    query_columns = query_columns.reshape(-1, 1)
    key_rows = (
        torch.arange(key_row_start, key_row_end, device=device).float() + 0.5
    ) / feature_height
    key_columns = (
        torch.arange(key_column_start, key_column_end, device=device).float() + 0.5
    ) / feature_width
    row_weights = _cosine_window_weights(
        (query_rows - key_rows).abs(),
        window_ratio,
        0.5 / feature_height,
    )
    column_weights = _cosine_window_weights(
        (query_columns - key_columns).abs(),
        window_ratio,
        0.5 / feature_width,
    )
    weights = row_weights.unsqueeze(2) * column_weights.unsqueeze(1)
    weights = weights.reshape(
        -1,
        (key_row_end - key_row_start) * (key_column_end - key_column_start),
    )
    bias = torch.where(
        weights > 0,
        weights.log(),
        torch.full_like(weights, float("-inf")),
    )
    return bias.to(dtype=dtype)


def _cosine_window_weights(
    distance: Any,
    window_ratio: float,
    half_feature_cell: float,
) -> Any:
    inner = max(0.0, window_ratio - half_feature_cell)
    outer = window_ratio + half_feature_cell
    progress = ((distance - inner) / (outer - inner)).clamp(0, 1)
    weights = 0.5 * (1 + torch.cos(torch.pi * progress))
    return torch.where(distance < outer, weights, torch.zeros_like(weights))


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


def _validate_attention_mode(attention_mode: str) -> None:
    if attention_mode not in {"hard", "soft"}:
        raise ValueError("attention_mode must be 'hard' or 'soft'.")


def _channel_values(values: Iterable[float], name: str, tensor: Any) -> Any:
    resolved = tuple(values)
    if len(resolved) != 3:
        raise ValueError(f"{name} must contain three values.")
    return torch.tensor(resolved, dtype=tensor.dtype, device=tensor.device)[
        None, :, None, None
    ]
