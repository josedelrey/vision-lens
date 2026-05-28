from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from vision_lens.models import ModelMetadata

LayerSelection = Literal["all"] | int | Iterable[int]
HeadFusion = Literal["mean", "max", "none"]


@dataclass(frozen=True)
class LayerAttentionMaps:
    layer_index: int
    maps: Any
    head_indices: tuple[int, ...] | None
    head_fusion: HeadFusion
    patch_grid: tuple[int, int]


@dataclass(frozen=True)
class AttentionExtractionResult:
    logits: Any
    layers: tuple[LayerAttentionMaps, ...]
    image_size: tuple[int, int]


def extract_attention_maps(
    model: Any,
    inputs: Any,
    metadata: ModelMetadata,
    layers: LayerSelection = "all",
    heads: Iterable[int] | None = None,
    head_fusion: HeadFusion = "mean",
) -> AttentionExtractionResult:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to extract attention maps. "
            "Install the project dependencies in the cv environment."
        ) from error

    blocks = _vit_blocks(model)
    layer_indices = _select_layers(layers, total_layers=len(blocks))
    head_indices = None if heads is None else tuple(heads)
    captured_attention: dict[int, Any] = {}

    handles = [
        blocks[layer_index].attn.register_forward_pre_hook(
            _capture_attention_hook(captured_attention, layer_index)
        )
        for layer_index in layer_indices
    ]

    try:
        with torch.no_grad():
            model_inputs = inputs.to(_model_device(model))
            logits = model(model_inputs)
    finally:
        for handle in handles:
            handle.remove()

    layer_maps = tuple(
        LayerAttentionMaps(
            layer_index=layer_index,
            maps=class_token_attention_to_map(
                captured_attention[layer_index],
                image_size=metadata.image_size,
                patch_size=metadata.patch_size,
                heads=head_indices,
                head_fusion=head_fusion,
            ),
            head_indices=head_indices,
            head_fusion=head_fusion,
            patch_grid=infer_patch_grid(
                num_patches=captured_attention[layer_index].shape[-1] - 1,
                image_size=metadata.image_size,
                patch_size=metadata.patch_size,
            ),
        )
        for layer_index in layer_indices
    )

    return AttentionExtractionResult(
        logits=logits.detach().cpu(),
        layers=layer_maps,
        image_size=metadata.image_size,
    )


def class_token_attention_to_map(
    attention: Any,
    image_size: tuple[int, int],
    patch_size: tuple[int, int] | None,
    heads: Iterable[int] | None = None,
    head_fusion: HeadFusion = "mean",
    normalize: bool = True,
) -> Any:
    try:
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to convert attention tensors. "
            "Install the project dependencies in the cv environment."
        ) from error

    _validate_attention_tensor(attention)
    selected_heads = _select_heads(attention, heads)
    cls_attention = selected_heads[:, :, 0, 1:]
    patch_grid = infer_patch_grid(
        num_patches=cls_attention.shape[-1],
        image_size=image_size,
        patch_size=patch_size,
    )
    fused_attention = _fuse_heads(cls_attention, head_fusion)
    patch_maps = fused_attention.reshape(
        fused_attention.shape[0],
        fused_attention.shape[1],
        patch_grid[0],
        patch_grid[1],
    )
    resized_maps = functional.interpolate(
        patch_maps,
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )

    if normalize:
        return normalize_maps(resized_maps)
    return resized_maps


def infer_patch_grid(
    num_patches: int,
    image_size: tuple[int, int],
    patch_size: tuple[int, int] | None,
) -> tuple[int, int]:
    if patch_size is not None:
        grid = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        if grid[0] * grid[1] == num_patches:
            return grid

    grid_size = int(num_patches**0.5)
    if grid_size * grid_size != num_patches:
        raise ValueError(
            "Could not infer a 2D patch grid from "
            f"{num_patches} patches and patch_size={patch_size}."
        )
    return (grid_size, grid_size)


def normalize_maps(maps: Any, eps: float = 1e-8) -> Any:
    flat_maps = maps.flatten(start_dim=2)
    minimum = flat_maps.min(dim=-1).values[:, :, None, None]
    maximum = flat_maps.max(dim=-1).values[:, :, None, None]
    return (maps - minimum) / (maximum - minimum + eps)


def _capture_attention_hook(captured_attention: dict[int, Any], layer_index: int):
    def hook(module: Any, args: tuple[Any, ...]) -> None:
        captured_attention[layer_index] = _compute_timm_attention(module, args[0])

    return hook


def _compute_timm_attention(attention_module: Any, tokens: Any) -> Any:
    batch_size, token_count, _ = tokens.shape
    qkv = attention_module.qkv(tokens)
    qkv = qkv.reshape(
        batch_size,
        token_count,
        3,
        attention_module.num_heads,
        -1,
    )
    qkv = qkv.permute(2, 0, 3, 1, 4)
    query, key, _value = qkv.unbind(0)
    query = attention_module.q_norm(query)
    key = attention_module.k_norm(key)
    attention = (query @ key.transpose(-2, -1)) * attention_module.scale
    return attention.softmax(dim=-1).detach().cpu()


def _vit_blocks(model: Any) -> tuple[Any, ...]:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("Expected a ViT-like model with a `blocks` attribute.")
    return tuple(blocks)


def _select_layers(layers: LayerSelection, total_layers: int) -> tuple[int, ...]:
    if layers == "all":
        return tuple(range(total_layers))
    if isinstance(layers, int):
        selected_layers = (layers,)
    else:
        selected_layers = tuple(layers)

    if not selected_layers:
        raise ValueError("At least one attention layer must be selected.")
    if len(set(selected_layers)) != len(selected_layers):
        raise ValueError("Attention layer selection must not contain duplicates.")
    for layer_index in selected_layers:
        if not isinstance(layer_index, int) or layer_index < 0:
            raise ValueError("Attention layers must be non-negative integers.")
        if layer_index >= total_layers:
            raise ValueError(
                f"Layer {layer_index} is out of range for {total_layers} layers."
            )
    return selected_layers


def _select_heads(attention: Any, heads: Iterable[int] | None) -> Any:
    if heads is None:
        return attention

    head_indices = tuple(heads)
    if not head_indices:
        raise ValueError("Head selection must not be empty.")
    if len(set(head_indices)) != len(head_indices):
        raise ValueError("Head selection must not contain duplicates.")
    num_heads = attention.shape[1]
    for head_index in head_indices:
        if not isinstance(head_index, int) or head_index < 0:
            raise ValueError("Attention heads must be non-negative integers.")
        if head_index >= num_heads:
            raise ValueError(
                f"Head {head_index} is out of range for {num_heads} heads."
            )
    return attention[:, head_indices, :, :]


def _fuse_heads(cls_attention: Any, head_fusion: HeadFusion) -> Any:
    if head_fusion == "mean":
        return cls_attention.mean(dim=1, keepdim=True)
    if head_fusion == "max":
        return cls_attention.max(dim=1, keepdim=True).values
    if head_fusion == "none":
        return cls_attention
    raise ValueError("head_fusion must be one of: max, mean, none.")


def _validate_attention_tensor(attention: Any) -> None:
    if len(attention.shape) != 4:
        raise ValueError(
            "Attention tensor must have shape "
            "(batch, heads, query_tokens, key_tokens)."
        )
    if attention.shape[-1] < 2:
        raise ValueError("Attention tensor must include class and patch tokens.")
    if attention.shape[-1] != attention.shape[-2]:
        raise ValueError("Attention query and key dimensions must match.")


def _model_device(model: Any) -> Any:
    return next(model.parameters()).device
