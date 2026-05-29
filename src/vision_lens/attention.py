from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as functional

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


@dataclass(frozen=True)
class GradCamResult:
    logits: Any
    maps: Any
    target_layer: str
    target_classes: tuple[int, ...]
    image_size: tuple[int, int]


def extract_attention_maps(
    model: Any,
    inputs: Any,
    metadata: ModelMetadata,
    layers: LayerSelection = "all",
    heads: Iterable[int] | None = None,
    head_fusion: HeadFusion = "mean",
) -> AttentionExtractionResult:
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


def extract_attention_rollout(
    model: Any,
    inputs: Any,
    metadata: ModelMetadata,
    layers: LayerSelection = "all",
) -> AttentionExtractionResult:
    blocks = _vit_blocks(model)
    layer_indices = _select_layers(layers, total_layers=len(blocks))
    max_layer = max(layer_indices)
    captured_attention: dict[int, Any] = {}
    capture_layers = tuple(range(max_layer + 1))
    handles = [
        blocks[layer_index].attn.register_forward_pre_hook(
            _capture_attention_hook(captured_attention, layer_index)
        )
        for layer_index in capture_layers
    ]

    try:
        with torch.no_grad():
            model_inputs = inputs.to(_model_device(model))
            logits = model(model_inputs)
    finally:
        for handle in handles:
            handle.remove()

    rollout_by_layer = compute_attention_rollout(
        tuple(captured_attention[layer_index] for layer_index in capture_layers)
    )
    layer_maps = tuple(
        LayerAttentionMaps(
            layer_index=layer_index,
            maps=token_attention_to_map(
                rollout_by_layer[layer_index],
                image_size=metadata.image_size,
                patch_size=metadata.patch_size,
            ),
            head_indices=None,
            head_fusion="mean",
            patch_grid=infer_patch_grid(
                num_patches=rollout_by_layer[layer_index].shape[-1] - 1,
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


def compute_attention_rollout(attentions: Iterable[Any]) -> tuple[Any, ...]:
    rollout_layers = []
    joint_attention = None
    for attention in attentions:
        _validate_attention_tensor(attention)
        fused_attention = attention.mean(dim=1)
        identity = torch.eye(
            fused_attention.shape[-1],
            device=fused_attention.device,
            dtype=fused_attention.dtype,
        )
        fused_attention = fused_attention + identity
        fused_attention = fused_attention / fused_attention.sum(
            dim=-1,
            keepdim=True,
        )
        if joint_attention is None:
            joint_attention = fused_attention
        else:
            joint_attention = fused_attention @ joint_attention
        rollout_layers.append(joint_attention.detach().cpu())

    return tuple(rollout_layers)


def token_attention_to_map(
    token_attention: Any,
    image_size: tuple[int, int],
    patch_size: tuple[int, int] | None,
    normalize: bool = True,
) -> Any:
    if len(token_attention.shape) != 3:
        raise ValueError("token_attention must have shape (batch, tokens, tokens).")
    if token_attention.shape[-1] != token_attention.shape[-2]:
        raise ValueError("token_attention query and key dimensions must match.")

    cls_attention = token_attention[:, 0, 1:]
    patch_grid = infer_patch_grid(
        num_patches=cls_attention.shape[-1],
        image_size=image_size,
        patch_size=patch_size,
    )
    patch_maps = cls_attention.reshape(
        cls_attention.shape[0],
        1,
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


def extract_gradcam(
    model: Any,
    inputs: Any,
    metadata: ModelMetadata,
    target_layer: str | None = None,
    target_classes: Iterable[int] | None = None,
) -> GradCamResult:
    resolved_layer_name, layer_module = _resolve_gradcam_layer(model, target_layer)
    activations = None
    gradients = None

    def forward_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
        nonlocal activations
        activations = output

    def backward_hook(_module: Any, _grad_input: Any, grad_output: Any) -> None:
        nonlocal gradients
        gradients = grad_output[0]

    forward_handle = layer_module.register_forward_hook(forward_hook)
    backward_handle = layer_module.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        model_inputs = inputs.to(_model_device(model))
        logits = model(model_inputs)
        if target_classes is None:
            class_tensor = logits.argmax(dim=1)
        else:
            class_tensor = torch.tensor(
                tuple(target_classes),
                device=logits.device,
                dtype=torch.long,
            )
        score = logits.gather(1, class_tensor[:, None]).sum()
        score.backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    maps = (weights * activations).sum(dim=1, keepdim=True)
    maps = functional.relu(maps)
    maps = functional.interpolate(
        maps,
        size=metadata.image_size,
        mode="bilinear",
        align_corners=False,
    )

    return GradCamResult(
        logits=logits.detach().cpu(),
        maps=normalize_maps(maps.detach().cpu()),
        target_layer=resolved_layer_name,
        target_classes=tuple(class_tensor.detach().cpu().tolist()),
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


def _resolve_gradcam_layer(model: Any, target_layer: str | None) -> tuple[str, Any]:
    if target_layer is not None:
        modules = dict(model.named_modules())
        if target_layer not in modules:
            raise ValueError(f"Grad-CAM target layer does not exist: {target_layer}")
        return target_layer, modules[target_layer]

    last_conv_name = None
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_name = name
            last_conv = module

    if last_conv_name is None or last_conv is None:
        raise ValueError("Could not find a convolutional layer for Grad-CAM.")
    return last_conv_name, last_conv


def _model_device(model: Any) -> Any:
    return next(model.parameters()).device
