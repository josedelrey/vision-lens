from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_lens.attention import AttentionExtractionResult, LayerAttentionMaps
from vision_lens.config import VisionLensConfig, load_config
from vision_lens.images import build_preprocess, load_images, preprocess_images
from vision_lens.models import LoadedModel, load_model
from vision_lens.visualization import (
    make_image_comparison_grid,
    make_layer_comparison_grid,
    overlay_attention,
    render_heatmap,
    save_image,
)


@dataclass(frozen=True)
class PipelineResult:
    config: VisionLensConfig
    loaded_model: LoadedModel
    attention: AttentionExtractionResult
    output_paths: tuple[Path, ...]


def run_vit_attention(
    config_path: str | Path = "configs/vit_attention.example.yaml",
) -> PipelineResult:
    config = load_config(config_path)
    return run_vit_attention_from_config(config)


def run_vit_attention_from_config(config: VisionLensConfig) -> PipelineResult:
    if config.task != "vit_attention":
        raise ValueError(f"Expected task='vit_attention', got {config.task!r}.")
    if config.attention is None:
        raise ValueError("ViT attention pipeline requires an attention config.")

    loaded_model = load_model(config)
    images = load_images(config.images.paths)
    transform = build_preprocess(
        loaded_model.model,
        backend=config.model.backend,
        image_size=config.runtime.image_size,
    )
    inputs = preprocess_images(images, transform)
    attention = _extract_attention(
        loaded_model=loaded_model,
        inputs=inputs,
        config=config,
    )

    output_paths = export_attention_outputs(
        images=images,
        image_paths=config.images.paths,
        attention=attention,
        output_dir=config.output.directory,
        alpha=config.visualization.overlay_alpha,
    )

    return PipelineResult(
        config=config,
        loaded_model=loaded_model,
        attention=attention,
        output_paths=output_paths,
    )


def export_attention_outputs(
    images: list[Any],
    image_paths: tuple[Path, ...],
    attention: AttentionExtractionResult,
    output_dir: Path,
    alpha: float,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    labels = [path.stem for path in image_paths]

    for image_index, image in enumerate(images):
        for layer in attention.layers:
            image_layer = _layer_for_image(layer, image_index)
            for head_index in range(_head_count(image_layer)):
                suffix = _head_suffix(image_layer, head_index)
                stem = f"{labels[image_index]}_layer-{layer.layer_index}_{suffix}"

                heatmap = render_heatmap(image_layer.maps, head_index=head_index)
                overlay = overlay_attention(
                    image,
                    image_layer.maps,
                    alpha=alpha,
                    head_index=head_index,
                )

                output_paths.append(
                    save_image(heatmap, output_dir / f"{stem}_heatmap.png")
                )
                output_paths.append(
                    save_image(overlay, output_dir / f"{stem}_overlay.png")
                )

    for image_index, image in enumerate(images):
        image_layers = tuple(
            _layer_for_image(layer, image_index) for layer in attention.layers
        )
        for head_index in range(_head_count(image_layers[0])):
            suffix = _head_suffix(image_layers[0], head_index)
            output_path = output_dir / f"{labels[image_index]}_layers_{suffix}.png"
            make_layer_comparison_grid(
                image,
                image_layers,
                output_path=output_path,
                alpha=alpha,
                head_index=head_index,
            )
            output_paths.append(output_path)

    for layer in attention.layers:
        image_maps = [_slice_batch(layer.maps, index) for index in range(len(images))]
        for head_index in range(_head_count(layer)):
            suffix = _head_suffix(layer, head_index)
            output_path = output_dir / f"layer-{layer.layer_index}_images_{suffix}.png"
            make_image_comparison_grid(
                images,
                image_maps,
                labels=labels,
                output_path=output_path,
                alpha=alpha,
                head_index=head_index,
            )
            output_paths.append(output_path)

    return tuple(output_paths)


def _extract_attention(
    loaded_model: LoadedModel,
    inputs: Any,
    config: VisionLensConfig,
) -> AttentionExtractionResult:
    from vision_lens.attention import extract_attention_maps

    if config.attention is None:
        raise ValueError("ViT attention pipeline requires an attention config.")

    return extract_attention_maps(
        loaded_model.model,
        inputs,
        loaded_model.metadata,
        layers=config.attention.layers,
        heads=config.attention.heads,
        head_fusion=config.attention.head_fusion,
    )


def _layer_for_image(layer: LayerAttentionMaps, image_index: int) -> LayerAttentionMaps:
    return LayerAttentionMaps(
        layer_index=layer.layer_index,
        maps=_slice_batch(layer.maps, image_index),
        head_indices=layer.head_indices,
        head_fusion=layer.head_fusion,
        patch_grid=layer.patch_grid,
    )


def _slice_batch(maps: Any, batch_index: int) -> Any:
    return maps[batch_index : batch_index + 1]


def _head_count(layer: LayerAttentionMaps) -> int:
    return int(layer.maps.shape[1])


def _head_suffix(layer: LayerAttentionMaps, head_index: int) -> str:
    if layer.head_fusion != "none":
        return f"heads-{layer.head_fusion}"
    if layer.head_indices is None:
        return f"head-{head_index}"
    return f"head-{layer.head_indices[head_index]}"
