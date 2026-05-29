from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_lens.attention import (
    AttentionExtractionResult,
    GradCamResult,
    LayerAttentionMaps,
    extract_gradcam,
)
from vision_lens.config import VisionLensConfig, load_config
from vision_lens.images import (
    build_preprocess,
    load_images,
    preprocess_images,
    tensors_to_display_images,
)
from vision_lens.models import LoadedModel, load_model
from vision_lens.visualization import (
    image_grid,
    labeled_image,
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


@dataclass(frozen=True)
class GradCamPipelineResult:
    config: VisionLensConfig
    loaded_model: LoadedModel
    gradcam: GradCamResult
    output_paths: tuple[Path, ...]


def run_vit_attention(
    config_path: str | Path = "configs/vit_attention.example.yaml",
) -> PipelineResult:
    config = load_config(config_path)
    return run_vit_attention_from_config(config)


def run_pipeline(config_path: str | Path) -> PipelineResult | GradCamPipelineResult:
    config = load_config(config_path)
    if config.task == "vit_attention":
        return run_vit_attention_from_config(config)
    if config.task == "gradcam":
        return run_gradcam_from_config(config)

    raise ValueError(f"Unsupported task: {config.task}")


def run_vit_rollout_comparison(
    config_path: str | Path = "configs/vit_attention.example.yaml",
    output_dir: str | Path | None = None,
) -> PipelineResult:
    config = load_config(config_path)
    return run_vit_rollout_comparison_from_config(config, output_dir=output_dir)


def run_vit_rollout_comparison_from_config(
    config: VisionLensConfig,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    from vision_lens.attention import extract_attention_rollout

    if config.task != "vit_attention":
        raise ValueError(f"Expected task='vit_attention', got {config.task!r}.")
    if config.attention is None:
        raise ValueError("ViT rollout comparison requires an attention config.")

    loaded_model = load_model(config)
    images = load_images(config.images.paths)
    transform = build_preprocess(
        loaded_model.model,
        backend=config.model.backend,
        image_size=config.runtime.image_size,
    )
    inputs = preprocess_images(images, transform)
    display_images = tensors_to_display_images(
        inputs,
        loaded_model.metadata.data_config,
    )

    rollout = extract_attention_rollout(
        loaded_model.model,
        inputs,
        loaded_model.metadata,
        layers=config.attention.layers,
    )
    layer_attention = _extract_attention(
        loaded_model=loaded_model,
        inputs=inputs,
        config=config,
    )

    resolved_output_dir = Path(output_dir) if output_dir else config.output.directory
    output_paths = export_rollout_comparison_outputs(
        images=display_images,
        image_paths=config.images.paths,
        layer_attention=layer_attention,
        rollout=rollout,
        output_dir=resolved_output_dir,
        alpha=config.visualization.overlay_alpha,
        cmap=config.visualization.cmap,
    )

    return PipelineResult(
        config=config,
        loaded_model=loaded_model,
        attention=rollout,
        output_paths=output_paths,
    )


def run_gradcam(
    config_path: str | Path = "configs/gradcam.example.yaml",
) -> GradCamPipelineResult:
    config = load_config(config_path)
    return run_gradcam_from_config(config)


def run_gradcam_from_config(config: VisionLensConfig) -> GradCamPipelineResult:
    if config.model.architecture != "cnn":
        raise ValueError("Grad-CAM pipeline expects a CNN model config.")

    loaded_model = load_model(config)
    images = load_images(config.images.paths)
    transform = build_preprocess(
        loaded_model.model,
        backend=config.model.backend,
        image_size=config.runtime.image_size,
    )
    inputs = preprocess_images(images, transform)
    display_images = tensors_to_display_images(
        inputs,
        loaded_model.metadata.data_config,
    )

    target_layer = None
    if config.model.options is not None:
        target_layer = config.model.options.get("gradcam_target_layer")

    gradcam = extract_gradcam(
        loaded_model.model,
        inputs,
        loaded_model.metadata,
        target_layer=target_layer,
    )
    output_paths = export_gradcam_outputs(
        images=display_images,
        image_paths=config.images.paths,
        gradcam=gradcam,
        output_dir=config.output.directory,
        alpha=config.visualization.overlay_alpha,
        cmap=config.visualization.cmap,
    )
    return GradCamPipelineResult(
        config=config,
        loaded_model=loaded_model,
        gradcam=gradcam,
        output_paths=output_paths,
    )


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
    display_images = tensors_to_display_images(
        inputs,
        loaded_model.metadata.data_config,
    )

    attention = _extract_attention(
        loaded_model=loaded_model,
        inputs=inputs,
        config=config,
    )

    output_paths = export_attention_outputs(
        images=display_images,
        image_paths=config.images.paths,
        attention=attention,
        output_dir=config.output.directory,
        alpha=config.visualization.overlay_alpha,
        cmap=config.visualization.cmap,
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
    cmap: str,
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

                heatmap = render_heatmap(
                    image_layer.maps,
                    cmap=cmap,
                    head_index=head_index,
                )
                overlay = overlay_attention(
                    image,
                    image_layer.maps,
                    alpha=alpha,
                    cmap=cmap,
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
                cmap=cmap,
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
                cmap=cmap,
                head_index=head_index,
            )
            output_paths.append(output_path)

    return tuple(output_paths)


def export_rollout_comparison_outputs(
    images: list[Any],
    image_paths: tuple[Path, ...],
    layer_attention: AttentionExtractionResult,
    rollout: AttentionExtractionResult,
    output_dir: Path,
    alpha: float,
    cmap: str,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    labels = [path.stem for path in image_paths]

    for image_index, image in enumerate(images):
        tiles = []
        for layer, rollout_layer in zip(layer_attention.layers, rollout.layers):
            layer_for_image = _layer_for_image(layer, image_index)
            rollout_for_image = _layer_for_image(rollout_layer, image_index)

            tiles.append(
                labeled_image(
                    overlay_attention(
                        image,
                        layer_for_image.maps,
                        alpha=alpha,
                        cmap=cmap,
                    ),
                    f"layer {layer.layer_index}",
                )
            )
            tiles.append(
                labeled_image(
                    overlay_attention(
                        image,
                        rollout_for_image.maps,
                        alpha=alpha,
                        cmap=cmap,
                    ),
                    f"rollout {rollout_layer.layer_index}",
                )
            )

            stem = f"{labels[image_index]}_rollout-{rollout_layer.layer_index}"
            output_paths.append(
                save_image(
                    render_heatmap(rollout_for_image.maps, cmap=cmap),
                    output_dir / f"{stem}_heatmap.png",
                )
            )
            output_paths.append(
                save_image(
                    overlay_attention(
                        image,
                        rollout_for_image.maps,
                        alpha=alpha,
                        cmap=cmap,
                    ),
                    output_dir / f"{stem}_overlay.png",
                )
            )

        grid_path = output_dir / f"{labels[image_index]}_rollout_comparison.png"
        save_image(image_grid(tiles, columns=2), grid_path)
        output_paths.append(grid_path)

    return tuple(output_paths)


def export_gradcam_outputs(
    images: list[Any],
    image_paths: tuple[Path, ...],
    gradcam: GradCamResult,
    output_dir: Path,
    alpha: float,
    cmap: str,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    labels = [path.stem for path in image_paths]

    for image_index, image in enumerate(images):
        maps = _slice_batch(gradcam.maps, image_index)
        stem = f"{labels[image_index]}_gradcam"
        output_paths.append(
            save_image(
                render_heatmap(maps, cmap=cmap),
                output_dir / f"{stem}_heatmap.png",
            )
        )
        output_paths.append(
            save_image(
                overlay_attention(image, maps, alpha=alpha, cmap=cmap),
                output_dir / f"{stem}_overlay.png",
            )
        )

    image_maps = [_slice_batch(gradcam.maps, index) for index in range(len(images))]
    grid_path = output_dir / "gradcam_images.png"
    make_image_comparison_grid(
        images,
        image_maps,
        labels=labels,
        output_path=grid_path,
        alpha=alpha,
        cmap=cmap,
    )
    output_paths.append(grid_path)
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
