from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from PIL import Image

from vision_lens.anyup import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    is_anyup_interpolation,
    prepare_anyup_image,
)
from vision_lens.attention import (
    AttentionExtractionResult,
    GradCamResult,
    LayerAttentionMaps,
    extract_gradcam,
    infer_patch_grid_from_image,
)
from vision_lens.config import (
    ColormapSpec,
    OutputConfig,
    VisionLensConfig,
    VisualizationConfig,
    load_config,
)
from vision_lens.feature_pca import (
    PatchPCAResult,
    extract_patch_embeddings,
    extract_patch_pca,
    fit_patch_pca_projection_batches,
    load_patch_pca_projection,
    project_patch_embeddings,
    render_patch_pca_images,
    save_patch_pca_projection,
)
from vision_lens.manifest import (
    check_manifest_overwrite,
    write_run_manifest,
)
from vision_lens.models import LoadedModel, load_model
from vision_lens.processing import (
    InputBatch,
    build_batch_preprocessor,
    iter_input_batches,
    preprocess_batch,
    unique_input_labels,
)
from vision_lens.progress import status, track_image_batches, track_units
from vision_lens.visualization import (
    image_grid,
    make_image_comparison_grid,
    make_layer_comparison_grid,
    overlay_attention,
    render_heatmap,
    save_grid,
    save_image,
    shared_value_range,
)

ROLLOUT_GRID_MAX_COLUMNS = 4


def _load_model_with_status(config: VisionLensConfig) -> LoadedModel:
    image_count = len(config.images.paths)
    input_label = "image" if image_count == 1 else "images"
    status(f"{config.analysis.method}: {image_count} {input_label}")
    status(f"Loading model {config.model.name}")
    loaded_model = load_model(config)
    status(f"Model ready on {loaded_model.metadata.device}")
    return loaded_model


def _tracked_input_batches(
    config: VisionLensConfig,
    labels: tuple[str, ...],
    description: str,
) -> Iterator[InputBatch]:
    return track_image_batches(
        iter_input_batches(
            config.images.paths,
            labels,
            batch_size=config.runtime.batch_size,
            workers=config.runtime.workers,
        ),
        total=len(config.images.paths),
        description=description,
    )


@dataclass(frozen=True)
class PipelineResult:
    config: VisionLensConfig
    loaded_model: LoadedModel
    attention: AttentionExtractionResult | None
    output_paths: tuple[Path, ...]
    processed_inputs: int = 0


@dataclass(frozen=True)
class GradCamPipelineResult:
    config: VisionLensConfig
    loaded_model: LoadedModel
    gradcam: GradCamResult | None
    output_paths: tuple[Path, ...]
    processed_inputs: int = 0


@dataclass(frozen=True)
class PatchPCAPipelineResult:
    config: VisionLensConfig
    loaded_model: LoadedModel
    patch_pca: PatchPCAResult | None
    output_paths: tuple[Path, ...]
    processed_inputs: int = 0


def run_vit_attention(
    config_path: str | Path = "configs/vit_attention.yaml",
) -> PipelineResult:
    config = load_config(config_path)
    return run_vit_attention_from_config(config)


def run_patch_pca(
    config_path: str | Path = "configs/patch_pca.dinov2.yaml",
) -> PatchPCAPipelineResult:
    config = load_config(config_path)
    return run_patch_pca_from_config(config)


def run_patch_pca_from_config(
    config: VisionLensConfig,
) -> PatchPCAPipelineResult:
    if config.task != "patch_pca":
        raise ValueError(f"Expected task='patch_pca', got {config.task!r}.")
    if config.model.architecture != "vit":
        raise ValueError("Patch PCA pipeline expects a ViT model config.")
    if config.patch_pca is None:
        raise ValueError("Patch PCA pipeline requires a patch_pca config.")

    started_at = datetime.now(timezone.utc)
    check_manifest_overwrite(config)
    labels = unique_input_labels(config.images.paths)
    projection = (
        load_patch_pca_projection(config.analysis.projection_path)
        if config.analysis.projection == "load"
        else None
    )
    _apply_seed(config.runtime.seed)
    loaded_model = _load_model_with_status(config)
    transform = build_batch_preprocessor(loaded_model, config.preprocessing)
    patch_grid = _patch_grid(loaded_model)
    page_offsets, total_grid_pages = _grid_page_plan(
        len(config.images.paths),
        config.runtime.batch_size,
        config.visualization.items_per_grid,
    )
    output_paths: list[Path] = []
    retained_patch_pca = None

    if projection is not None:
        for input_batch in _tracked_input_batches(config, labels, "Project images"):
            batch = preprocess_batch(
                input_batch,
                transform,
                loaded_model,
                include_display_images=False,
            )
            patch_pca = extract_patch_pca(
                loaded_model.model,
                batch.inputs,
                loaded_model.metadata,
                projection=projection,
                interpolation=config.visualization.interpolation,
                guidance_image=_anyup_guidance(config, loaded_model, batch.inputs),
                output_size=_analysis_output_size(config, loaded_model),
                anyup_query_chunk_size=(config.visualization.anyup_query_chunk_size),
            )
            if len(config.images.paths) <= config.runtime.batch_size:
                retained_patch_pca = patch_pca
            output_paths.extend(
                export_patch_pca_outputs(
                    patch_pca,
                    image_paths=input_batch.paths,
                    output_dir=config.output.directory,
                    output_config=config.output,
                    visualization_config=config.visualization,
                    input_sizes=tuple(image.size for image in input_batch.images),
                    input_labels=input_batch.labels,
                    grid_page_offset=page_offsets[input_batch.index],
                    total_grid_pages=total_grid_pages,
                )
            )
    elif len(config.images.paths) <= config.runtime.batch_size:
        for input_batch in _tracked_input_batches(config, labels, "Analyze images"):
            batch = preprocess_batch(
                input_batch,
                transform,
                loaded_model,
                include_display_images=False,
            )
            patch_pca = extract_patch_pca(
                loaded_model.model,
                batch.inputs,
                loaded_model.metadata,
                foreground_threshold=config.patch_pca.foreground_threshold,
                foreground_side=config.patch_pca.foreground_side,
                rgb_fit_scope=config.patch_pca.rgb_fit_scope,
                interpolation=config.visualization.interpolation,
                guidance_image=_anyup_guidance(config, loaded_model, batch.inputs),
                output_size=_analysis_output_size(config, loaded_model),
                anyup_query_chunk_size=(config.visualization.anyup_query_chunk_size),
            )
            projection = patch_pca.projection
            retained_patch_pca = patch_pca
            output_paths.extend(
                export_patch_pca_outputs(
                    patch_pca,
                    image_paths=input_batch.paths,
                    output_dir=config.output.directory,
                    output_config=config.output,
                    visualization_config=config.visualization,
                    input_sizes=tuple(image.size for image in input_batch.images),
                    input_labels=input_batch.labels,
                    grid_page_offset=page_offsets[input_batch.index],
                    total_grid_pages=total_grid_pages,
                )
            )
    else:
        with TemporaryDirectory(prefix="vision-lens-pca-") as temporary_directory:
            staged_batches: list[
                tuple[
                    int,
                    tuple[Path, ...],
                    tuple[str, ...],
                    tuple[tuple[int, int], ...],
                    Path,
                    Path | None,
                ]
            ] = []
            for input_batch in _tracked_input_batches(
                config, labels, "Extract embeddings"
            ):
                batch = preprocess_batch(
                    input_batch,
                    transform,
                    loaded_model,
                    include_display_images=False,
                )
                embeddings = extract_patch_embeddings(
                    loaded_model.model,
                    batch.inputs,
                    patch_grid,
                )
                staged_path = (
                    Path(temporary_directory) / f"batch-{input_batch.index}.npy"
                )
                _save_array(embeddings, staged_path)
                guidance = _anyup_guidance(config, loaded_model, batch.inputs)
                guidance_path = None
                if guidance is not None:
                    guidance_path = (
                        Path(temporary_directory) / f"guidance-{input_batch.index}.npy"
                    )
                    _save_array(guidance, guidance_path)
                staged_batches.append(
                    (
                        input_batch.index,
                        input_batch.paths,
                        input_batch.labels,
                        tuple(image.size for image in input_batch.images),
                        staged_path,
                        guidance_path,
                    )
                )

            def embedding_batches() -> Any:
                for (
                    _index,
                    _paths,
                    _labels,
                    _sizes,
                    staged_path,
                    _guidance_path,
                ) in staged_batches:
                    yield np.load(staged_path, allow_pickle=False)

            status("Fitting PCA projection")
            projection = fit_patch_pca_projection_batches(
                embedding_batches,
                foreground_threshold=config.patch_pca.foreground_threshold,
                foreground_side=config.patch_pca.foreground_side,
                rgb_fit_scope=config.patch_pca.rgb_fit_scope,
            )
            for (
                batch_index,
                batch_paths,
                batch_labels,
                batch_sizes,
                staged_path,
                guidance_path,
            ) in track_units(
                staged_batches,
                total=len(config.images.paths),
                description="Render PCA",
                unit="image",
                size=lambda item: len(item[1]),
            ):
                embeddings = np.load(staged_path, allow_pickle=False)
                patch_pca = project_patch_embeddings(
                    embeddings,
                    patch_grid=patch_grid,
                    image_size=_analysis_output_size(config, loaded_model),
                    projection=projection,
                    interpolation=config.visualization.interpolation,
                    anyup_query_chunk_size=(
                        config.visualization.anyup_query_chunk_size
                    ),
                    guidance_image=(
                        None
                        if guidance_path is None
                        else torch.as_tensor(
                            np.load(guidance_path, allow_pickle=False)
                        ).to(loaded_model.metadata.device)
                    ),
                )
                output_paths.extend(
                    export_patch_pca_outputs(
                        patch_pca,
                        image_paths=batch_paths,
                        output_dir=config.output.directory,
                        output_config=config.output,
                        visualization_config=config.visualization,
                        input_sizes=batch_sizes,
                        input_labels=batch_labels,
                        grid_page_offset=page_offsets[batch_index],
                        total_grid_pages=total_grid_pages,
                    )
                )

    if config.analysis.save_projection is not None:
        assert projection is not None
        projection_path = _write_projection(
            projection,
            config.analysis.save_projection,
            config.output.overwrite,
        )
        if projection_path is not None:
            output_paths.insert(0, projection_path)
    output_paths_tuple = tuple(output_paths)
    write_run_manifest(
        config,
        loaded_model,
        labels,
        output_paths_tuple,
        started_at=started_at,
    )
    return PatchPCAPipelineResult(
        config=config,
        loaded_model=loaded_model,
        patch_pca=retained_patch_pca,
        output_paths=output_paths_tuple,
        processed_inputs=len(config.images.paths),
    )


def run_vit_rollout_comparison(
    config_path: str | Path = "configs/vit_attention.yaml",
    output_dir: str | Path | None = None,
) -> PipelineResult:
    config = load_config(config_path)
    return run_vit_rollout_comparison_from_config(config, output_dir=output_dir)


def run_vit_rollout_comparison_from_config(
    config: VisionLensConfig,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    from vision_lens.attention import extract_attention_rollout

    if config.task not in {"vit_attention", "vit_rollout"}:
        raise ValueError(
            f"Expected task='vit_attention' or task='vit_rollout', got {config.task!r}."
        )
    if config.attention is None:
        raise ValueError("ViT rollout comparison requires an attention config.")

    started_at = datetime.now(timezone.utc)
    check_manifest_overwrite(config)
    labels = unique_input_labels(config.images.paths)
    _apply_seed(config.runtime.seed)
    loaded_model = _load_model_with_status(config)
    transform = build_batch_preprocessor(loaded_model, config.preprocessing)
    rendering = config.visualization
    if rendering.normalization == "shared":
        normalization_range = None
        for input_batch in _tracked_input_batches(config, labels, "Fit normalization"):
            batch = preprocess_batch(
                input_batch,
                transform,
                loaded_model,
                include_display_images=False,
            )
            fitted_rollout = extract_attention_rollout(
                loaded_model.model,
                batch.inputs,
                loaded_model.metadata,
                layers=config.attention.layers,
                normalize=False,
                interpolation=config.visualization.interpolation,
                guidance_image=_anyup_guidance(config, loaded_model, batch.inputs),
                output_size=_analysis_output_size(config, loaded_model),
                anyup_query_chunk_size=(config.visualization.anyup_query_chunk_size),
            )
            fitted_attention = _extract_attention(
                loaded_model=loaded_model,
                inputs=batch.inputs,
                config=config,
            )
            normalization_range = _extend_value_range(
                normalization_range,
                [
                    *(layer.maps for layer in fitted_attention.layers),
                    *(layer.maps for layer in fitted_rollout.layers),
                ],
            )
        rendering = replace(
            rendering,
            normalization="fixed",
            normalization_range=normalization_range,
        )
    output_paths = []
    retained_rollout = None
    for input_batch in _tracked_input_batches(config, labels, "Analyze images"):
        batch = preprocess_batch(input_batch, transform, loaded_model)
        rollout = extract_attention_rollout(
            loaded_model.model,
            batch.inputs,
            loaded_model.metadata,
            layers=config.attention.layers,
            normalize=config.visualization.normalization == "per_map",
            interpolation=config.visualization.interpolation,
            guidance_image=_anyup_guidance(config, loaded_model, batch.inputs),
            output_size=_analysis_output_size(config, loaded_model),
            anyup_query_chunk_size=config.visualization.anyup_query_chunk_size,
        )
        layer_attention = _extract_attention(
            loaded_model=loaded_model,
            inputs=batch.inputs,
            config=config,
        )
        if len(config.images.paths) <= config.runtime.batch_size:
            retained_rollout = rollout
        resolved_output_dir = (
            Path(output_dir) if output_dir else config.output.directory
        )
        output_paths.extend(
            export_rollout_comparison_outputs(
                images=_visualization_images(batch, rendering),
                image_paths=input_batch.paths,
                layer_attention=layer_attention,
                rollout=rollout,
                output_dir=resolved_output_dir,
                alpha=config.visualization.overlay_alpha,
                cmap=config.visualization.render_cmap,
                grid_format=config.visualization.grid_format,
                output_config=config.output,
                visualization_config=rendering,
                input_labels=input_batch.labels,
            )
        )
    output_paths_tuple = tuple(output_paths)
    write_run_manifest(
        config,
        loaded_model,
        labels,
        output_paths_tuple,
        started_at=started_at,
    )

    return PipelineResult(
        config=config,
        loaded_model=loaded_model,
        attention=retained_rollout,
        output_paths=output_paths_tuple,
        processed_inputs=len(config.images.paths),
    )


def run_gradcam(
    config_path: str | Path = "configs/gradcam.yaml",
) -> GradCamPipelineResult:
    config = load_config(config_path)
    return run_gradcam_from_config(config)


def run_gradcam_from_config(config: VisionLensConfig) -> GradCamPipelineResult:
    if config.model.architecture != "cnn":
        raise ValueError("Grad-CAM pipeline expects a CNN model config.")

    started_at = datetime.now(timezone.utc)
    check_manifest_overwrite(config)
    labels = unique_input_labels(config.images.paths)
    _apply_seed(config.runtime.seed)
    loaded_model = _load_model_with_status(config)
    if (
        config.analysis.target_class is not None
        and loaded_model.metadata.num_classes is not None
        and config.analysis.target_class >= loaded_model.metadata.num_classes
    ):
        raise ValueError(
            f"analysis.target_class={config.analysis.target_class} is outside the "
            f"model's class range 0 through {loaded_model.metadata.num_classes - 1}."
        )
    transform = build_batch_preprocessor(loaded_model, config.preprocessing)
    rendering = config.visualization
    if rendering.normalization == "shared":
        normalization_range = None
        for input_batch in _tracked_input_batches(config, labels, "Fit normalization"):
            batch = preprocess_batch(
                input_batch,
                transform,
                loaded_model,
                include_display_images=False,
            )
            fitted_gradcam = extract_gradcam(
                loaded_model.model,
                batch.inputs,
                loaded_model.metadata,
                target_layer=config.analysis.target_layer,
                target_classes=(
                    None
                    if config.analysis.target_class is None
                    else [config.analysis.target_class] * len(input_batch.paths)
                ),
                normalize=False,
                interpolation=config.visualization.interpolation,
                guidance_image=_anyup_guidance(config, loaded_model, batch.inputs),
                output_size=_analysis_output_size(config, loaded_model),
                anyup_query_chunk_size=(config.visualization.anyup_query_chunk_size),
            )
            normalization_range = _extend_value_range(
                normalization_range,
                [fitted_gradcam.maps],
            )
        rendering = replace(
            rendering,
            normalization="fixed",
            normalization_range=normalization_range,
        )
    page_offsets, total_grid_pages = _grid_page_plan(
        len(config.images.paths),
        config.runtime.batch_size,
        config.visualization.items_per_grid,
    )
    output_paths = []
    retained_gradcam = None
    for input_batch in _tracked_input_batches(config, labels, "Analyze images"):
        batch = preprocess_batch(input_batch, transform, loaded_model)
        gradcam = extract_gradcam(
            loaded_model.model,
            batch.inputs,
            loaded_model.metadata,
            target_layer=config.analysis.target_layer,
            target_classes=(
                None
                if config.analysis.target_class is None
                else [config.analysis.target_class] * len(input_batch.paths)
            ),
            normalize=config.visualization.normalization == "per_map",
            interpolation=config.visualization.interpolation,
            guidance_image=_anyup_guidance(config, loaded_model, batch.inputs),
            output_size=_analysis_output_size(config, loaded_model),
            anyup_query_chunk_size=config.visualization.anyup_query_chunk_size,
        )
        if len(config.images.paths) <= config.runtime.batch_size:
            retained_gradcam = gradcam
        output_paths.extend(
            export_gradcam_outputs(
                images=_visualization_images(batch, rendering),
                image_paths=input_batch.paths,
                gradcam=gradcam,
                output_dir=config.output.directory,
                alpha=config.visualization.overlay_alpha,
                cmap=config.visualization.render_cmap,
                grid_format=config.visualization.grid_format,
                output_config=config.output,
                visualization_config=rendering,
                input_labels=input_batch.labels,
                grid_page_offset=page_offsets[input_batch.index],
                total_grid_pages=total_grid_pages,
            )
        )
    output_paths_tuple = tuple(output_paths)
    write_run_manifest(
        config,
        loaded_model,
        labels,
        output_paths_tuple,
        started_at=started_at,
    )
    return GradCamPipelineResult(
        config=config,
        loaded_model=loaded_model,
        gradcam=retained_gradcam,
        output_paths=output_paths_tuple,
        processed_inputs=len(config.images.paths),
    )


def run_vit_attention_from_config(config: VisionLensConfig) -> PipelineResult:
    if config.task != "vit_attention":
        raise ValueError(f"Expected task='vit_attention', got {config.task!r}.")
    if config.attention is None:
        raise ValueError("ViT attention pipeline requires an attention config.")

    started_at = datetime.now(timezone.utc)
    check_manifest_overwrite(config)
    labels = unique_input_labels(config.images.paths)
    _apply_seed(config.runtime.seed)
    loaded_model = _load_model_with_status(config)
    transform = build_batch_preprocessor(loaded_model, config.preprocessing)
    rendering = config.visualization
    if rendering.normalization == "shared":
        normalization_range = None
        for input_batch in _tracked_input_batches(config, labels, "Fit normalization"):
            batch = preprocess_batch(
                input_batch,
                transform,
                loaded_model,
                include_display_images=False,
            )
            fitted_attention = _extract_attention(
                loaded_model=loaded_model,
                inputs=batch.inputs,
                config=config,
            )
            normalization_range = _extend_value_range(
                normalization_range,
                [layer.maps for layer in fitted_attention.layers],
            )
        rendering = replace(
            rendering,
            normalization="fixed",
            normalization_range=normalization_range,
        )
    page_offsets, total_grid_pages = _grid_page_plan(
        len(config.images.paths),
        config.runtime.batch_size,
        config.visualization.items_per_grid,
    )
    output_paths = []
    retained_attention = None
    for input_batch in _tracked_input_batches(config, labels, "Analyze images"):
        batch = preprocess_batch(input_batch, transform, loaded_model)
        attention = _extract_attention(
            loaded_model=loaded_model,
            inputs=batch.inputs,
            config=config,
        )
        if len(config.images.paths) <= config.runtime.batch_size:
            retained_attention = attention
        output_paths.extend(
            export_attention_outputs(
                images=_visualization_images(batch, rendering),
                image_paths=input_batch.paths,
                attention=attention,
                output_dir=config.output.directory,
                alpha=config.visualization.overlay_alpha,
                cmap=config.visualization.render_cmap,
                grid_format=config.visualization.grid_format,
                output_config=config.output,
                visualization_config=rendering,
                input_labels=input_batch.labels,
                grid_page_offset=page_offsets[input_batch.index],
                total_grid_pages=total_grid_pages,
            )
        )
    output_paths_tuple = tuple(output_paths)
    write_run_manifest(
        config,
        loaded_model,
        labels,
        output_paths_tuple,
        started_at=started_at,
    )

    return PipelineResult(
        config=config,
        loaded_model=loaded_model,
        attention=retained_attention,
        output_paths=output_paths_tuple,
        processed_inputs=len(config.images.paths),
    )


def export_attention_outputs(
    images: list[Any],
    image_paths: tuple[Path, ...],
    attention: AttentionExtractionResult,
    output_dir: Path,
    alpha: float,
    cmap: str | ColormapSpec,
    grid_format: str,
    output_config: OutputConfig | None = None,
    visualization_config: VisualizationConfig | None = None,
    input_labels: tuple[str, ...] | None = None,
    grid_page_offset: int = 0,
    total_grid_pages: int | None = None,
) -> tuple[Path, ...]:
    output = output_config or OutputConfig(output_dir)
    visualization = visualization_config or VisualizationConfig(
        overlay_alpha=alpha,
        cmap=cmap.name if isinstance(cmap, ColormapSpec) else cmap,
        cmap_black=(
            (cmap.black_threshold, cmap.black_blend_width, cmap.black_transparent)
            if isinstance(cmap, ColormapSpec)
            else None
        ),
        grid_format=grid_format,
    )
    images = _images_at_output_size(images, visualization, attention.image_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    labels = list(input_labels or tuple(path.stem for path in image_paths))
    normalization_range = _rendering_range(
        visualization,
        [layer.maps for layer in attention.layers],
    )

    for image_index, image in enumerate(images):
        for layer in attention.layers:
            image_layer = _layer_for_image(layer, image_index)
            for head_index in range(_head_count(image_layer)):
                suffix = _head_suffix(image_layer, head_index)
                stem = f"{labels[image_index]}_layer-{layer.layer_index}_{suffix}"

                if output.heatmaps:
                    path = output_dir / f"{stem}_heatmap.{output.image_format}"
                    if _can_write(path, output.overwrite):
                        heatmap = render_heatmap(
                            image_layer.maps,
                            cmap=cmap,
                            head_index=head_index,
                            normalization=visualization.normalization,
                            normalization_range=normalization_range,
                        )
                        heatmap = _resize_visualization(
                            heatmap,
                            image.size,
                            visualization,
                        )
                        output_paths.append(save_image(heatmap, path))
                if output.overlays:
                    path = output_dir / f"{stem}_overlay.{output.image_format}"
                    if _can_write(path, output.overwrite):
                        overlay = overlay_attention(
                            image,
                            image_layer.maps,
                            alpha=alpha,
                            alpha_curve_steepness=(
                                visualization.overlay_alpha_curve_steepness
                            ),
                            alpha_curve_midpoint=(
                                visualization.overlay_alpha_curve_midpoint
                            ),
                            cmap=cmap,
                            head_index=head_index,
                            normalization=visualization.normalization,
                            normalization_range=normalization_range,
                        )
                        output_paths.append(save_image(overlay, path))
                if output.raw_arrays:
                    path = output_dir / f"{stem}.{output.raw_format}"
                    if _can_write(path, output.overwrite):
                        output_paths.append(
                            _save_array(image_layer.maps[0, head_index], path)
                        )

    if not output.grids:
        return tuple(output_paths)

    for image_index, image in enumerate(images):
        image_layers = tuple(
            _layer_for_image(layer, image_index) for layer in attention.layers
        )
        for head_index in range(_head_count(image_layers[0])):
            suffix = _head_suffix(image_layers[0], head_index)
            pages = _chunks(image_layers, visualization.items_per_grid)
            for page_index, layer_page in enumerate(pages):
                stem = f"{labels[image_index]}_layers_{suffix}"
                output_path = _page_path(
                    output_dir,
                    stem,
                    grid_format,
                    page_index,
                    len(pages),
                )
                if not _can_write(output_path, output.overwrite):
                    continue
                make_layer_comparison_grid(
                    image,
                    layer_page,
                    output_path=output_path,
                    alpha=alpha,
                    alpha_curve_steepness=(visualization.overlay_alpha_curve_steepness),
                    alpha_curve_midpoint=visualization.overlay_alpha_curve_midpoint,
                    cmap=cmap,
                    head_index=head_index,
                    columns=visualization.columns,
                    tile_size=visualization.tile_size,
                    spacing=visualization.spacing,
                    padding=visualization.padding,
                    show_labels=visualization.labels,
                    background=visualization.background,
                    dpi=visualization.dpi,
                    normalization=visualization.normalization,
                    normalization_range=normalization_range,
                )
                output_paths.append(output_path)

    for layer in attention.layers:
        image_maps = [_slice_batch(layer.maps, index) for index in range(len(images))]
        for head_index in range(_head_count(layer)):
            suffix = _head_suffix(layer, head_index)
            image_pages = _chunks(
                tuple(range(len(images))), visualization.items_per_grid
            )
            for page_index, indices in enumerate(image_pages):
                stem = f"layer-{layer.layer_index}_images_{suffix}"
                output_path = _page_path(
                    output_dir,
                    stem,
                    grid_format,
                    grid_page_offset + page_index,
                    total_grid_pages or len(image_pages),
                )
                if not _can_write(output_path, output.overwrite):
                    continue
                make_image_comparison_grid(
                    [images[index] for index in indices],
                    [image_maps[index] for index in indices],
                    labels=[labels[index] for index in indices],
                    output_path=output_path,
                    alpha=alpha,
                    alpha_curve_steepness=(visualization.overlay_alpha_curve_steepness),
                    alpha_curve_midpoint=visualization.overlay_alpha_curve_midpoint,
                    cmap=cmap,
                    head_index=head_index,
                    columns=visualization.columns,
                    tile_size=visualization.tile_size,
                    spacing=visualization.spacing,
                    padding=visualization.padding,
                    show_labels=visualization.labels,
                    background=visualization.background,
                    dpi=visualization.dpi,
                    normalization=visualization.normalization,
                    normalization_range=normalization_range,
                )
                output_paths.append(output_path)

    return tuple(output_paths)


def export_patch_pca_outputs(
    patch_pca: PatchPCAResult,
    image_paths: tuple[Path, ...],
    output_dir: Path,
    output_config: OutputConfig | None = None,
    visualization_config: VisualizationConfig | None = None,
    input_sizes: tuple[tuple[int, int], ...] | None = None,
    input_labels: tuple[str, ...] | None = None,
    grid_page_offset: int = 0,
    total_grid_pages: int | None = None,
) -> tuple[Path, ...]:
    if len(patch_pca.images) != len(image_paths):
        raise ValueError("Patch PCA images and image_paths must have the same length.")

    output = output_config or OutputConfig(output_dir)
    visualization = visualization_config or VisualizationConfig()
    if input_sizes is not None and len(input_sizes) != len(patch_pca.images):
        raise ValueError("Patch PCA images and input_sizes must have the same length.")
    if visualization.output_size == "match" and input_sizes is None:
        raise ValueError("input_sizes are required when output_size is 'match'.")
    rendered_images = tuple(
        _render_pca_at_output_size(
            patch_pca,
            index,
            _resolved_image_output_size(
                visualization,
                input_sizes[index] if input_sizes is not None else None,
                image.size,
            ),
            visualization,
        )
        for index, image in enumerate(patch_pca.images)
    )
    labels = input_labels or tuple(path.stem for path in image_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    if output.heatmaps:
        for label, image in zip(labels, rendered_images, strict=True):
            output_path = output_dir / f"{label}_patch_pca.{output.image_format}"
            if _can_write(output_path, output.overwrite):
                output_paths.append(save_image(image, output_path))
    if output.raw_arrays:
        for index, label in enumerate(labels):
            embedding_path = (
                output_dir / f"{label}_patch_embeddings.{output.raw_format}"
            )
            mask_path = output_dir / f"{label}_foreground_mask.{output.raw_format}"
            if _can_write(embedding_path, output.overwrite):
                output_paths.append(
                    _save_array(patch_pca.patch_embeddings[index], embedding_path)
                )
            if _can_write(mask_path, output.overwrite):
                output_paths.append(
                    _save_array(patch_pca.foreground_mask[index], mask_path)
                )
    single_image_run = len(patch_pca.images) == 1 and total_grid_pages in {None, 1}
    if not output.grids or single_image_run:
        return tuple(output_paths)

    indices_pages = _chunks(
        tuple(range(len(patch_pca.images))),
        visualization.items_per_grid,
    )
    for page_index, indices in enumerate(indices_pages):
        page_images = [rendered_images[index] for index in indices]
        show_labels = False if visualization.labels is None else visualization.labels
        if show_labels:
            from vision_lens.visualization import labeled_image

            page_images = [
                labeled_image(image, labels[index])
                for image, index in zip(page_images, indices, strict=True)
            ]
        comparison = image_grid(
            page_images,
            columns=(
                min(len(page_images), 3)
                if visualization.columns is None
                else visualization.columns
            ),
            background=visualization.background or "white",
            gap=16 if visualization.spacing is None else visualization.spacing,
            padding=16 if visualization.padding is None else visualization.padding,
            tile_size=visualization.tile_size,
        )
        output_path = _page_path(
            output_dir,
            "patch_pca_comparison",
            visualization.grid_format,
            grid_page_offset + page_index,
            total_grid_pages or len(indices_pages),
        )
        if _can_write(output_path, output.overwrite):
            output_paths.append(
                save_image(comparison, output_path, dpi=visualization.dpi)
            )
    return tuple(output_paths)


def export_rollout_comparison_outputs(
    images: list[Any],
    image_paths: tuple[Path, ...],
    layer_attention: AttentionExtractionResult,
    rollout: AttentionExtractionResult,
    output_dir: Path,
    alpha: float,
    cmap: str | ColormapSpec,
    grid_format: str,
    output_config: OutputConfig | None = None,
    visualization_config: VisualizationConfig | None = None,
    input_labels: tuple[str, ...] | None = None,
) -> tuple[Path, ...]:
    output = output_config or OutputConfig(output_dir)
    visualization = visualization_config or VisualizationConfig(
        overlay_alpha=alpha,
        cmap=cmap.name if isinstance(cmap, ColormapSpec) else cmap,
        cmap_black=(
            (cmap.black_threshold, cmap.black_blend_width, cmap.black_transparent)
            if isinstance(cmap, ColormapSpec)
            else None
        ),
        grid_format=grid_format,
    )
    images = _images_at_output_size(images, visualization, rollout.image_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    labels = list(input_labels or tuple(path.stem for path in image_paths))
    normalization_range = _rendering_range(
        visualization,
        [
            *(layer.maps for layer in layer_attention.layers),
            *(layer.maps for layer in rollout.layers),
        ],
    )

    for image_index, image in enumerate(images):
        for _layer, rollout_layer in zip(
            layer_attention.layers,
            rollout.layers,
            strict=True,
        ):
            rollout_for_image = _layer_for_image(rollout_layer, image_index)
            stem = f"{labels[image_index]}_rollout-{rollout_layer.layer_index}"
            if output.heatmaps:
                path = output_dir / f"{stem}_heatmap.{output.image_format}"
                if _can_write(path, output.overwrite):
                    output_paths.append(
                        save_image(
                            _resize_visualization(
                                render_heatmap(
                                    rollout_for_image.maps,
                                    cmap=cmap,
                                    normalization=visualization.normalization,
                                    normalization_range=normalization_range,
                                ),
                                image.size,
                                visualization,
                            ),
                            path,
                        )
                    )
            if output.overlays:
                path = output_dir / f"{stem}_overlay.{output.image_format}"
                if _can_write(path, output.overwrite):
                    output_paths.append(
                        save_image(
                            overlay_attention(
                                image,
                                rollout_for_image.maps,
                                alpha=alpha,
                                alpha_curve_steepness=(
                                    visualization.overlay_alpha_curve_steepness
                                ),
                                alpha_curve_midpoint=(
                                    visualization.overlay_alpha_curve_midpoint
                                ),
                                cmap=cmap,
                                normalization=visualization.normalization,
                                normalization_range=normalization_range,
                            ),
                            path,
                        )
                    )
            if output.raw_arrays:
                path = output_dir / f"{stem}.{output.raw_format}"
                if _can_write(path, output.overwrite):
                    output_paths.append(_save_array(rollout_for_image.maps[0, 0], path))

        if output.grids:
            layer_pages = _chunks(
                tuple(range(len(rollout.layers))),
                visualization.items_per_grid,
            )
            for page_index, indices in enumerate(layer_pages):
                layer_page = _attention_subset(layer_attention, indices)
                rollout_page = _attention_subset(rollout, indices)
                grid_path = _page_path(
                    output_dir,
                    f"{labels[image_index]}_rollout_comparison",
                    grid_format,
                    page_index,
                    len(layer_pages),
                )
                if not _can_write(grid_path, output.overwrite):
                    continue
                comparison_images, comparison_labels, columns = _rollout_grid_items(
                    image=image,
                    layer_attention=layer_page,
                    rollout=rollout_page,
                    image_index=image_index,
                    alpha=alpha,
                    alpha_curve_steepness=(visualization.overlay_alpha_curve_steepness),
                    alpha_curve_midpoint=visualization.overlay_alpha_curve_midpoint,
                    cmap=cmap,
                    columns=visualization.columns,
                    normalization=visualization.normalization,
                    normalization_range=normalization_range,
                )
                save_grid(
                    comparison_images,
                    labels=comparison_labels,
                    output_path=grid_path,
                    columns=columns,
                    tile_size=visualization.tile_size,
                    spacing=visualization.spacing,
                    padding=visualization.padding,
                    show_labels=(
                        True if visualization.labels is None else visualization.labels
                    ),
                    background=visualization.background,
                    dpi=visualization.dpi,
                )
                output_paths.append(grid_path)

    return tuple(output_paths)


def export_gradcam_outputs(
    images: list[Any],
    image_paths: tuple[Path, ...],
    gradcam: GradCamResult,
    output_dir: Path,
    alpha: float,
    cmap: str | ColormapSpec,
    grid_format: str,
    output_config: OutputConfig | None = None,
    visualization_config: VisualizationConfig | None = None,
    input_labels: tuple[str, ...] | None = None,
    grid_page_offset: int = 0,
    total_grid_pages: int | None = None,
) -> tuple[Path, ...]:
    output = output_config or OutputConfig(output_dir)
    visualization = visualization_config or VisualizationConfig(
        overlay_alpha=alpha,
        cmap=cmap.name if isinstance(cmap, ColormapSpec) else cmap,
        cmap_black=(
            (cmap.black_threshold, cmap.black_blend_width, cmap.black_transparent)
            if isinstance(cmap, ColormapSpec)
            else None
        ),
        grid_format=grid_format,
    )
    images = _images_at_output_size(images, visualization, gradcam.image_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    labels = list(input_labels or tuple(path.stem for path in image_paths))
    normalization_range = _rendering_range(visualization, [gradcam.maps])

    for image_index, image in enumerate(images):
        maps = _slice_batch(gradcam.maps, image_index)
        stem = f"{labels[image_index]}_gradcam"
        if output.heatmaps:
            path = output_dir / f"{stem}_heatmap.{output.image_format}"
            if _can_write(path, output.overwrite):
                output_paths.append(
                    save_image(
                        _resize_visualization(
                            render_heatmap(
                                maps,
                                cmap=cmap,
                                normalization=visualization.normalization,
                                normalization_range=normalization_range,
                            ),
                            image.size,
                            visualization,
                        ),
                        path,
                    )
                )
        if output.overlays:
            path = output_dir / f"{stem}_overlay.{output.image_format}"
            if _can_write(path, output.overwrite):
                output_paths.append(
                    save_image(
                        overlay_attention(
                            image,
                            maps,
                            alpha=alpha,
                            alpha_curve_steepness=(
                                visualization.overlay_alpha_curve_steepness
                            ),
                            alpha_curve_midpoint=(
                                visualization.overlay_alpha_curve_midpoint
                            ),
                            cmap=cmap,
                            normalization=visualization.normalization,
                            normalization_range=normalization_range,
                        ),
                        path,
                    )
                )
        if output.raw_arrays:
            path = output_dir / f"{stem}.{output.raw_format}"
            if _can_write(path, output.overwrite):
                output_paths.append(_save_array(maps[0, 0], path))

    if output.grids:
        image_maps = [_slice_batch(gradcam.maps, index) for index in range(len(images))]
        image_pages = _chunks(tuple(range(len(images))), visualization.items_per_grid)
        for page_index, indices in enumerate(image_pages):
            grid_path = _page_path(
                output_dir,
                "gradcam_images",
                grid_format,
                grid_page_offset + page_index,
                total_grid_pages or len(image_pages),
            )
            if not _can_write(grid_path, output.overwrite):
                continue
            make_image_comparison_grid(
                [images[index] for index in indices],
                [image_maps[index] for index in indices],
                labels=[labels[index] for index in indices],
                output_path=grid_path,
                alpha=alpha,
                alpha_curve_steepness=(visualization.overlay_alpha_curve_steepness),
                alpha_curve_midpoint=visualization.overlay_alpha_curve_midpoint,
                cmap=cmap,
                columns=visualization.columns,
                tile_size=visualization.tile_size,
                spacing=visualization.spacing,
                padding=visualization.padding,
                show_labels=visualization.labels,
                background=visualization.background,
                dpi=visualization.dpi,
                normalization=visualization.normalization,
                normalization_range=normalization_range,
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
        normalize=config.visualization.normalization == "per_map",
        interpolation=config.visualization.interpolation,
        guidance_image=_anyup_guidance(config, loaded_model, inputs),
        output_size=_analysis_output_size(config, loaded_model),
        anyup_query_chunk_size=config.visualization.anyup_query_chunk_size,
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


def _rollout_grid_items(
    image: Any,
    layer_attention: AttentionExtractionResult,
    rollout: AttentionExtractionResult,
    image_index: int,
    alpha: float,
    cmap: str | ColormapSpec,
    columns: int | None = None,
    normalization: str = "per_map",
    normalization_range: tuple[float, float] | None = None,
    alpha_curve_steepness: float | None = None,
    alpha_curve_midpoint: float = 0.5,
) -> tuple[list[Any], list[str], int]:
    pairs = tuple(zip(layer_attention.layers, rollout.layers, strict=True))
    if not pairs:
        raise ValueError("Rollout comparison requires at least one layer.")

    columns = (
        min(columns, len(pairs))
        if columns
        else min(
            len(pairs),
            ROLLOUT_GRID_MAX_COLUMNS,
        )
    )
    comparison_images = []
    comparison_labels = []

    for chunk_start in range(0, len(pairs), columns):
        chunk = pairs[chunk_start : chunk_start + columns]
        layer_images = []
        layer_labels = []
        rollout_images = []
        rollout_labels = []

        for layer, rollout_layer in chunk:
            layer_for_image = _layer_for_image(layer, image_index)
            rollout_for_image = _layer_for_image(rollout_layer, image_index)
            layer_images.append(
                overlay_attention(
                    image,
                    layer_for_image.maps,
                    alpha=alpha,
                    alpha_curve_steepness=alpha_curve_steepness,
                    alpha_curve_midpoint=alpha_curve_midpoint,
                    cmap=cmap,
                    normalization=normalization,
                    normalization_range=normalization_range,
                )
            )
            layer_labels.append(f"layer {layer.layer_index}")
            rollout_images.append(
                overlay_attention(
                    image,
                    rollout_for_image.maps,
                    alpha=alpha,
                    alpha_curve_steepness=alpha_curve_steepness,
                    alpha_curve_midpoint=alpha_curve_midpoint,
                    cmap=cmap,
                    normalization=normalization,
                    normalization_range=normalization_range,
                )
            )
            rollout_labels.append(f"rollout {rollout_layer.layer_index}")

        padding = columns - len(chunk)
        if padding:
            blanks = [_blank_like(image) for _ in range(padding)]
            layer_images.extend(blanks)
            layer_labels.extend([""] * padding)
            rollout_images.extend(blanks)
            rollout_labels.extend([""] * padding)

        comparison_images.extend(layer_images)
        comparison_labels.extend(layer_labels)
        comparison_images.extend(rollout_images)
        comparison_labels.extend(rollout_labels)

    return comparison_images, comparison_labels, columns


def _blank_like(image: Any) -> Any:
    base_size = getattr(image, "size", (224, 224))
    return Image.new("RGB", base_size, "white")


def _analysis_output_size(
    config: VisionLensConfig,
    loaded_model: LoadedModel,
) -> tuple[int, int]:
    output_size = config.visualization.output_size
    if isinstance(output_size, tuple):
        width, height = output_size
        return (height, width)
    return loaded_model.metadata.image_size


def _images_at_output_size(
    images: list[Any],
    visualization: VisualizationConfig,
    native_size: tuple[int, int],
) -> list[Any]:
    if visualization.output_size == "match":
        return images
    target_size = (
        visualization.output_size
        if isinstance(visualization.output_size, tuple)
        else (native_size[1], native_size[0])
    )
    return [
        image
        if image.size == target_size
        else image.resize(target_size, Image.Resampling.BILINEAR)
        for image in images
    ]


def _resolved_image_output_size(
    visualization: VisualizationConfig,
    input_size: tuple[int, int] | None,
    native_size: tuple[int, int],
) -> tuple[int, int]:
    if visualization.output_size == "match":
        if input_size is None:
            raise ValueError("input_size is required when output_size is 'match'.")
        return input_size
    if isinstance(visualization.output_size, tuple):
        return visualization.output_size
    return native_size


def _visualization_images(
    batch: Any,
    visualization: VisualizationConfig,
) -> list[Any]:
    if visualization.output_size == "match":
        return list(batch.source.images)
    images = list(batch.display_images)
    if isinstance(visualization.output_size, tuple):
        images = [
            image.resize(visualization.output_size, Image.Resampling.BILINEAR)
            for image in images
        ]
    return images


def _resize_visualization(
    image: Image.Image,
    size: tuple[int, int],
    visualization: VisualizationConfig,
) -> Image.Image:
    if image.size == size:
        return image
    resampling = (
        Image.Resampling.NEAREST
        if visualization.interpolation == "nearest"
        else Image.Resampling.BILINEAR
    )
    return image.resize(size, resampling)


def _render_pca_at_output_size(
    patch_pca: PatchPCAResult,
    index: int,
    size: tuple[int, int],
    visualization: VisualizationConfig,
) -> Image.Image:
    image = patch_pca.images[index]
    if image.size == size:
        return image
    if patch_pca.rgb_patches is None:
        return _resize_visualization(image, size, visualization)
    if is_anyup_interpolation(visualization.interpolation):
        return _resize_visualization(image, size, visualization)
    return render_patch_pca_images(
        patch_pca.rgb_patches[index : index + 1],
        patch_pca.foreground_mask[index : index + 1],
        patch_pca.patch_grid,
        (size[1], size[0]),
        visualization.interpolation,
    )[0]


def _chunks(values: Any, size: int | None) -> tuple[Any, ...]:
    if size is None or size >= len(values):
        return (values,)
    return tuple(values[start : start + size] for start in range(0, len(values), size))


def _patch_grid(loaded_model: LoadedModel) -> tuple[int, int]:
    patch_size = loaded_model.metadata.patch_size
    if patch_size is None:
        raise ValueError("Patch PCA requires a model with a known patch size.")
    return infer_patch_grid_from_image(loaded_model.metadata.image_size, patch_size)


def _anyup_guidance(
    config: VisionLensConfig,
    loaded_model: LoadedModel,
    inputs: Any,
) -> Any | None:
    if not is_anyup_interpolation(config.visualization.interpolation):
        return None
    source_mean = None
    source_std = None
    if config.preprocessing.normalize:
        source_mean = (
            config.preprocessing.mean
            or loaded_model.metadata.data_config.get("mean", IMAGENET_MEAN)
        )
        source_std = config.preprocessing.std or loaded_model.metadata.data_config.get(
            "std", IMAGENET_STD
        )
    return prepare_anyup_image(
        inputs,
        source_mean=source_mean,
        source_std=source_std,
    ).to(loaded_model.metadata.device)


def _apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rendering_range(
    visualization: VisualizationConfig,
    values: list[Any],
) -> tuple[float, float] | None:
    if visualization.normalization == "per_map":
        return None
    if visualization.normalization == "fixed":
        return visualization.normalization_range
    return shared_value_range(values)


def _extend_value_range(
    current: tuple[float, float] | None,
    values: list[Any],
) -> tuple[float, float]:
    batch_minimum, batch_maximum = shared_value_range(values)
    if current is None:
        return batch_minimum, batch_maximum
    return min(current[0], batch_minimum), max(current[1], batch_maximum)


def _can_write(path: Path, policy: str) -> bool:
    if not path.exists() or policy == "replace":
        return True
    if policy == "skip":
        return False
    raise FileExistsError(
        f"Output already exists: {path}. Set output.overwrite to 'replace' or 'skip'."
    )


def _save_array(value: Any, path: Path) -> Path:
    array = (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        if path.suffix == ".npz":
            np.savez_compressed(file, data=array)
        else:
            np.save(file, array)
    return path


def _page_path(
    directory: Path,
    stem: str,
    extension: str,
    page_index: int,
    page_count: int,
) -> Path:
    if page_count == 1:
        return directory / f"{stem}.{extension}"
    return directory / f"{stem}_part-{page_index + 1:03d}.{extension}"


def _grid_page_plan(
    item_count: int,
    batch_size: int,
    items_per_grid: int | None,
) -> tuple[tuple[int, ...], int]:
    page_offsets = []
    total_pages = 0
    for start in range(0, item_count, batch_size):
        batch_count = min(batch_size, item_count - start)
        page_size = items_per_grid or batch_count
        page_offsets.append(total_pages)
        total_pages += (batch_count + page_size - 1) // page_size
    return tuple(page_offsets), total_pages


def _attention_subset(
    result: AttentionExtractionResult,
    indices: tuple[int, ...],
) -> AttentionExtractionResult:
    return AttentionExtractionResult(
        logits=result.logits,
        layers=tuple(result.layers[index] for index in indices),
        image_size=result.image_size,
    )


def _write_projection(projection: Any, path: Path, policy: str) -> Path | None:
    if not _can_write(path, policy):
        return None
    return save_patch_pca_projection(projection, path)
