from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from vision_lens.analysis.anyup import is_anyup_interpolation
from vision_lens.analysis.attention import (
    AttentionExtractionResult,
    GradCamResult,
    LayerAttentionMaps,
    extract_gradcam,
    infer_patch_grid_from_image,
)
from vision_lens.analysis.patch_pca import (
    ForegroundThreshold,
    PatchPCAProjection,
    PatchPCAResult,
    RGBFitScope,
    extract_patch_embeddings,
    extract_patch_pca,
    fit_patch_pca_projection_batches,
    load_patch_pca_projection,
    project_patch_embeddings,
    render_patch_pca_images,
    save_patch_pca_projection,
)
from vision_lens.config import (
    AttentionAnalysisConfig,
    ColormapSpec,
    OutputConfig,
    PatchPCAAnalysisConfig,
    RolloutAnalysisConfig,
    VisionLensConfig,
    VisualizationConfig,
    validate_config,
)
from vision_lens.media.processing import (
    InputBatch,
    build_batch_preprocessor,
    iter_input_batches,
    preprocess_batch,
)
from vision_lens.models import LoadedModel, load_model
from vision_lens.output.artifacts import (
    PATCH_PCA_GRID_STEM,
    attention_image_stem,
    attention_images_grid_stem,
    attention_layers_grid_stem,
    check_artifact_overwrite,
    gradcam_grid_stem,
    gradcam_image_stem,
    grid_page_path,
    image_artifact_path,
    named_artifact_path,
    rollout_grid_stem,
    rollout_image_stem,
    unique_input_labels,
)
from vision_lens.output.manifest import can_write_output as _can_write
from vision_lens.output.manifest import write_run_manifest
from vision_lens.output.visualization import (
    make_image_comparison_grid,
    make_layer_comparison_grid,
    overlay_attention,
    render_heatmap,
    save_grid,
    save_image,
    shared_value_range,
)
from vision_lens.pipeline.progress import status, track_image_batches, track_units
from vision_lens.pipeline.runtime import anyup_guidance, apply_seed

ROLLOUT_GRID_MAX_COLUMNS = 4


def _load_model_with_status(config: VisionLensConfig) -> LoadedModel:
    image_count = len(config.input.paths)
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
            config.input.paths,
            labels,
            batch_size=config.runtime.batch_size,
            workers=config.runtime.workers,
        ),
        total=len(config.input.paths),
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


@dataclass
class _GridSeries:
    images: list[Any]
    labels: list[str]
    seen: int = 0
    page_index: int = 0


class _FigureGridCollector:
    """Collect comparison tiles across inference batches before writing pages."""

    def __init__(
        self,
        *,
        total_items: int,
        output_dir: Path,
        output: OutputConfig,
        visualization: VisualizationConfig,
        grid_format: str,
    ) -> None:
        self.total_items = total_items
        self.output_dir = output_dir
        self.output = output
        self.visualization = visualization
        self.grid_format = grid_format
        self.page_size = visualization.items_per_grid or total_items
        self.page_count = (total_items + self.page_size - 1) // self.page_size
        self.series: dict[str, _GridSeries] = {}
        self.output_paths: list[Path] = []

    def add(
        self,
        stem: str,
        images: list[Any],
        labels: list[str],
    ) -> None:
        if len(images) != len(labels):
            raise ValueError("Grid images and labels must have the same length.")
        series = self.series.setdefault(stem, _GridSeries([], []))
        for image, label in zip(images, labels, strict=True):
            series.images.append(image)
            series.labels.append(label)
            series.seen += 1
            if len(series.images) == self.page_size or series.seen == self.total_items:
                self._write_page(stem, series)

    def finish(self) -> tuple[Path, ...]:
        incomplete = [
            stem
            for stem, series in self.series.items()
            if series.seen != self.total_items or series.images
        ]
        if incomplete:
            names = ", ".join(sorted(incomplete))
            raise RuntimeError(f"Incomplete grid series: {names}.")
        return tuple(self.output_paths)

    def _write_page(self, stem: str, series: _GridSeries) -> None:
        output_path = grid_page_path(
            self.output_dir,
            stem,
            self.grid_format,
            series.page_index,
            self.page_count,
        )
        if _can_write(output_path, self.output.overwrite):
            save_grid(
                series.images,
                labels=series.labels,
                output_path=output_path,
                columns=self.visualization.columns,
                tile_size=self.visualization.tile_size,
                spacing=self.visualization.spacing,
                padding=self.visualization.padding,
                show_labels=(
                    True
                    if self.visualization.labels is None
                    else self.visualization.labels
                ),
                background=self.visualization.background,
                dpi=self.visualization.dpi,
            )
            self.output_paths.append(output_path)
        series.images.clear()
        series.labels.clear()
        series.page_index += 1


class _PatchPCAGridCollector:
    """Collect rendered PCA tiles across inference batches before writing pages."""

    def __init__(
        self,
        *,
        total_items: int,
        output_dir: Path,
        output: OutputConfig,
        visualization: VisualizationConfig,
    ) -> None:
        self.total_items = total_items
        self.output_dir = output_dir
        self.output = output
        self.visualization = visualization
        self.page_size = visualization.items_per_grid or total_items
        self.page_count = (total_items + self.page_size - 1) // self.page_size
        self.series = _GridSeries([], [])
        self.output_paths: list[Path] = []

    def add(self, images: list[Any], labels: list[str]) -> None:
        if len(images) != len(labels):
            raise ValueError("Grid images and labels must have the same length.")
        for image, label in zip(images, labels, strict=True):
            self.series.images.append(image)
            self.series.labels.append(label)
            self.series.seen += 1
            if (
                len(self.series.images) == self.page_size
                or self.series.seen == self.total_items
            ):
                self._write_page()

    def finish(self) -> tuple[Path, ...]:
        if self.series.seen != self.total_items or self.series.images:
            raise RuntimeError("Incomplete patch PCA grid series.")
        return tuple(self.output_paths)

    def _write_page(self) -> None:
        output_path = grid_page_path(
            self.output_dir,
            PATCH_PCA_GRID_STEM,
            self.visualization.grid_format,
            self.series.page_index,
            self.page_count,
        )
        if _can_write(output_path, self.output.overwrite):
            save_grid(
                self.series.images,
                labels=self.series.labels,
                output_path=output_path,
                columns=self.visualization.columns,
                tile_size=self.visualization.tile_size,
                spacing=self.visualization.spacing,
                padding=self.visualization.padding,
                show_labels=False,
                background=self.visualization.background,
                dpi=self.visualization.dpi,
            )
            self.output_paths.append(output_path)
        self.series.images.clear()
        self.series.labels.clear()
        self.series.page_index += 1


def run_patch_pca_from_config(
    config: VisionLensConfig,
) -> PatchPCAPipelineResult:
    validate_config(config)
    if not isinstance(config.analysis, PatchPCAAnalysisConfig):
        raise ValueError(
            f"Expected analysis.method='patch_pca', got {config.analysis.method!r}."
        )
    if config.model.architecture != "vit":
        raise ValueError("Patch PCA pipeline expects a ViT model config.")
    if (
        config.analysis.projection != "load"
        and config.analysis.foreground_separation is None
    ):
        raise ValueError("Image patch PCA requires foreground analysis settings.")
    fit_settings = _pca_fit_settings(config)

    started_at = datetime.now(UTC)
    check_artifact_overwrite(config)
    labels = unique_input_labels(config.input.paths)
    projection = (
        load_patch_pca_projection(config.analysis.projection_path)
        if config.analysis.projection == "load"
        else None
    )
    apply_seed(config.runtime.seed)
    loaded_model = _load_model_with_status(config)
    transform = build_batch_preprocessor(loaded_model, config.preprocessing)
    patch_grid = _patch_grid(loaded_model)
    projection_only = (
        config.analysis.projection == "fit"
        and config.analysis.save_projection is not None
        and not config.output.heatmaps
        and not config.output.grids
        and not config.output.raw_arrays
    )
    if projection_only:
        projection = _fit_image_pca_projection(
            config,
            labels,
            loaded_model,
            transform,
            patch_grid,
            fit_settings,
        )
        assert config.analysis.save_projection is not None
        projection_path = _write_projection(
            projection,
            config.analysis.save_projection,
            config.output.overwrite,
        )
        output_paths = () if projection_path is None else (projection_path,)
        write_run_manifest(
            config,
            loaded_model,
            labels,
            output_paths,
            started_at=started_at,
        )
        return PatchPCAPipelineResult(
            config=config,
            loaded_model=loaded_model,
            patch_pca=None,
            output_paths=output_paths,
            processed_inputs=len(config.input.paths),
        )
    grid_collector = _PatchPCAGridCollector(
        total_items=len(config.input.paths),
        output_dir=config.output.directory,
        output=config.output,
        visualization=config.visualization,
    )
    output_paths: list[Path] = []
    retained_patch_pca = None
    render_pca_images = config.output.heatmaps or config.output.grids

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
                guidance_image=(
                    anyup_guidance(config, loaded_model, batch.inputs)
                    if render_pca_images
                    else None
                ),
                output_size=_analysis_output_size(config, loaded_model),
                anyup_query_chunk_size=(config.visualization.anyup_query_chunk_size),
                render_images=render_pca_images,
            )
            if len(config.input.paths) <= config.runtime.batch_size:
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
                    grid_collector=grid_collector,
                )
            )
    elif len(config.input.paths) <= config.runtime.batch_size:
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
                foreground_separation=fit_settings[0],
                foreground_threshold=fit_settings[1],
                foreground_side=fit_settings[2],
                rgb_fit_scope=fit_settings[3],
                interpolation=config.visualization.interpolation,
                guidance_image=(
                    anyup_guidance(config, loaded_model, batch.inputs)
                    if render_pca_images
                    else None
                ),
                output_size=_analysis_output_size(config, loaded_model),
                anyup_query_chunk_size=(config.visualization.anyup_query_chunk_size),
                render_images=render_pca_images,
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
                    grid_collector=grid_collector,
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
                guidance = (
                    anyup_guidance(config, loaded_model, batch.inputs)
                    if render_pca_images
                    else None
                )
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
                foreground_separation=fit_settings[0],
                foreground_threshold=fit_settings[1],
                foreground_side=fit_settings[2],
                rgb_fit_scope=fit_settings[3],
            )
            for (
                _batch_index,
                batch_paths,
                batch_labels,
                batch_sizes,
                staged_path,
                guidance_path,
            ) in track_units(
                staged_batches,
                total=len(config.input.paths),
                description="Project PCA",
                unit="image",
                size=lambda item: len(item[1]),
            ):
                embeddings = np.load(staged_path, allow_pickle=False)
                patch_pca = project_patch_embeddings(
                    embeddings,
                    patch_grid=patch_grid,
                    image_size=_analysis_output_size(config, loaded_model),
                    foreground_separation=fit_settings[0],
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
                    render_images=render_pca_images,
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
                        grid_collector=grid_collector,
                    )
                )

    if config.output.grids:
        output_paths.extend(grid_collector.finish())
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
        processed_inputs=len(config.input.paths),
    )


def run_vit_rollout_comparison_from_config(
    config: VisionLensConfig,
) -> PipelineResult:
    validate_config(config)
    if not isinstance(config.analysis, RolloutAnalysisConfig):
        raise ValueError(
            f"Expected analysis.method='rollout', got {config.analysis.method!r}."
        )
    from vision_lens.analysis.attention import extract_attention_rollout

    started_at = datetime.now(UTC)
    check_artifact_overwrite(config)
    labels = unique_input_labels(config.input.paths)
    apply_seed(config.runtime.seed)
    loaded_model = _load_model_with_status(config)
    transform = build_batch_preprocessor(loaded_model, config.preprocessing)
    rendering = config.visualization
    if rendering.normalization == "shared" and _renders_maps(config):
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
                layers=config.analysis.layers,
                normalize=False,
                interpolation=config.visualization.interpolation,
                guidance_image=anyup_guidance(config, loaded_model, batch.inputs),
                output_size=_analysis_output_size(config, loaded_model),
                anyup_query_chunk_size=(config.visualization.anyup_query_chunk_size),
            )
            fitted_attention = (
                _extract_attention(
                    loaded_model=loaded_model,
                    inputs=batch.inputs,
                    config=config,
                )
                if config.output.grids
                else None
            )
            fitted_maps = [layer.maps for layer in fitted_rollout.layers]
            if fitted_attention is not None:
                fitted_maps[:0] = [layer.maps for layer in fitted_attention.layers]
            normalization_range = _extend_value_range(
                normalization_range,
                fitted_maps,
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
            layers=config.analysis.layers,
            normalize=False,
            interpolation=config.visualization.interpolation,
            guidance_image=anyup_guidance(config, loaded_model, batch.inputs),
            output_size=_analysis_output_size(config, loaded_model),
            anyup_query_chunk_size=config.visualization.anyup_query_chunk_size,
        )
        layer_attention = (
            _extract_attention(
                loaded_model=loaded_model,
                inputs=batch.inputs,
                config=config,
            )
            if config.output.grids
            else None
        )
        if len(config.input.paths) <= config.runtime.batch_size:
            retained_rollout = rollout
        output_paths.extend(
            export_rollout_comparison_outputs(
                images=_visualization_images(batch, rendering),
                image_paths=input_batch.paths,
                layer_attention=layer_attention,
                rollout=rollout,
                output_dir=config.output.directory,
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
        processed_inputs=len(config.input.paths),
    )


def run_gradcam_from_config(config: VisionLensConfig) -> GradCamPipelineResult:
    validate_config(config)
    if config.model.architecture != "cnn":
        raise ValueError("Grad-CAM pipeline expects a CNN model config.")

    started_at = datetime.now(UTC)
    check_artifact_overwrite(config)
    labels = unique_input_labels(config.input.paths)
    apply_seed(config.runtime.seed)
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
    if rendering.normalization == "shared" and _renders_maps(config):
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
                guidance_image=anyup_guidance(config, loaded_model, batch.inputs),
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
    grid_collector = _FigureGridCollector(
        total_items=len(config.input.paths),
        output_dir=config.output.directory,
        output=config.output,
        visualization=rendering,
        grid_format=config.visualization.grid_format,
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
            normalize=False,
            interpolation=config.visualization.interpolation,
            guidance_image=anyup_guidance(config, loaded_model, batch.inputs),
            output_size=_analysis_output_size(config, loaded_model),
            anyup_query_chunk_size=config.visualization.anyup_query_chunk_size,
        )
        if len(config.input.paths) <= config.runtime.batch_size:
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
                grid_collector=grid_collector,
            )
        )
    if config.output.grids:
        output_paths.extend(grid_collector.finish())
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
        processed_inputs=len(config.input.paths),
    )


def run_vit_attention_from_config(config: VisionLensConfig) -> PipelineResult:
    validate_config(config)
    if not isinstance(config.analysis, AttentionAnalysisConfig):
        raise ValueError(
            f"Expected analysis.method='attention', got {config.analysis.method!r}."
        )

    started_at = datetime.now(UTC)
    check_artifact_overwrite(config)
    labels = unique_input_labels(config.input.paths)
    apply_seed(config.runtime.seed)
    loaded_model = _load_model_with_status(config)
    transform = build_batch_preprocessor(loaded_model, config.preprocessing)
    rendering = config.visualization
    if rendering.normalization == "shared" and _renders_maps(config):
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
    grid_collector = _FigureGridCollector(
        total_items=len(config.input.paths),
        output_dir=config.output.directory,
        output=config.output,
        visualization=rendering,
        grid_format=config.visualization.grid_format,
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
        if len(config.input.paths) <= config.runtime.batch_size:
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
                grid_collector=grid_collector,
            )
        )
    if config.output.grids:
        output_paths.extend(grid_collector.finish())
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
        processed_inputs=len(config.input.paths),
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
    grid_collector: _FigureGridCollector | None = None,
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
                stem = attention_image_stem(
                    labels[image_index], layer.layer_index, suffix
                )

                if output.heatmaps:
                    path = image_artifact_path(
                        output_dir, stem, "heatmap", output.image_format
                    )
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
                    path = image_artifact_path(
                        output_dir, stem, "overlay", output.image_format
                    )
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
                    path = named_artifact_path(output_dir, stem, output.raw_format)
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
                stem = attention_layers_grid_stem(labels[image_index], suffix)
                output_path = grid_page_path(
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
            stem = attention_images_grid_stem(layer.layer_index, suffix)
            if grid_collector is not None:
                overlays = [
                    overlay_attention(
                        image,
                        image_map,
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
                    for image, image_map in zip(images, image_maps, strict=True)
                ]
                grid_collector.add(stem, overlays, labels)
                continue
            image_pages = _chunks(
                tuple(range(len(images))), visualization.items_per_grid
            )
            for page_index, indices in enumerate(image_pages):
                output_path = grid_page_path(
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
    grid_collector: _PatchPCAGridCollector | None = None,
) -> tuple[Path, ...]:
    output = output_config or OutputConfig(output_dir)
    visualization = visualization_config or VisualizationConfig()
    labels = input_labels or tuple(path.stem for path in image_paths)
    if len(labels) != len(image_paths):
        raise ValueError("input_labels and image_paths must have the same length.")
    renders_images = output.heatmaps or output.grids
    if renders_images and len(patch_pca.images) != len(image_paths):
        raise ValueError(
            "Rendered patch PCA images and image_paths must have the same length."
        )
    if input_sizes is not None and len(input_sizes) != len(image_paths):
        raise ValueError("input_sizes and image_paths must have the same length.")
    if renders_images and visualization.output_size == "match" and input_sizes is None:
        raise ValueError("input_sizes are required when output_size is 'match'.")
    rendered_images = (
        tuple(
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
        if renders_images
        else ()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    if output.heatmaps:
        for label, image in zip(labels, rendered_images, strict=True):
            output_path = image_artifact_path(
                output_dir, label, "patch_pca", output.image_format
            )
            if _can_write(output_path, output.overwrite):
                output_paths.append(save_image(image, output_path))
    if output.raw_arrays:
        for index, label in enumerate(labels):
            embedding_path = image_artifact_path(
                output_dir, label, "patch_embeddings", output.raw_format
            )
            mask_path = image_artifact_path(
                output_dir, label, "foreground_mask", output.raw_format
            )
            if _can_write(embedding_path, output.overwrite):
                output_paths.append(
                    _save_array(patch_pca.patch_embeddings[index], embedding_path)
                )
            if _can_write(mask_path, output.overwrite):
                output_paths.append(
                    _save_array(patch_pca.foreground_mask[index], mask_path)
                )
    if not output.grids:
        return tuple(output_paths)

    if grid_collector is not None:
        grid_collector.add(list(rendered_images), list(labels))
        return tuple(output_paths)

    indices_pages = _chunks(
        tuple(range(len(patch_pca.images))),
        visualization.items_per_grid,
    )
    for page_index, indices in enumerate(indices_pages):
        page_images = [rendered_images[index] for index in indices]
        page_labels = [labels[index] for index in indices]
        output_path = grid_page_path(
            output_dir,
            PATCH_PCA_GRID_STEM,
            visualization.grid_format,
            grid_page_offset + page_index,
            total_grid_pages or len(indices_pages),
        )
        if _can_write(output_path, output.overwrite):
            save_grid(
                page_images,
                labels=page_labels,
                output_path=output_path,
                columns=visualization.columns,
                tile_size=visualization.tile_size,
                spacing=visualization.spacing,
                padding=visualization.padding,
                show_labels=False,
                background=visualization.background,
                dpi=visualization.dpi,
            )
            output_paths.append(output_path)
    return tuple(output_paths)


def export_rollout_comparison_outputs(
    images: list[Any],
    image_paths: tuple[Path, ...],
    layer_attention: AttentionExtractionResult | None,
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
    normalization_maps = [layer.maps for layer in rollout.layers]
    if layer_attention is not None:
        normalization_maps[:0] = [layer.maps for layer in layer_attention.layers]
    normalization_range = _rendering_range(visualization, normalization_maps)

    for image_index, image in enumerate(images):
        for rollout_layer in rollout.layers:
            rollout_for_image = _layer_for_image(rollout_layer, image_index)
            stem = rollout_image_stem(labels[image_index], rollout_layer.layer_index)
            if output.heatmaps:
                path = image_artifact_path(
                    output_dir, stem, "heatmap", output.image_format
                )
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
                path = image_artifact_path(
                    output_dir, stem, "overlay", output.image_format
                )
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
                path = named_artifact_path(output_dir, stem, output.raw_format)
                if _can_write(path, output.overwrite):
                    output_paths.append(_save_array(rollout_for_image.maps[0, 0], path))

        if output.grids:
            if layer_attention is None:
                raise ValueError("Rollout grids require layer-attention maps.")
            layer_pages = _chunks(
                tuple(range(len(rollout.layers))),
                visualization.items_per_grid,
            )
            for page_index, indices in enumerate(layer_pages):
                layer_page = _attention_subset(layer_attention, indices)
                rollout_page = _attention_subset(rollout, indices)
                grid_path = grid_page_path(
                    output_dir,
                    rollout_grid_stem(labels[image_index]),
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
    grid_collector: _FigureGridCollector | None = None,
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
        stem = gradcam_image_stem(labels[image_index])
        if output.heatmaps:
            path = image_artifact_path(output_dir, stem, "heatmap", output.image_format)
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
            path = image_artifact_path(output_dir, stem, "overlay", output.image_format)
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
            path = named_artifact_path(output_dir, stem, output.raw_format)
            if _can_write(path, output.overwrite):
                output_paths.append(_save_array(maps[0, 0], path))

    if output.grids:
        image_maps = [_slice_batch(gradcam.maps, index) for index in range(len(images))]
        if grid_collector is not None:
            overlays = [
                overlay_attention(
                    image,
                    image_map,
                    alpha=alpha,
                    alpha_curve_steepness=(visualization.overlay_alpha_curve_steepness),
                    alpha_curve_midpoint=(visualization.overlay_alpha_curve_midpoint),
                    cmap=cmap,
                    normalization=visualization.normalization,
                    normalization_range=normalization_range,
                )
                for image, image_map in zip(images, image_maps, strict=True)
            ]
            grid_collector.add(gradcam_grid_stem(), overlays, labels)
            return tuple(output_paths)
        image_pages = _chunks(tuple(range(len(images))), visualization.items_per_grid)
        for page_index, indices in enumerate(image_pages):
            grid_path = grid_page_path(
                output_dir,
                gradcam_grid_stem(),
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
    from vision_lens.analysis.attention import extract_attention_maps

    return extract_attention_maps(
        loaded_model.model,
        inputs,
        loaded_model.metadata,
        layers=config.analysis.layers,
        heads=config.analysis.heads,
        head_fusion=config.analysis.head_fusion,
        normalize=False,
        interpolation=config.visualization.interpolation,
        guidance_image=anyup_guidance(config, loaded_model, inputs),
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


def _pca_fit_settings(
    config: VisionLensConfig,
) -> tuple[
    bool,
    ForegroundThreshold,
    Literal["high", "low"],
    RGBFitScope,
]:
    foreground_separation = config.analysis.foreground_separation
    if foreground_separation is not True:
        return False, 0.5, "high", "all"
    if any(
        value is None
        for value in (
            config.analysis.foreground_threshold,
            config.analysis.foreground_side,
            config.analysis.rgb_fit_scope,
        )
    ):
        raise ValueError("Foreground-separated patch PCA requires fit settings.")
    return (
        True,
        config.analysis.foreground_threshold,
        config.analysis.foreground_side,
        config.analysis.rgb_fit_scope,
    )


def _fit_image_pca_projection(
    config: VisionLensConfig,
    labels: tuple[str, ...],
    loaded_model: LoadedModel,
    transform: Any,
    patch_grid: tuple[int, int],
    fit_settings: tuple[
        bool,
        ForegroundThreshold,
        Literal["high", "low"],
        RGBFitScope,
    ],
) -> PatchPCAProjection:
    with TemporaryDirectory(prefix="vision-lens-pca-fit-") as temporary_directory:
        staged_paths: list[Path] = []
        for input_batch in _tracked_input_batches(
            config, labels, "Extract PCA embeddings"
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
            path = Path(temporary_directory) / f"batch-{input_batch.index}.npy"
            _save_array(embeddings, path)
            staged_paths.append(path)

        def embedding_batches() -> Any:
            for path in staged_paths:
                yield np.load(path, allow_pickle=False)

        status("Fitting PCA projection")
        return fit_patch_pca_projection_batches(
            embedding_batches,
            foreground_separation=fit_settings[0],
            foreground_threshold=fit_settings[1],
            foreground_side=fit_settings[2],
            rgb_fit_scope=fit_settings[3],
        )


def _rendering_range(
    visualization: VisualizationConfig,
    values: list[Any],
) -> tuple[float, float] | None:
    if visualization.normalization == "per_map":
        return None
    if visualization.normalization == "fixed":
        return visualization.normalization_range
    return shared_value_range(values)


def _renders_maps(config: VisionLensConfig) -> bool:
    return config.output.heatmaps or config.output.overlays or config.output.grids


def _extend_value_range(
    current: tuple[float, float] | None,
    values: list[Any],
) -> tuple[float, float]:
    batch_minimum, batch_maximum = shared_value_range(values)
    if current is None:
        return batch_minimum, batch_maximum
    return min(current[0], batch_minimum), max(current[1], batch_maximum)


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
