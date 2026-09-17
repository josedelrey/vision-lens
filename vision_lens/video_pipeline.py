from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from PIL import Image

from vision_lens.artifacts import (
    check_artifact_overwrite,
    map_stream_name,
    rendered_stream_name,
    video_artifact_path,
    video_raw_batch_path,
    video_run_layouts,
)
from vision_lens.attention import (
    AttentionExtractionResult,
    GradCamResult,
    extract_attention_maps,
    extract_attention_rollout,
    extract_gradcam,
    infer_patch_grid_from_image,
)
from vision_lens.config import (
    AttentionAnalysisConfig,
    GradCAMAnalysisConfig,
    PatchPCAAnalysisConfig,
    RolloutAnalysisConfig,
    VisionLensConfig,
    validate_config,
)
from vision_lens.feature_pca import (
    PatchPCAProjection,
    extract_patch_embeddings,
    fit_patch_pca_projection_batches,
    load_patch_pca_projection,
    project_patch_embeddings,
    save_patch_pca_projection,
)
from vision_lens.manifest import can_write_output as _can_write
from vision_lens.manifest import write_run_manifest
from vision_lens.models import LoadedModel, load_model
from vision_lens.processing import (
    InputBatch,
    build_batch_preprocessor,
    preprocess_batch,
)
from vision_lens.progress import status, track_video_batches
from vision_lens.runtime import anyup_guidance, apply_seed
from vision_lens.video import (
    SampledVideoFrame,
    VideoMetadata,
    VideoWriter,
    close_video_writers,
    estimated_sample_count,
    iter_video_batches,
    probe_video,
    representative_frame_indices,
    require_video_dependencies,
    resolve_sampling_rate,
    resolved_output_resolution,
)
from vision_lens.visualization import overlay_attention, render_heatmap


@dataclass(frozen=True)
class VideoPipelineResult:
    config: VisionLensConfig
    loaded_model: LoadedModel
    source: VideoMetadata
    output_paths: tuple[Path, ...]
    processed_frames: int
    frame_rate: float
    duration: float


@dataclass(frozen=True)
class VideoBatchPipelineResult:
    config: VisionLensConfig
    videos: tuple[VideoPipelineResult, ...]
    output_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _MapStream:
    name: str
    maps: Any


def run_video_from_config(
    config: VisionLensConfig,
) -> VideoPipelineResult | VideoBatchPipelineResult:
    validate_config(config)
    if config.video is None:
        raise ValueError("Video pipeline requires a video configuration section.")
    if len(config.input.paths) == 1:
        return _run_single_video_from_config(config)

    video_configs = []
    for layout in video_run_layouts(config):
        analysis = config.analysis
        if isinstance(analysis, PatchPCAAnalysisConfig):
            analysis = replace(
                analysis,
                save_projection=layout.projection_path,
            )
        video_config = replace(
            config,
            input=replace(
                config.input,
                paths=(layout.source_path,),
            ),
            output=replace(config.output, directory=layout.output_directory),
            analysis=analysis,
        )
        check_artifact_overwrite(video_config)
        video_configs.append(video_config)

    require_video_dependencies()
    apply_seed(config.runtime.seed)
    status(f"Loading model {config.model.name}")
    loaded_model = load_model(config, dynamic_img_size=True)
    status(f"Model ready on {loaded_model.metadata.device}")
    results = [
        _run_single_video_from_config(video_config, loaded_model=loaded_model)
        for video_config in video_configs
    ]

    return VideoBatchPipelineResult(
        config=config,
        videos=tuple(results),
        output_paths=tuple(path for result in results for path in result.output_paths),
    )


def _run_single_video_from_config(
    config: VisionLensConfig,
    *,
    loaded_model: LoadedModel | None = None,
) -> VideoPipelineResult:
    assert config.video is not None

    started_at = datetime.now(timezone.utc)
    check_artifact_overwrite(config)
    if loaded_model is None:
        require_video_dependencies()
    source_path = config.input.paths[0]
    source = probe_video(source_path)
    requested_config = config
    frame_rate = resolve_sampling_rate(
        config.video.sampling_rate, source.source_frame_rate
    )
    config = replace(config, video=replace(config.video, sampling_rate=frame_rate))
    sample_count = estimated_sample_count(source, config.video)
    status(
        f"{config.analysis.method}: {source_path.name} "
        f"at {config.video.sampling_rate:g} FPS"
    )
    apply_seed(config.runtime.seed)
    if loaded_model is None:
        status(f"Loading model {config.model.name}")
        loaded_model = load_model(config, dynamic_img_size=True)
        status(f"Model ready on {loaded_model.metadata.device}")
    loaded_model = _video_model_for_source(loaded_model, config, source)
    _validate_gradcam_class(config, loaded_model)
    configured_output_size = config.visualization.output_size
    resolution = resolved_output_resolution(
        source,
        configured_output_size if isinstance(configured_output_size, tuple) else None,
        default=(
            None
            if configured_output_size == "match"
            else (
                loaded_model.metadata.image_size[1],
                loaded_model.metadata.image_size[0],
            )
        ),
    )
    analysis_output_size = (
        (resolution[1], resolution[0])
        if isinstance(configured_output_size, tuple)
        else loaded_model.metadata.image_size
    )
    transform = build_batch_preprocessor(
        loaded_model,
        replace(config.preprocessing, resize="stretch", crop="none", pad="none"),
    )
    projection = _video_pca_projection(config, loaded_model, transform, source)
    pca_analysis = (
        config.analysis if isinstance(config.analysis, PatchPCAAnalysisConfig) else None
    )
    projection_only = (
        pca_analysis is not None
        and pca_analysis.projection == "fit"
        and pca_analysis.save_projection is not None
        and not config.output.heatmaps
        and not config.output.raw_arrays
    )
    if projection_only:
        assert projection is not None
        assert pca_analysis is not None
        assert pca_analysis.save_projection is not None
        output_paths: tuple[Path, ...] = ()
        if _can_write(pca_analysis.save_projection, config.output.overwrite):
            save_patch_pca_projection(projection, pca_analysis.save_projection)
            output_paths = (pca_analysis.save_projection,)
        write_run_manifest(
            requested_config,
            loaded_model,
            (source_path.stem,),
            output_paths,
            started_at=started_at,
        )
        return VideoPipelineResult(
            config=config,
            loaded_model=loaded_model,
            source=source,
            output_paths=output_paths,
            processed_frames=0,
            frame_rate=frame_rate,
            duration=0.0,
        )
    normalization_range = _video_normalization_range(
        config,
        loaded_model,
        transform,
        source,
        analysis_output_size,
    )
    exports = _VideoExports(config, source_path.stem, resolution)
    smoothing_state: dict[str, Any] = {}
    processed_frames = 0
    try:
        for frame_batch in track_video_batches(
            iter_video_batches(
                source_path,
                config.video,
                config.runtime.batch_size,
            ),
            total=sample_count,
            description="Analyze video",
        ):
            batch = _preprocess_frames(
                frame_batch.index,
                frame_batch.frames,
                source_path,
                transform,
                loaded_model,
                include_display_images=False,
            )
            original_frames = tuple(frame.image for frame in frame_batch.frames)
            if config.analysis.method == "patch_pca":
                assert projection is not None
                embeddings = extract_patch_embeddings(
                    loaded_model.model,
                    batch.inputs,
                    _patch_grid(loaded_model),
                )
                pca = project_patch_embeddings(
                    embeddings,
                    patch_grid=_patch_grid(loaded_model),
                    image_size=(resolution[1], resolution[0]),
                    foreground_separation=False,
                    projection=projection,
                    interpolation=config.visualization.interpolation,
                    anyup_query_chunk_size=(
                        config.visualization.anyup_query_chunk_size
                    ),
                    guidance_image=(
                        anyup_guidance(config, loaded_model, batch.inputs)
                        if config.output.heatmaps
                        else None
                    ),
                    render_images=config.output.heatmaps,
                )
                pca_images = (
                    _smooth_images(
                        pca.images,
                        smoothing_state,
                        "patch-pca",
                        config.video.temporal_smoothing,
                    )
                    if config.output.heatmaps
                    else ()
                )
                exports.write_pca_batch(
                    frame_batch.frames,
                    pca_images,
                    pca.rgb_patches,
                    frame_batch.index,
                )
            else:
                analysis = _analyze_maps(
                    config,
                    loaded_model,
                    batch.inputs,
                    analysis_output_size,
                )
                streams = _smooth_streams(
                    _map_streams(config, analysis),
                    smoothing_state,
                    config.video.temporal_smoothing,
                )
                exports.write_map_batch(
                    frame_batch.frames,
                    original_frames,
                    streams,
                    frame_batch.index,
                    normalization_range,
                )
            processed_frames += len(frame_batch.frames)
    finally:
        exports.close()

    if processed_frames == 0:
        raise ValueError("The configured video time range selected no frames.")

    output_paths = list(exports.output_paths)
    if (
        pca_analysis is not None
        and pca_analysis.save_projection is not None
        and projection is not None
    ):
        if _can_write(pca_analysis.save_projection, config.output.overwrite):
            save_patch_pca_projection(projection, pca_analysis.save_projection)
            output_paths.insert(0, pca_analysis.save_projection)
    output_paths_tuple = tuple(output_paths)
    encoded_duration = processed_frames / config.video.sampling_rate
    write_run_manifest(
        requested_config,
        loaded_model,
        (source_path.stem,),
        output_paths_tuple,
        started_at=started_at,
        run_details={
            "media": "video",
            "source_width": source.width,
            "source_height": source.height,
            "source_duration": source.duration,
            "source_frame_rate": source.source_frame_rate,
            "sampled_frames": processed_frames,
            "sampling_rate": config.video.sampling_rate,
            "encoded_duration": encoded_duration,
            "output_resolution": list(resolution),
            "audio": "omitted",
        },
    )
    return VideoPipelineResult(
        config=requested_config,
        loaded_model=loaded_model,
        source=source,
        output_paths=output_paths_tuple,
        processed_frames=processed_frames,
        frame_rate=config.video.sampling_rate,
        duration=encoded_duration,
    )


class _VideoExports:
    def __init__(
        self,
        config: VisionLensConfig,
        source_stem: str,
        resolution: tuple[int, int],
    ) -> None:
        assert config.video is not None
        self.config = config
        self.source_stem = source_stem
        self.resolution = resolution
        self._writers: dict[Path, VideoWriter | None] = {}
        self._output_paths: list[Path] = []

    @property
    def output_paths(self) -> tuple[Path, ...]:
        return tuple(self._output_paths)

    def write_map_batch(
        self,
        frames: tuple[SampledVideoFrame, ...],
        originals: tuple[Image.Image, ...],
        streams: tuple[_MapStream, ...],
        batch_index: int,
        normalization_range: tuple[float, float] | None,
    ) -> None:
        normalization = self.config.visualization.normalization
        if normalization == "shared":
            normalization = "fixed"
        for stream in streams:
            for frame_index, original in enumerate(originals):
                if self.config.output.heatmaps:
                    heatmap = render_heatmap(
                        stream.maps,
                        cmap=self.config.visualization.render_cmap,
                        batch_index=frame_index,
                        normalization=normalization,
                        normalization_range=normalization_range,
                    )
                    self._write_video(
                        rendered_stream_name(stream.name, "heatmap"), heatmap
                    )
                if self.config.output.overlays:
                    display_frame = _frame_at_output_size(original, self.resolution)
                    overlay = overlay_attention(
                        display_frame,
                        stream.maps,
                        alpha=self.config.visualization.overlay_alpha,
                        alpha_curve_steepness=(
                            self.config.visualization.overlay_alpha_curve_steepness
                        ),
                        alpha_curve_midpoint=(
                            self.config.visualization.overlay_alpha_curve_midpoint
                        ),
                        cmap=self.config.visualization.render_cmap,
                        batch_index=frame_index,
                        normalization=normalization,
                        normalization_range=normalization_range,
                    )
                    self._write_video(
                        rendered_stream_name(stream.name, "overlay"), overlay
                    )
            if self.config.output.raw_arrays:
                self._write_raw_batch(
                    stream.name,
                    stream.maps,
                    frames,
                    batch_index,
                )

    def write_pca_batch(
        self,
        frames: tuple[SampledVideoFrame, ...],
        pca_images: tuple[Image.Image, ...],
        rgb_patches: Any,
        batch_index: int,
    ) -> None:
        for pca_image in pca_images:
            if self.config.output.heatmaps:
                self._write_video("patch_pca", pca_image)
        if self.config.output.raw_arrays:
            assert rgb_patches is not None
            self._write_raw_batch(
                "patch_pca_rgb",
                rgb_patches,
                frames,
                batch_index,
            )

    def close(self) -> None:
        close_video_writers(
            tuple(writer for writer in self._writers.values() if writer is not None)
        )

    def _write_video(self, name: str, image: Image.Image) -> None:
        path = video_artifact_path(
            self.config.output.directory,
            self.source_stem,
            name,
        )
        resolution = self.resolution
        if self.config.visualization.interpolation == "nearest" and name.endswith(
            "_heatmap"
        ):
            image = image.resize(resolution, Image.Resampling.NEAREST)
        writer = self._writer(path, resolution)
        if writer is not None:
            writer.write(image)

    def _writer(self, path: Path, resolution: tuple[int, int]) -> VideoWriter | None:
        if path not in self._writers:
            if not _can_write(path, self.config.output.overwrite):
                self._writers[path] = None
            else:
                assert self.config.video is not None
                self._writers[path] = VideoWriter(
                    path,
                    frame_rate=self.config.video.sampling_rate,
                    resolution=resolution,
                    codec=self.config.video.codec,
                )
                self._output_paths.append(path)
        return self._writers[path]

    def _write_raw_batch(
        self,
        name: str,
        values: Any,
        frames: tuple[SampledVideoFrame, ...],
        batch_index: int,
    ) -> None:
        extension = self.config.output.raw_format
        path = video_raw_batch_path(
            self.config.output.directory,
            self.source_stem,
            name,
            batch_index,
            extension,
        )
        if not _can_write(path, self.config.output.overwrite):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        array = _as_numpy(values)
        timestamps = np.asarray([frame.timestamp for frame in frames])
        with path.open("wb") as file:
            if extension == "npz":
                np.savez_compressed(file, data=array, timestamps=timestamps)
            else:
                np.save(file, array)
        self._output_paths.append(path)


def _video_pca_projection(
    config: VisionLensConfig,
    loaded_model: LoadedModel,
    transform: Any,
    source: VideoMetadata,
) -> PatchPCAProjection | None:
    if not isinstance(config.analysis, PatchPCAAnalysisConfig):
        return None
    analysis = config.analysis
    if analysis.projection == "load":
        return load_patch_pca_projection(analysis.projection_path)

    assert config.video is not None
    sample_count = estimated_sample_count(source, config.video)
    selected = representative_frame_indices(sample_count, config.video.pca_fit_frames)
    selected_set = None if selected is None else set(selected)
    staged_paths = []
    staged_frame_count = 0
    with TemporaryDirectory(prefix="vision-lens-video-pca-") as temporary_directory:
        for frame_batch in track_video_batches(
            iter_video_batches(
                config.input.paths[0],
                config.video,
                config.runtime.batch_size,
            ),
            total=sample_count,
            description="Scan PCA frames",
        ):
            selected_frames = tuple(
                frame
                for frame in frame_batch.frames
                if (
                    frame.index < config.video.pca_fit_frames
                    if selected_set is None
                    else frame.index in selected_set
                )
            )
            if not selected_frames:
                continue
            batch = _preprocess_frames(
                len(staged_paths),
                selected_frames,
                config.input.paths[0],
                transform,
                loaded_model,
                include_display_images=False,
            )
            embeddings = extract_patch_embeddings(
                loaded_model.model,
                batch.inputs,
                _patch_grid(loaded_model),
            )
            path = Path(temporary_directory) / f"fit-{len(staged_paths):06d}.npy"
            with path.open("wb") as file:
                np.save(file, _as_numpy(embeddings))
            staged_paths.append(path)
            staged_frame_count += len(selected_frames)
            if (
                selected_set is None
                and staged_frame_count >= config.video.pca_fit_frames
            ):
                break

        if not staged_paths:
            raise ValueError("The configured video time range selected no PCA frames.")

        def embedding_batches() -> Any:
            for path in staged_paths:
                yield np.load(path, allow_pickle=False)

        status("Fitting PCA projection")
        return fit_patch_pca_projection_batches(
            embedding_batches,
            foreground_separation=False,
            rgb_fit_scope="all",
            rgb_percentile_bounds=(0.01, 0.99),
        )


def _video_normalization_range(
    config: VisionLensConfig,
    loaded_model: LoadedModel,
    transform: Any,
    source: VideoMetadata,
    output_size: tuple[int, int],
) -> tuple[float, float] | None:
    if config.analysis.method == "patch_pca" or not (
        config.output.heatmaps or config.output.overlays
    ):
        return None
    if config.visualization.normalization == "fixed":
        return config.visualization.normalization_range
    if config.visualization.normalization == "per_map":
        return None

    assert config.video is not None
    current = None
    smoothing_state: dict[str, Any] = {}
    for frame_batch in track_video_batches(
        iter_video_batches(
            config.input.paths[0],
            config.video,
            config.runtime.batch_size,
        ),
        total=estimated_sample_count(source, config.video),
        description="Fit normalization",
    ):
        batch = _preprocess_frames(
            frame_batch.index,
            frame_batch.frames,
            config.input.paths[0],
            transform,
            loaded_model,
            include_display_images=False,
        )
        streams = _smooth_streams(
            _map_streams(
                config,
                _analyze_maps(config, loaded_model, batch.inputs, output_size),
            ),
            smoothing_state,
            config.video.temporal_smoothing,
        )
        for stream in streams:
            minimum = float(torch.as_tensor(stream.maps).min())
            maximum = float(torch.as_tensor(stream.maps).max())
            current = (
                (minimum, maximum)
                if current is None
                else (min(current[0], minimum), max(current[1], maximum))
            )
    if current is None:
        raise ValueError("The configured video time range selected no frames.")
    return current


def _analyze_maps(
    config: VisionLensConfig,
    loaded_model: LoadedModel,
    inputs: Any,
    output_size: tuple[int, int],
) -> AttentionExtractionResult | GradCamResult:
    guidance_image = anyup_guidance(config, loaded_model, inputs)
    if config.analysis.method == "attention":
        assert isinstance(config.analysis, AttentionAnalysisConfig)
        return extract_attention_maps(
            loaded_model.model,
            inputs,
            loaded_model.metadata,
            layers=config.analysis.layers,
            heads=config.analysis.heads,
            head_fusion=config.analysis.head_fusion,
            normalize=False,
            interpolation=config.visualization.interpolation,
            guidance_image=guidance_image,
            output_size=output_size,
            anyup_query_chunk_size=config.visualization.anyup_query_chunk_size,
        )
    if config.analysis.method == "rollout":
        assert isinstance(config.analysis, RolloutAnalysisConfig)
        return extract_attention_rollout(
            loaded_model.model,
            inputs,
            loaded_model.metadata,
            layers=config.analysis.layers,
            normalize=False,
            interpolation=config.visualization.interpolation,
            guidance_image=guidance_image,
            output_size=output_size,
            anyup_query_chunk_size=config.visualization.anyup_query_chunk_size,
        )
    assert isinstance(config.analysis, GradCAMAnalysisConfig)
    target_classes = (
        None
        if config.analysis.target_class is None
        else [config.analysis.target_class] * len(inputs)
    )
    return extract_gradcam(
        loaded_model.model,
        inputs,
        loaded_model.metadata,
        target_layer=config.analysis.target_layer,
        target_classes=target_classes,
        normalize=False,
        interpolation=config.visualization.interpolation,
        guidance_image=guidance_image,
        output_size=output_size,
        anyup_query_chunk_size=config.visualization.anyup_query_chunk_size,
    )


def _map_streams(
    config: VisionLensConfig,
    result: AttentionExtractionResult | GradCamResult,
) -> tuple[_MapStream, ...]:
    method = config.analysis.method
    if isinstance(result, GradCamResult):
        return (_MapStream("gradcam", result.maps),)

    streams = []
    for layer in result.layers:
        map_count = int(layer.maps.shape[1])
        for map_index in range(map_count):
            if map_count == 1 and layer.head_fusion != "none":
                head_name = f"heads-{layer.head_fusion}"
            elif layer.head_indices is None:
                head_name = f"head-{map_index}"
            else:
                head_name = f"head-{layer.head_indices[map_index]}"
            streams.append(
                _MapStream(
                    map_stream_name(method, layer.layer_index, head_name),
                    layer.maps[:, map_index : map_index + 1],
                )
            )
    return tuple(streams)


def _smooth_streams(
    streams: tuple[_MapStream, ...],
    state: dict[str, Any],
    strength: float,
) -> tuple[_MapStream, ...]:
    if strength == 0:
        return streams
    smoothed_streams = []
    for stream in streams:
        frames = []
        previous = state.get(stream.name)
        for current in torch.as_tensor(stream.maps):
            smoothed = (
                current
                if previous is None
                else strength * previous + (1.0 - strength) * current
            )
            frames.append(smoothed)
            previous = smoothed
        state[stream.name] = previous
        smoothed_streams.append(_MapStream(stream.name, torch.stack(frames)))
    return tuple(smoothed_streams)


def _smooth_images(
    images: tuple[Image.Image, ...],
    state: dict[str, Any],
    key: str,
    strength: float,
) -> tuple[Image.Image, ...]:
    if strength == 0:
        return images
    result = []
    previous = state.get(key)
    for image in images:
        current = np.asarray(image, dtype=np.float32)
        smoothed = (
            current
            if previous is None
            else strength * previous + (1.0 - strength) * current
        )
        result.append(
            Image.fromarray(np.clip(np.rint(smoothed), 0, 255).astype(np.uint8))
        )
        previous = smoothed
    state[key] = previous
    return tuple(result)


def _preprocess_frames(
    batch_index: int,
    frames: tuple[SampledVideoFrame, ...],
    source_path: Path,
    transform: Any,
    loaded_model: LoadedModel,
    *,
    include_display_images: bool = True,
) -> Any:
    input_batch = InputBatch(
        index=batch_index,
        count=len(frames),
        paths=tuple(source_path for _ in frames),
        labels=tuple(f"frame-{frame.index:06d}" for frame in frames),
        images=tuple(frame.image for frame in frames),
    )
    return preprocess_batch(
        input_batch,
        transform,
        loaded_model,
        include_display_images=include_display_images,
    )


def _frame_at_output_size(
    frame: Image.Image,
    output_size: tuple[int, int],
) -> Image.Image:
    if frame.size == output_size:
        return frame
    return frame.resize(output_size, Image.Resampling.BILINEAR)


def _patch_grid(loaded_model: LoadedModel) -> tuple[int, int]:
    patch_size = loaded_model.metadata.patch_size
    if patch_size is None:
        raise ValueError("Patch PCA requires a model with a known patch size.")
    return infer_patch_grid_from_image(loaded_model.metadata.image_size, patch_size)


def _video_model_for_source(
    loaded_model: LoadedModel,
    config: VisionLensConfig,
    source: VideoMetadata,
) -> LoadedModel:
    if config.preprocessing.crop != "none" or config.preprocessing.pad != "none":
        raise ValueError(
            "Video aspect-ratio preprocessing requires preprocessing.crop and "
            "preprocessing.pad to be 'none'."
        )
    if source.width <= 0 or source.height <= 0:
        raise ValueError("Video dimensions must be positive.")
    patch_size = loaded_model.metadata.patch_size
    if loaded_model.metadata.architecture == "vit" and patch_size is None:
        raise ValueError("Video ViT preprocessing requires a known patch size.")
    scale = config.preprocessing.image_size / max(source.width, source.height)
    height = max(1, round(source.height * scale))
    width = max(1, round(source.width * scale))
    if patch_size is not None:
        height = max(patch_size[0], round(height / patch_size[0]) * patch_size[0])
        width = max(patch_size[1], round(width / patch_size[1]) * patch_size[1])
    metadata = replace(
        loaded_model.metadata,
        input_size=(3, height, width),
        image_size=(height, width),
        data_config={
            **loaded_model.metadata.data_config,
            "input_size": (3, height, width),
        },
    )
    return replace(loaded_model, metadata=metadata)


def _validate_gradcam_class(
    config: VisionLensConfig,
    loaded_model: LoadedModel,
) -> None:
    if not isinstance(config.analysis, GradCAMAnalysisConfig):
        return
    target_class = config.analysis.target_class
    class_count = loaded_model.metadata.num_classes
    if (
        target_class is not None
        and class_count is not None
        and target_class >= class_count
    ):
        raise ValueError(
            f"analysis.target_class={target_class} is outside the model's class "
            f"range 0 through {class_count - 1}."
        )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
