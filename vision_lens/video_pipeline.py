from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from PIL import Image

from vision_lens.attention import (
    AttentionExtractionResult,
    GradCamResult,
    extract_attention_maps,
    extract_attention_rollout,
    extract_gradcam,
    infer_patch_grid_from_image,
)
from vision_lens.config import VisionLensConfig
from vision_lens.feature_pca import (
    PatchPCAProjection,
    extract_patch_embeddings,
    fit_patch_pca_projection_batches,
    load_patch_pca_projection,
    project_patch_embeddings,
    save_patch_pca_projection,
)
from vision_lens.manifest import check_manifest_overwrite, write_run_manifest
from vision_lens.models import LoadedModel, load_model
from vision_lens.processing import (
    InputBatch,
    build_batch_preprocessor,
    preprocess_batch,
)
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
    resolved_output_resolution,
)
from vision_lens.visualization import image_grid, overlay_attention, render_heatmap


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
class _MapStream:
    name: str
    maps: Any


def run_video_from_config(config: VisionLensConfig) -> VideoPipelineResult:
    if config.video is None:
        raise ValueError("Video pipeline requires a video configuration section.")

    started_at = datetime.now(timezone.utc)
    check_manifest_overwrite(config)
    require_video_dependencies()
    source_path = config.input.paths[0]
    source = probe_video(source_path)
    _apply_seed(config.runtime.seed)
    loaded_model = load_model(config)
    _validate_gradcam_class(config, loaded_model)
    transform = build_batch_preprocessor(loaded_model, config.preprocessing)
    projection = _video_pca_projection(config, loaded_model, transform, source)
    normalization_range = _video_normalization_range(
        config,
        loaded_model,
        transform,
    )

    resolution = resolved_output_resolution(source, config.video.output_resolution)
    exports = _VideoExports(config, source_path.stem, resolution)
    smoothing_state: dict[str, Any] = {}
    processed_frames = 0
    try:
        for frame_batch in iter_video_batches(
            source_path,
            config.video,
            config.runtime.batch_size,
        ):
            batch = _preprocess_frames(
                frame_batch.index,
                frame_batch.frames,
                source_path,
                transform,
                loaded_model,
            )
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
                    image_size=loaded_model.metadata.image_size,
                    projection=projection,
                )
                pca_images = _smooth_images(
                    pca.images,
                    smoothing_state,
                    "patch-pca",
                    config.video.temporal_smoothing,
                )
                exports.write_pca_batch(
                    frame_batch.frames,
                    batch.display_images,
                    pca_images,
                    pca.foreground_mask,
                    frame_batch.index,
                )
            else:
                analysis = _analyze_maps(config, loaded_model, batch.inputs)
                streams = _smooth_streams(
                    _map_streams(config, analysis),
                    smoothing_state,
                    config.video.temporal_smoothing,
                )
                exports.write_map_batch(
                    frame_batch.frames,
                    batch.display_images,
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
    if config.analysis.save_projection is not None and projection is not None:
        if _can_write(config.analysis.save_projection, config.output.overwrite):
            save_patch_pca_projection(projection, config.analysis.save_projection)
            output_paths.insert(0, config.analysis.save_projection)
    output_paths_tuple = tuple(output_paths)
    encoded_duration = processed_frames / config.video.sampling_rate
    write_run_manifest(
        config,
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
        config=config,
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
                heatmap = render_heatmap(
                    stream.maps,
                    cmap=self.config.visualization.cmap,
                    batch_index=frame_index,
                    normalization=normalization,
                    normalization_range=normalization_range,
                )
                overlay = overlay_attention(
                    original,
                    stream.maps,
                    alpha=self.config.visualization.overlay_alpha,
                    cmap=self.config.visualization.cmap,
                    batch_index=frame_index,
                    normalization=normalization,
                    normalization_range=normalization_range,
                )
                if self.config.output.heatmaps:
                    self._write_video(f"{stream.name}_heatmap", heatmap)
                if self.config.output.overlays:
                    self._write_video(f"{stream.name}_overlay", overlay)
                if self.config.output.grids:
                    self._write_video(
                        f"{stream.name}_comparison",
                        _comparison_frame(original, overlay, self.config),
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
        originals: tuple[Image.Image, ...],
        pca_images: tuple[Image.Image, ...],
        foreground_mask: Any,
        batch_index: int,
    ) -> None:
        for original, pca_image in zip(originals, pca_images, strict=True):
            if self.config.output.heatmaps:
                self._write_video("patch_pca", pca_image)
            if self.config.output.grids:
                self._write_video(
                    "patch_pca_comparison",
                    _comparison_frame(original, pca_image, self.config),
                )
        if self.config.output.raw_arrays:
            self._write_raw_batch(
                "patch_pca_foreground",
                foreground_mask,
                frames,
                batch_index,
            )

    def close(self) -> None:
        close_video_writers(
            tuple(writer for writer in self._writers.values() if writer is not None)
        )

    def _write_video(self, name: str, image: Image.Image) -> None:
        path = self.config.output.directory / f"{self.source_stem}_{name}.mp4"
        writer = self._writer(path)
        if writer is not None:
            writer.write(image)

    def _writer(self, path: Path) -> VideoWriter | None:
        if path not in self._writers:
            if not _can_write(path, self.config.output.overwrite):
                self._writers[path] = None
            else:
                assert self.config.video is not None
                self._writers[path] = VideoWriter(
                    path,
                    frame_rate=self.config.video.sampling_rate,
                    resolution=self.resolution,
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
        path = self.config.output.directory / (
            f"{self.source_stem}_{name}_frames-{batch_index:06d}.{extension}"
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
    if config.analysis.method != "patch_pca":
        return None
    if config.analysis.projection == "load":
        return load_patch_pca_projection(config.analysis.projection_path)

    assert config.video is not None
    sample_count = estimated_sample_count(source, config.video)
    selected = representative_frame_indices(sample_count, config.video.pca_fit_frames)
    selected_set = None if selected is None else set(selected)
    staged_paths = []
    staged_frame_count = 0
    with TemporaryDirectory(prefix="vision-lens-video-pca-") as temporary_directory:
        for frame_batch in iter_video_batches(
            config.input.paths[0],
            config.video,
            config.runtime.batch_size,
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

        return fit_patch_pca_projection_batches(
            embedding_batches,
            foreground_threshold=config.analysis.foreground_threshold,
            foreground_side=config.analysis.foreground_side,
        )


def _video_normalization_range(
    config: VisionLensConfig,
    loaded_model: LoadedModel,
    transform: Any,
) -> tuple[float, float] | None:
    if config.analysis.method == "patch_pca":
        return None
    if config.visualization.normalization == "fixed":
        return config.visualization.normalization_range
    if config.visualization.normalization == "per_map":
        return None

    assert config.video is not None
    current = None
    smoothing_state: dict[str, Any] = {}
    for frame_batch in iter_video_batches(
        config.input.paths[0],
        config.video,
        config.runtime.batch_size,
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
            _map_streams(config, _analyze_maps(config, loaded_model, batch.inputs)),
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
) -> AttentionExtractionResult | GradCamResult:
    if config.analysis.method == "attention":
        return extract_attention_maps(
            loaded_model.model,
            inputs,
            loaded_model.metadata,
            layers=config.analysis.layers,
            heads=config.analysis.heads,
            head_fusion=config.analysis.head_fusion,
            normalize=False,
        )
    if config.analysis.method == "rollout":
        return extract_attention_rollout(
            loaded_model.model,
            inputs,
            loaded_model.metadata,
            layers=config.analysis.layers,
            normalize=False,
        )
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
                    f"{method}_layer-{layer.layer_index}_{head_name}",
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


def _comparison_frame(
    original: Image.Image,
    visualization: Image.Image,
    config: VisionLensConfig,
) -> Image.Image:
    return image_grid(
        (original, visualization),
        columns=2,
        background=config.visualization.background or "black",
        gap=0 if config.visualization.spacing is None else config.visualization.spacing,
        padding=(
            0 if config.visualization.padding is None else config.visualization.padding
        ),
    )


def _patch_grid(loaded_model: LoadedModel) -> tuple[int, int]:
    patch_size = loaded_model.metadata.patch_size
    if patch_size is None:
        raise ValueError("Patch PCA requires a model with a known patch size.")
    return infer_patch_grid_from_image(loaded_model.metadata.image_size, patch_size)


def _validate_gradcam_class(
    config: VisionLensConfig,
    loaded_model: LoadedModel,
) -> None:
    target_class = config.analysis.target_class
    class_count = loaded_model.metadata.num_classes
    if (
        config.analysis.method == "gradcam"
        and target_class is not None
        and class_count is not None
        and target_class >= class_count
    ):
        raise ValueError(
            f"analysis.target_class={target_class} is outside the model's class "
            f"range 0 through {class_count - 1}."
        )


def _can_write(path: Path, policy: str) -> bool:
    if not path.exists() or policy == "replace":
        return True
    if policy == "skip":
        return False
    raise FileExistsError(
        f"Output already exists: {path}. Set output.overwrite to 'replace' or 'skip'."
    )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
