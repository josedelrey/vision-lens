import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vision_lens.attention import GradCamResult
from vision_lens.config import PreprocessingConfig, VideoConfig, parse_config
from vision_lens.models import LoadedModel, ModelMetadata
from vision_lens.pipeline import run_pipeline_from_config
from vision_lens.processing import build_batch_preprocessor
from vision_lens.video import (
    VideoMetadata,
    VideoWriter,
    estimated_sample_count,
    iter_sampled_frames,
    iter_video_batches,
    probe_video,
    resolve_sampling_rate,
)
from vision_lens.video_pipeline import VideoBatchPipelineResult, _video_model_for_source

av = pytest.importorskip("av")


@pytest.mark.parametrize(
    ("architecture", "patch_size", "expected"),
    [
        ("vit", (14, 14), (378, 672)),
        ("vit", (8, 8), (376, 672)),
        ("cnn", None, (378, 672)),
    ],
)
def test_video_model_uses_rectangular_source_geometry(
    architecture, patch_size, expected
):
    config = SimpleNamespace(
        preprocessing=SimpleNamespace(image_size=672, crop="none", pad="none")
    )
    loaded = LoadedModel(
        model=object(),
        metadata=ModelMetadata(
            architecture=architecture,
            backend="timm" if architecture == "vit" else "torchvision",
            name="mock",
            pretrained=False,
            device="cpu",
            input_size=(3, 672, 672),
            image_size=(672, 672),
            patch_size=patch_size,
            num_classes=2,
            data_config={"input_size": (3, 672, 672)},
        ),
    )
    source = VideoMetadata(Path("clip.mp4"), 1920, 1080, 1.0, 30.0)

    result = _video_model_for_source(loaded, config, source)

    assert result.metadata.image_size == expected
    assert result.metadata.input_size == (3, *expected)
    assert result.metadata.data_config["input_size"] == (3, *expected)
    assert loaded.metadata.image_size == (672, 672)
    transform = build_batch_preprocessor(result, PreprocessingConfig(image_size=672))
    assert transform(Image.new("RGB", (1920, 1080))).shape == (3, *expected)
    portrait = _video_model_for_source(
        loaded, config, replace(source, width=1080, height=1920)
    )
    assert portrait.metadata.image_size == tuple(reversed(expected))


def test_timestamp_sampling_and_batching_use_requested_times(tmp_path):
    source = tmp_path / "source.mp4"
    _make_video(source, frame_count=10, frame_rate=10)
    settings = VideoConfig(
        start_time=0.2,
        end_time=0.71,
        sampling_rate=4,
    )

    frames = tuple(iter_sampled_frames(source, settings))
    batches = tuple(iter_video_batches(source, settings, batch_size=2))

    assert [frame.timestamp for frame in frames] == pytest.approx([0.2, 0.45, 0.7])
    assert all(frame.source_timestamp + 1e-9 >= frame.timestamp for frame in frames)
    assert [len(batch.frames) for batch in batches] == [2, 1]


def test_auto_sampling_rate_matches_source_fps(tmp_path):
    source = tmp_path / "source.mp4"
    _make_video(source, frame_count=6, frame_rate=6)
    metadata = probe_video(source)
    settings = VideoConfig(sampling_rate="auto")

    frames = tuple(iter_sampled_frames(source, settings))

    assert metadata.source_frame_rate == pytest.approx(6)
    assert estimated_sample_count(metadata, settings) == 6
    assert [frame.timestamp for frame in frames] == pytest.approx(
        [index / 6 for index in range(6)]
    )


def test_auto_sampling_rate_requires_source_fps():
    metadata = VideoMetadata(
        path=Path("missing.mp4"),
        width=64,
        height=48,
        duration=1.0,
        source_frame_rate=None,
    )

    with pytest.raises(ValueError, match="requires a valid source video FPS"):
        resolve_sampling_rate("auto", metadata.source_frame_rate)
    with pytest.raises(ValueError, match="requires a valid source video FPS"):
        estimated_sample_count(metadata, VideoConfig(sampling_rate="auto"))


def test_video_writer_assigns_explicit_constant_playback_timing(tmp_path):
    output = tmp_path / "timed.mp4"
    with VideoWriter(
        output,
        frame_rate=2.5,
        resolution=(64, 48),
        codec="libx264",
    ) as writer:
        for index in range(5):
            writer.write(Image.new("RGB", (20, 20), index * 30))

    metadata = probe_video(output)
    frames = tuple(
        iter_sampled_frames(
            output,
            VideoConfig(sampling_rate=2.5),
        )
    )

    assert metadata.duration == pytest.approx(2.0, abs=0.05)
    assert [frame.timestamp for frame in frames] == pytest.approx(
        [0.0, 0.4, 0.8, 1.2, 1.6]
    )


def test_temporal_smoothing_is_sequential_across_batches():
    from vision_lens.video_pipeline import _MapStream, _smooth_streams

    state = {}
    first = _smooth_streams(
        (_MapStream("map", torch.tensor([[[[0.0]]], [[[2.0]]]])),),
        state,
        0.5,
    )
    second = _smooth_streams(
        (_MapStream("map", torch.tensor([[[[3.0]]]])),),
        state,
        0.5,
    )
    unchanged = _MapStream("map", torch.tensor([[[[4.0]]]]))

    assert first[0].maps.flatten().tolist() == [0.0, 1.0]
    assert second[0].maps.flatten().tolist() == [2.0]
    assert _smooth_streams((unchanged,), {}, 0) == (unchanged,)


def test_short_pca_video_uses_frozen_projection_and_bounded_batches(
    monkeypatch,
    tmp_path,
):
    from vision_lens import video_pipeline

    source = tmp_path / "clip.mp4"
    output_dir = tmp_path / "outputs"
    _make_video(source, frame_count=6, frame_rate=6)
    config = parse_config(
        {
            "input": {"files": [str(source)]},
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
                "pretrained": False,
            },
            "preprocessing": {"image_size": 4},
            "analysis": {
                "method": "patch_pca",
                "foreground_threshold": 0.4,
                "foreground_side": "low",
            },
            "runtime": {"device": "cpu", "batch_size": 2},
            "visualization": {},
            "output": {
                "directory": str(output_dir),
                "overwrite": "error",
            },
            "video": {
                "sampling_rate": 3,
                "pca_fit_frames": 2,
                "output_resolution": [64, 48],
            },
        }
    )
    loaded_model = LoadedModel(
        model=object(),
        metadata=ModelMetadata(
            architecture="vit",
            backend="timm",
            name="mock_vit",
            pretrained=False,
            device="cpu",
            input_size=(3, 4, 4),
            image_size=(4, 4),
            patch_size=(2, 2),
            num_classes=2,
            data_config={"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
        ),
    )
    embedding_batch_sizes = []
    projection_ids = []
    original_project = video_pipeline.project_patch_embeddings

    def fake_embeddings(_model, inputs, _patch_grid):
        embedding_batch_sizes.append(len(inputs))
        start = len(embedding_batch_sizes) * 10
        return torch.arange(
            start,
            start + len(inputs) * 4 * 5,
            dtype=torch.float32,
        ).reshape(len(inputs), 4, 5)

    def record_projection(*args, **kwargs):
        projection_ids.append(id(kwargs["projection"]))
        return original_project(*args, **kwargs)

    monkeypatch.setattr(
        video_pipeline,
        "load_model",
        lambda _config, **_kwargs: loaded_model,
    )
    monkeypatch.setattr(
        video_pipeline,
        "build_batch_preprocessor",
        lambda *_args, **_kwargs: lambda _image: torch.ones(3, 4, 4),
    )
    monkeypatch.setattr(video_pipeline, "extract_patch_embeddings", fake_embeddings)
    monkeypatch.setattr(video_pipeline, "project_patch_embeddings", record_projection)

    result = run_pipeline_from_config(config)

    assert result.processed_frames == 3
    assert result.frame_rate == 3
    assert result.duration == pytest.approx(1.0)
    assert max(embedding_batch_sizes) <= 2
    assert len(set(projection_ids)) == 1
    assert {path.name for path in result.output_paths} == {
        "clip_patch_pca.mp4",
        "clip_patch_pca_comparison.mp4",
    }
    for output_path in result.output_paths:
        assert probe_video(output_path).duration == pytest.approx(1.0, abs=0.05)
        with av.open(str(output_path)) as container:
            assert not container.streams.audio

    manifest = json.loads((output_dir / "run-manifest.json").read_text())
    assert manifest["run"]["sampled_frames"] == 3
    assert manifest["run"]["sampling_rate"] == 3
    assert manifest["run"]["encoded_duration"] == 1
    assert manifest["run"]["audio"] == "omitted"


@pytest.mark.parametrize(("sampling_rate", "expected_frames"), [(2, 2), ("auto", 4)])
def test_gradcam_video_exports_overlays_with_one_fixed_class(
    monkeypatch,
    tmp_path,
    sampling_rate,
    expected_frames,
):
    from vision_lens import video_pipeline

    source = tmp_path / "clip.mp4"
    output_dir = tmp_path / "outputs"
    _make_video(source, frame_count=4, frame_rate=4)
    config = parse_config(
        {
            "input": {"files": [str(source)]},
            "model": {
                "architecture": "cnn",
                "backend": "torchvision",
                "name": "mock_cnn",
                "pretrained": False,
            },
            "preprocessing": {"image_size": 4},
            "analysis": {
                "method": "gradcam",
                "target_layer": "features.0",
                "target_class": 1,
            },
            "runtime": {"device": "cpu", "batch_size": 1},
            "visualization": {},
            "output": {
                "directory": str(output_dir),
                "heatmaps": False,
                "overlays": True,
                "grids": True,
            },
            "video": {
                "sampling_rate": sampling_rate,
                "output_resolution": [64, 48],
                "temporal_smoothing": 0.5,
            },
        }
    )
    loaded_model = LoadedModel(
        model=object(),
        metadata=ModelMetadata(
            architecture="cnn",
            backend="torchvision",
            name="mock_cnn",
            pretrained=False,
            device="cpu",
            input_size=(3, 4, 4),
            image_size=(4, 4),
            patch_size=None,
            num_classes=2,
            data_config={"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
        ),
    )
    target_classes = []
    overlay_sizes = []
    original_overlay = video_pipeline.overlay_attention

    def record_overlay(image, *args, **kwargs):
        overlay_sizes.append(image.size)
        return original_overlay(image, *args, **kwargs)

    def fake_gradcam(_model, inputs, _metadata, **kwargs):
        target_classes.append(kwargs["target_classes"])
        return GradCamResult(
            logits=torch.zeros(len(inputs), 2),
            maps=torch.arange(len(inputs) * 16, dtype=torch.float32).reshape(
                len(inputs), 1, 4, 4
            ),
            target_layer=kwargs["target_layer"],
            target_classes=tuple(kwargs["target_classes"]),
            image_size=(4, 4),
        )

    monkeypatch.setattr(
        video_pipeline,
        "load_model",
        lambda _config, **_kwargs: loaded_model,
    )
    monkeypatch.setattr(
        video_pipeline,
        "build_batch_preprocessor",
        lambda loaded, _config: (
            lambda _image: torch.ones(3, *loaded.metadata.image_size)
        ),
    )
    monkeypatch.setattr(video_pipeline, "extract_gradcam", fake_gradcam)
    monkeypatch.setattr(video_pipeline, "overlay_attention", record_overlay)

    result = run_pipeline_from_config(config)

    assert target_classes == [[1]] * expected_frames
    assert overlay_sizes == [(64, 48)] * expected_frames
    assert result.processed_frames == expected_frames
    assert result.frame_rate == (4 if sampling_rate == "auto" else 2)
    assert {path.name for path in result.output_paths} == {
        "clip_gradcam_overlay.mp4",
        "clip_gradcam_comparison.mp4",
    }
    assert all(
        probe_video(path).duration == pytest.approx(1.0, abs=0.05)
        for path in result.output_paths
    )
    assert probe_video(output_dir / "clip_gradcam_comparison.mp4").width == 128
    manifest = json.loads((output_dir / "run-manifest.json").read_text())
    assert manifest["model"]["input_size"] == [3, 3, 4]
    assert manifest["configuration"]["video"]["sampling_rate"] == sampling_rate
    assert manifest["run"]["sampling_rate"] == result.frame_rate


def test_multiple_videos_have_independent_outputs_and_sampling_rates(
    monkeypatch, tmp_path
):
    from vision_lens import video_pipeline

    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    _make_video(first, frame_count=2, frame_rate=2)
    _make_video(second, frame_count=3, frame_rate=3)
    output_dir = tmp_path / "outputs"
    config = parse_config(
        {
            "input": {"files": [str(first), str(second)]},
            "model": {
                "architecture": "cnn",
                "backend": "torchvision",
                "name": "mock_cnn",
                "pretrained": False,
            },
            "preprocessing": {"image_size": 4},
            "analysis": {
                "method": "gradcam",
                "target_layer": "features.0",
                "target_class": 1,
            },
            "runtime": {"device": "cpu", "batch_size": 1},
            "visualization": {},
            "output": {
                "directory": str(output_dir),
                "heatmaps": False,
                "overlays": True,
                "grids": False,
            },
            "video": {"sampling_rate": "auto", "output_resolution": [64, 48]},
        }
    )
    loaded_model = LoadedModel(
        model=object(),
        metadata=ModelMetadata(
            architecture="cnn",
            backend="torchvision",
            name="mock_cnn",
            pretrained=False,
            device="cpu",
            input_size=(3, 4, 4),
            image_size=(4, 4),
            patch_size=None,
            num_classes=2,
            data_config={"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
        ),
    )

    def fake_gradcam(_model, inputs, _metadata, **kwargs):
        return GradCamResult(
            logits=torch.zeros(len(inputs), 2),
            maps=torch.ones(len(inputs), 1, 4, 4),
            target_layer=kwargs["target_layer"],
            target_classes=tuple(kwargs["target_classes"]),
            image_size=(4, 4),
        )

    model_loads = []

    def load_once(_config, **_kwargs):
        model_loads.append(True)
        return loaded_model

    monkeypatch.setattr(video_pipeline, "load_model", load_once)
    monkeypatch.setattr(
        video_pipeline,
        "build_batch_preprocessor",
        lambda loaded, _config: (
            lambda _image: torch.ones(3, *loaded.metadata.image_size)
        ),
    )
    monkeypatch.setattr(video_pipeline, "extract_gradcam", fake_gradcam)

    result = run_pipeline_from_config(config)

    assert isinstance(result, VideoBatchPipelineResult)
    assert len(model_loads) == 1
    assert len(result.videos) == 2
    assert [video.processed_frames for video in result.videos] == [2, 3]
    assert [video.frame_rate for video in result.videos] == [2, 3]
    assert {path.parent.name for path in result.output_paths} == {"first", "second"}
    for name, frame_count in (("first", 2), ("second", 3)):
        manifest = json.loads((output_dir / name / "run-manifest.json").read_text())
        assert manifest["run"]["sampled_frames"] == frame_count
        assert manifest["inputs"][0]["path"].endswith(f"{name}.mp4")


def _make_video(path, *, frame_count, frame_rate):
    with VideoWriter(
        path,
        frame_rate=frame_rate,
        resolution=(64, 48),
        codec="libx264",
    ) as writer:
        for index in range(frame_count):
            array = np.zeros((48, 64, 3), dtype=np.uint8)
            array[:, :, 0] = index * 20
            array[:, :, 1] = np.arange(64, dtype=np.uint8)
            writer.write(Image.fromarray(array, mode="RGB"))
