import json

import numpy as np
import pytest
import torch
from PIL import Image

from vision_lens.attention import GradCamResult
from vision_lens.config import VideoConfig, parse_config
from vision_lens.models import LoadedModel, ModelMetadata
from vision_lens.pipeline import run_pipeline_from_config
from vision_lens.video import (
    VideoWriter,
    iter_sampled_frames,
    iter_video_batches,
    probe_video,
)

av = pytest.importorskip("av")


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

    monkeypatch.setattr(video_pipeline, "load_model", lambda _config: loaded_model)
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


def test_gradcam_video_exports_overlays_with_one_fixed_class(
    monkeypatch,
    tmp_path,
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
                "sampling_rate": 2,
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

    monkeypatch.setattr(video_pipeline, "load_model", lambda _config: loaded_model)
    monkeypatch.setattr(
        video_pipeline,
        "build_batch_preprocessor",
        lambda *_args, **_kwargs: lambda _image: torch.ones(3, 4, 4),
    )
    monkeypatch.setattr(video_pipeline, "extract_gradcam", fake_gradcam)

    result = run_pipeline_from_config(config)

    assert target_classes == [[1], [1]]
    assert result.processed_frames == 2
    assert {path.name for path in result.output_paths} == {
        "clip_gradcam_overlay.mp4",
        "clip_gradcam_comparison.mp4",
    }
    assert all(
        probe_video(path).duration == pytest.approx(1.0, abs=0.05)
        for path in result.output_paths
    )


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
