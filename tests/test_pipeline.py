import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vision_lens.analysis.attention import (
    AttentionExtractionResult,
    GradCamResult,
    LayerAttentionMaps,
)
from vision_lens.analysis.patch_pca import PatchPCAResult
from vision_lens.config import (
    OutputConfig,
    VisualizationConfig,
    parse_config,
)
from vision_lens.errors import PipelineError
from vision_lens.models import LoadedModel, ModelMetadata
from vision_lens.pipeline import run_pipeline_from_config
from vision_lens.pipeline.image import (
    export_attention_outputs,
    export_gradcam_outputs,
    export_patch_pca_outputs,
    export_rollout_comparison_outputs,
    run_gradcam_from_config,
    run_patch_pca_from_config,
    run_vit_attention_from_config,
    run_vit_rollout_comparison_from_config,
)


def test_gradcam_rejects_target_class_outside_torchvision_output_range(tmp_path):
    config = parse_config(
        {
            "input": {"files": ["examples/1.jpg"]},
            "model": {
                "architecture": "cnn",
                "backend": "torchvision",
                "name": "resnet18",
                "pretrained": False,
                "options": {"num_classes": 2},
            },
            "preprocessing": {"image_size": 32},
            "analysis": {
                "method": "gradcam",
                "target_layer": "layer4",
                "target_class": 2,
            },
            "runtime": {"device": "cpu"},
            "output": {"directory": str(tmp_path)},
        }
    )

    with pytest.raises(
        PipelineError,
        match=r"target_class=2.*class range 0 through 1",
    ):
        run_pipeline_from_config(config, show_progress=False)


def test_mixed_pipeline_dispatches_images_and_videos_separately(
    monkeypatch,
    tmp_path,
):
    from vision_lens import pipeline

    image = tmp_path / "image.jpg"
    video = tmp_path / "video.mp4"
    image.touch()
    video.touch()
    config = parse_config(
        {
            "input": {"files": [str(image), str(video)]},
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
            },
            "analysis": {"method": "attention", "layers": [0]},
            "output": {"directory": str(tmp_path / "results")},
        }
    )
    calls = []

    def preflight(full_config, branch_config):
        calls.append(("preflight", full_config, branch_config))

    def run_images(branch_config):
        calls.append(("images", branch_config))
        return SimpleNamespace(output_paths=(tmp_path / "image-output.png",))

    def run_videos(branch_config):
        calls.append(("videos", branch_config))
        return SimpleNamespace(output_paths=(tmp_path / "video-output.mp4",))

    monkeypatch.setattr(pipeline, "_preflight_pipeline", preflight)
    monkeypatch.setattr(
        pipeline, "_write_root_manifest", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(pipeline, "_dispatch_image_pipeline", run_images)
    monkeypatch.setattr(pipeline._video_pipeline, "run_video_from_config", run_videos)

    result = pipeline._dispatch_pipeline(config)

    assert isinstance(result, pipeline.MixedPipelineResult)
    assert result.output_paths == (
        tmp_path / "image-output.png",
        tmp_path / "video-output.mp4",
    )
    assert calls[0][0] == "preflight"
    assert calls[1][0] == "images"
    assert calls[1][1].input.paths == (image,)
    assert calls[1][1].output.directory == tmp_path / "results" / "images"
    assert calls[2][0] == "videos"
    assert calls[2][1].input.paths == (video,)
    assert calls[2][1].output.directory == (tmp_path / "results" / "videos" / "video")


def test_mixed_pipeline_preflight_failure_prevents_image_writes(monkeypatch, tmp_path):
    from vision_lens import pipeline

    image = tmp_path / "image.jpg"
    video = tmp_path / "video.mp4"
    image.touch()
    video.touch()
    config = parse_config(
        {
            "input": {"files": [str(image), str(video)]},
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
            },
            "analysis": {"method": "attention", "layers": [0]},
            "output": {"directory": str(tmp_path / "results")},
        }
    )
    image_dispatched = False

    def fail_video_preflight(branch_config):
        raise RuntimeError("video dependency unavailable")

    def run_images(branch_config):
        nonlocal image_dispatched
        image_dispatched = True

    monkeypatch.setattr(
        pipeline._video_pipeline,
        "preflight_video_dependencies",
        fail_video_preflight,
    )
    monkeypatch.setattr(pipeline, "_dispatch_image_pipeline", run_images)

    with pytest.raises(RuntimeError, match="video dependency unavailable"):
        pipeline._dispatch_pipeline(config)

    assert image_dispatched is False


def test_mixed_pipeline_checks_video_artifacts_before_image_writes(
    monkeypatch,
    tmp_path,
):
    from vision_lens import pipeline

    image = tmp_path / "image.jpg"
    video = tmp_path / "video.mp4"
    image.touch()
    video.touch()
    output_directory = tmp_path / "results"
    config = parse_config(
        {
            "input": {"files": [str(image), str(video)]},
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
            },
            "analysis": {"method": "attention", "layers": [0]},
            "output": {"directory": str(output_directory)},
        }
    )
    video_manifest = output_directory / "videos" / "video" / "run-manifest.json"
    video_manifest.parent.mkdir(parents=True)
    video_manifest.touch()
    image_dispatched = False

    def run_images(branch_config):
        nonlocal image_dispatched
        image_dispatched = True

    monkeypatch.setattr(pipeline, "_dispatch_image_pipeline", run_images)

    with pytest.raises(FileExistsError, match="run-manifest.json"):
        pipeline._dispatch_pipeline(config)

    assert image_dispatched is False


def test_vit_pipeline_exports_figures_with_mocked_model(monkeypatch, tmp_path):
    from vision_lens.media import processing
    from vision_lens.pipeline import image as image_pipeline

    config = parse_config(
        {
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
                "pretrained": False,
                "options": {},
            },
            "input": {
                "files": ["examples/1.jpg"],
            },
            "output": {
                "directory": str(tmp_path),
                "overwrite": "replace",
            },
            "preprocessing": {"image_size": 4},
            "analysis": {
                "method": "attention",
                "layers": [0],
                "heads": None,
                "head_fusion": "mean",
            },
            "runtime": {
                "device": "cpu",
            },
            "visualization": {
                "output_size": "match",
                "overlay_alpha": 0.35,
                "cmap": "viridis",
                "grid_format": "svg",
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
            data_config={
                "mean": (0.0, 0.0, 0.0),
                "std": (1.0, 1.0, 1.0),
            },
        ),
    )
    attention = AttentionExtractionResult(
        logits=torch.zeros(1, 2),
        layers=(
            LayerAttentionMaps(
                layer_index=0,
                maps=torch.tensor([[[[0.0, 0.25], [0.75, 1.0]]]]),
                head_indices=None,
                head_fusion="mean",
                patch_grid=(2, 2),
            ),
        ),
        image_size=(4, 4),
    )

    monkeypatch.setattr(image_pipeline, "load_model", lambda _config: loaded_model)
    monkeypatch.setattr(
        processing,
        "load_images",
        lambda _paths, workers=0: [Image.new("RGB", (7, 5), "white")],
    )
    monkeypatch.setattr(
        image_pipeline,
        "build_batch_preprocessor",
        lambda *_args, **_kwargs: lambda _image: torch.ones(3, 4, 4),
    )
    monkeypatch.setattr(
        image_pipeline,
        "_extract_attention",
        lambda **_kwargs: attention,
    )
    stale_output = tmp_path / "images" / "stale.png"
    stale_output.parent.mkdir()
    stale_output.touch()
    (tmp_path / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "inputs": [],
                "outputs": [str(stale_output.resolve())],
                "run": {"manifests": []},
            }
        ),
        encoding="utf-8",
    )

    result = run_pipeline_from_config(config)
    output_dir = tmp_path / "images"

    assert len(result.output_paths) == 4
    assert {path.name for path in result.output_paths} == {
        "1_layer-0_heads-mean_heatmap.png",
        "1_layer-0_heads-mean_overlay.png",
        "1_layers_heads-mean.svg",
        "layer-0_images_heads-mean.svg",
    }
    assert all(Path(path).is_file() for path in result.output_paths)
    assert not stale_output.exists()
    with Image.open(output_dir / "1_layer-0_heads-mean_heatmap.png") as heatmap:
        assert heatmap.size == (7, 5)
    with Image.open(output_dir / "1_layer-0_heads-mean_overlay.png") as overlay:
        assert overlay.size == (7, 5)
    branch_manifest = output_dir / "run-manifest.json"
    manifest = json.loads(branch_manifest.read_text())
    assert manifest["status"] == "completed"
    assert manifest["model"]["name"] == "mock_vit"
    assert manifest["inputs"][0]["id"] == "1"
    assert len(manifest["outputs"]) == 4
    root_manifest = json.loads((tmp_path / "run-manifest.json").read_text())
    assert root_manifest["run"] == {
        "manifests": [str(branch_manifest.resolve())],
        "media": "image",
    }


def test_rollout_grid_items_place_layer_row_above_rollout_row():
    from vision_lens.pipeline.image import _rollout_grid_items

    layer_attention = _attention_result(layer_indices=(0, 1, 2, 3, 4))
    rollout = _attention_result(layer_indices=(0, 1, 2, 3, 4))

    _images, labels, columns = _rollout_grid_items(
        image=Image.new("RGB", (4, 4), "white"),
        layer_attention=layer_attention,
        rollout=rollout,
        image_index=0,
        alpha=0.35,
        cmap="viridis",
    )

    assert columns == 4
    assert labels == [
        "layer 0",
        "layer 1",
        "layer 2",
        "layer 3",
        "rollout 0",
        "rollout 1",
        "rollout 2",
        "rollout 3",
        "layer 4",
        "",
        "",
        "",
        "rollout 4",
        "",
        "",
        "",
    ]


def test_rollout_grid_caps_requested_columns_to_available_layer_pairs():
    from vision_lens.pipeline.image import _rollout_grid_items

    layer_attention = _attention_result(layer_indices=(0, 1))
    rollout = _attention_result(layer_indices=(0, 1))

    images, labels, columns = _rollout_grid_items(
        image=Image.new("RGB", (4, 4), "white"),
        layer_attention=layer_attention,
        rollout=rollout,
        image_index=0,
        alpha=0.35,
        cmap="viridis",
        columns=5,
    )

    assert columns == 2
    assert len(images) == 4
    assert labels == ["layer 0", "layer 1", "rollout 0", "rollout 1"]


def test_rollout_only_grid_items_exclude_layer_attention():
    from vision_lens.pipeline.image import _rollout_grid_items

    images, labels, columns = _rollout_grid_items(
        image=Image.new("RGB", (4, 4), "white"),
        layer_attention=None,
        rollout=_attention_result(layer_indices=(0, 1, 2, 3, 4)),
        image_index=0,
        alpha=0.35,
        cmap="viridis",
    )

    assert len(images) == 5
    assert labels == [
        "rollout 0",
        "rollout 1",
        "rollout 2",
        "rollout 3",
        "rollout 4",
    ]
    assert columns == 4


def test_rollout_only_export_accepts_no_layer_attention(tmp_path):
    rollout = _attention_result(layer_indices=(0, 1))

    paths = export_rollout_comparison_outputs(
        images=[Image.new("RGB", (4, 4), "white")],
        image_paths=(Path("example.jpg"),),
        layer_attention=None,
        rollout=rollout,
        output_dir=tmp_path,
        alpha=0.35,
        cmap="viridis",
        grid_format="png",
        output_config=OutputConfig(
            tmp_path,
            heatmaps=False,
            overlays=False,
            grids=True,
        ),
        visualization_config=VisualizationConfig(rollout_grid="rollout"),
    )

    assert [path.name for path in paths] == ["example_rollout_layers.png"]
    assert paths[0].is_file()


def test_rollout_only_pipeline_skips_direct_attention_extraction(
    monkeypatch,
    tmp_path,
):
    from vision_lens.analysis import attention as attention_analysis
    from vision_lens.media.processing import InputBatch, PreprocessedBatch
    from vision_lens.pipeline import image as image_pipeline

    source = tmp_path / "input.jpg"
    source.touch()
    config = parse_config(
        {
            "input": {"files": [str(source)]},
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
                "pretrained": False,
            },
            "analysis": {"method": "rollout", "layers": [0]},
            "visualization": {"rollout_grid": "rollout"},
            "output": {
                "directory": str(tmp_path / "results"),
                "heatmaps": False,
                "overlays": False,
                "grids": True,
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
    image = Image.new("RGB", (4, 4), "white")
    input_batch = InputBatch(
        index=0,
        count=1,
        paths=(source,),
        labels=("input",),
        images=(image,),
    )
    preprocessed = PreprocessedBatch(
        source=input_batch,
        inputs=torch.ones(1, 3, 4, 4),
        display_images=(image,),
    )
    rollout = _attention_result(layer_indices=(0,))
    exported_layer_attention = []

    monkeypatch.setattr(
        image_pipeline,
        "_load_model_with_status",
        lambda _config: loaded_model,
    )
    monkeypatch.setattr(
        image_pipeline,
        "build_batch_preprocessor",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        image_pipeline,
        "_tracked_input_batches",
        lambda *_args: iter((input_batch,)),
    )
    monkeypatch.setattr(image_pipeline, "preprocess_batch", lambda *_args: preprocessed)
    monkeypatch.setattr(
        attention_analysis,
        "extract_attention_rollout",
        lambda *_args, **_kwargs: rollout,
    )
    monkeypatch.setattr(
        image_pipeline,
        "_extract_attention",
        lambda **_kwargs: pytest.fail("rollout-only mode extracted direct attention"),
    )

    def fake_export(**kwargs):
        exported_layer_attention.append(kwargs["layer_attention"])
        return ()

    monkeypatch.setattr(
        image_pipeline,
        "export_rollout_comparison_outputs",
        fake_export,
    )

    result = run_vit_rollout_comparison_from_config(config)

    assert exported_layer_attention == [None]
    assert result.attention is rollout


def _minimal_vit_config(method, tmp_path):
    source = tmp_path / "input.jpg"
    source.touch()
    return parse_config(
        {
            "input": {"files": [str(source)]},
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
            },
            "analysis": {"method": method, "layers": [0]},
            "output": {"directory": str(tmp_path / "results")},
        }
    )


def test_rollout_config_dispatches_to_rollout_pipeline(monkeypatch, tmp_path):
    from vision_lens.pipeline import image as image_pipeline

    config = _minimal_vit_config("rollout", tmp_path)
    sentinel = object()
    monkeypatch.setattr(
        image_pipeline,
        "run_vit_rollout_comparison_from_config",
        lambda _received: sentinel,
    )
    from vision_lens import pipeline

    monkeypatch.setattr(
        pipeline, "_write_root_manifest", lambda *_args, **_kwargs: None
    )

    assert run_pipeline_from_config(config) is sentinel


def test_rollout_helper_rejects_attention_config_before_loading_model(tmp_path):
    config = _minimal_vit_config("attention", tmp_path)

    with pytest.raises(ValueError, match="Expected analysis.method='rollout'"):
        run_vit_rollout_comparison_from_config(config)


def test_gradcam_export_preserves_heatmap_overlay_and_grid_layout(tmp_path):
    images = [
        Image.new("RGB", (4, 4), "white"),
        Image.new("RGB", (4, 4), "black"),
    ]
    gradcam = GradCamResult(
        logits=torch.zeros(2, 2),
        maps=torch.tensor(
            [
                [[[0.0, 0.25], [0.75, 1.0]]],
                [[[1.0, 0.75], [0.25, 0.0]]],
            ]
        ),
        target_classes=(0, 1),
        target_layer="layer4",
        image_size=(4, 4),
    )

    paths = export_gradcam_outputs(
        images=images,
        image_paths=(Path("first.jpg"), Path("second.jpg")),
        gradcam=gradcam,
        output_dir=tmp_path,
        alpha=0.8,
        cmap="viridis",
        grid_format="png",
    )

    assert {path.name for path in paths} == {
        "first_gradcam_heatmap.png",
        "first_gradcam_overlay.png",
        "second_gradcam_heatmap.png",
        "second_gradcam_overlay.png",
        "gradcam_images.png",
    }
    with Image.open(tmp_path / "gradcam_images.png") as grid:
        assert grid.size == (460, 256)


def test_gradcam_exports_independent_transparent_png(tmp_path):
    image = Image.new("RGB", (4, 4), "red")
    gradcam = GradCamResult(
        logits=torch.zeros(1, 2),
        maps=torch.tensor([[[[0.0, 0.25], [0.75, 1.0]]]]),
        target_classes=(0,),
        target_layer="layer4",
        image_size=(4, 4),
    )
    output = OutputConfig(
        tmp_path,
        heatmaps=False,
        overlays=False,
        transparent_overlays=True,
        grids=False,
        image_format="jpeg",
    )

    paths = export_gradcam_outputs(
        images=[image],
        image_paths=(Path("first.jpg"),),
        gradcam=gradcam,
        output_dir=tmp_path,
        alpha=0.8,
        cmap="viridis",
        grid_format="png",
        output_config=output,
    )

    assert [path.name for path in paths] == ["first_gradcam_transparent_overlay.png"]
    with Image.open(paths[0]) as overlay:
        assert overlay.mode == "RGBA"
        assert set(np.asarray(overlay)[..., 3].flat) == {204}


def test_matching_output_size_applies_to_all_image_exporters(tmp_path):
    input_size = (9, 5)
    image = Image.new("RGB", input_size, "white")
    visualization = VisualizationConfig(output_size="match")
    attention = _attention_result(layer_indices=(0,))
    gradcam = GradCamResult(
        logits=torch.zeros(1, 2),
        maps=torch.rand(1, 1, 4, 4),
        target_classes=(0,),
        target_layer="layer4",
        image_size=(4, 4),
    )
    patch_pca = PatchPCAResult(
        patch_embeddings=torch.rand(1, 4, 3),
        foreground_mask=torch.ones(1, 4, dtype=torch.bool),
        images=(Image.new("RGB", (4, 4), "red"),),
        patch_grid=(2, 2),
        image_size=(4, 4),
    )
    output = OutputConfig(tmp_path, grids=False)

    attention_paths = export_attention_outputs(
        images=[image],
        image_paths=(Path("attention.jpg"),),
        attention=attention,
        output_dir=tmp_path,
        alpha=0.45,
        cmap="viridis",
        grid_format="png",
        output_config=output,
        visualization_config=visualization,
    )
    rollout_paths = export_rollout_comparison_outputs(
        images=[image],
        image_paths=(Path("rollout.jpg"),),
        layer_attention=attention,
        rollout=attention,
        output_dir=tmp_path,
        alpha=0.45,
        cmap="viridis",
        grid_format="png",
        output_config=output,
        visualization_config=visualization,
    )
    gradcam_paths = export_gradcam_outputs(
        images=[image],
        image_paths=(Path("gradcam.jpg"),),
        gradcam=gradcam,
        output_dir=tmp_path,
        alpha=0.45,
        cmap="viridis",
        grid_format="png",
        output_config=output,
        visualization_config=visualization,
    )
    pca_paths = export_patch_pca_outputs(
        patch_pca,
        image_paths=(Path("pca.jpg"),),
        output_dir=tmp_path,
        output_config=output,
        visualization_config=visualization,
        input_sizes=(input_size,),
    )

    for path in (*attention_paths, *rollout_paths, *gradcam_paths, *pca_paths):
        if path.suffix == ".png":
            with Image.open(path) as rendered:
                assert rendered.size == input_size

    native_dir = tmp_path / "native"
    native_paths = export_gradcam_outputs(
        images=[image],
        image_paths=(Path("gradcam.jpg"),),
        gradcam=gradcam,
        output_dir=native_dir,
        alpha=0.45,
        cmap="viridis",
        grid_format="png",
        output_config=OutputConfig(native_dir, overlays=False, grids=False),
        visualization_config=VisualizationConfig(output_size=None),
    )
    with Image.open(native_paths[0]) as rendered:
        assert rendered.size == (4, 4)


def test_patch_pca_pipeline_exports_reference_style_images(monkeypatch, tmp_path):
    from vision_lens.media import processing
    from vision_lens.pipeline import image as image_pipeline

    horse_a = tmp_path / "horse-a.jpg"
    horse_b = tmp_path / "horse-b.jpg"
    Image.new("RGB", (4, 4), "white").save(horse_a)
    Image.new("RGB", (4, 4), "white").save(horse_b)
    config = parse_config(
        {
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
                "pretrained": False,
                "options": {},
            },
            "input": {"files": [str(horse_a), str(horse_b)]},
            "output": {"directory": str(tmp_path)},
            "preprocessing": {"image_size": 4},
            "runtime": {"device": "cpu"},
            "analysis": {
                "method": "patch_pca",
                "foreground_separation": True,
                "foreground_threshold": 0.5,
                "foreground_side": "low",
            },
            "visualization": {},
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
    patch_pca = PatchPCAResult(
        patch_embeddings=torch.rand(2, 4, 3),
        foreground_mask=torch.ones(2, 4, dtype=torch.bool),
        images=(
            Image.new("RGB", (4, 4), "red"),
            Image.new("RGB", (4, 4), "green"),
        ),
        patch_grid=(2, 2),
        image_size=(4, 4),
    )

    monkeypatch.setattr(image_pipeline, "load_model", lambda _config: loaded_model)
    monkeypatch.setattr(
        processing,
        "load_images",
        lambda _paths, workers=0: [Image.new("RGB", (4, 4), "white")] * 2,
    )
    monkeypatch.setattr(
        image_pipeline,
        "build_batch_preprocessor",
        lambda *_args, **_kwargs: lambda _image: torch.ones(3, 4, 4),
    )
    monkeypatch.setattr(
        image_pipeline,
        "extract_patch_pca",
        lambda *_args, **_kwargs: patch_pca,
    )

    result = run_patch_pca_from_config(config)

    assert result.patch_pca is patch_pca
    assert {path.name for path in result.output_paths} == {
        "horse-a_patch_pca.png",
        "horse-b_patch_pca.png",
        "patch_pca_comparison.png",
    }
    assert all(path.is_file() for path in result.output_paths)
    with Image.open(tmp_path / "patch_pca_comparison.png") as comparison:
        assert comparison.size == (460, 224)


def test_raw_only_patch_pca_skips_rendering(monkeypatch, tmp_path):
    from vision_lens.media import processing
    from vision_lens.pipeline import image as image_pipeline

    source = tmp_path / "source.jpg"
    Image.new("RGB", (4, 4), "white").save(source)
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
            "analysis": {"method": "patch_pca"},
            "runtime": {"device": "cpu"},
            "output": {
                "directory": str(tmp_path / "outputs"),
                "heatmaps": False,
                "grids": False,
                "raw_arrays": True,
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
    calls = []

    def extract(*_args, **kwargs):
        calls.append(kwargs)
        return PatchPCAResult(
            patch_embeddings=torch.rand(1, 4, 3),
            foreground_mask=torch.ones(1, 4, dtype=torch.bool),
            images=(),
            patch_grid=(2, 2),
            image_size=(4, 4),
        )

    monkeypatch.setattr(image_pipeline, "load_model", lambda _config: loaded_model)
    monkeypatch.setattr(
        processing,
        "load_images",
        lambda _paths, workers=0: [Image.new("RGB", (4, 4), "white")],
    )
    monkeypatch.setattr(
        image_pipeline,
        "build_batch_preprocessor",
        lambda *_args, **_kwargs: lambda _image: torch.ones(3, 4, 4),
    )
    monkeypatch.setattr(image_pipeline, "extract_patch_pca", extract)
    monkeypatch.setattr(
        image_pipeline,
        "anyup_guidance",
        lambda *_args, **_kwargs: pytest.fail("raw-only PCA must not build guidance"),
    )

    result = run_patch_pca_from_config(config)

    assert calls[0]["render_images"] is False
    assert {path.name for path in result.output_paths} == {
        "source_patch_embeddings.npy",
        "source_foreground_mask.npy",
    }
    assert all(path.is_file() for path in result.output_paths)


def test_patch_pca_grid_uses_matplotlib_without_labels(monkeypatch, tmp_path):
    from vision_lens.pipeline import image as image_pipeline

    patch_pca = PatchPCAResult(
        patch_embeddings=torch.rand(2, 4, 3),
        foreground_mask=torch.ones(2, 4, dtype=torch.bool),
        images=(
            Image.new("RGB", (4, 4), "red"),
            Image.new("RGB", (4, 4), "green"),
        ),
        patch_grid=(2, 2),
        image_size=(4, 4),
    )
    calls = []
    original = image_pipeline.save_grid

    def record_save_grid(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(image_pipeline, "save_grid", record_save_grid)
    paths = export_patch_pca_outputs(
        patch_pca,
        (Path("first.jpg"), Path("second.jpg")),
        tmp_path,
        output_config=OutputConfig(
            tmp_path,
            heatmaps=False,
            overlays=False,
            grids=True,
        ),
        visualization_config=VisualizationConfig(
            tile_size=(20, 10),
            spacing=3,
            padding=5,
            grid_format="svg",
        ),
    )

    assert [path.name for path in paths] == ["patch_pca_comparison.svg"]
    assert paths[0].is_file()
    assert calls == [
        {
            "labels": ["first", "second"],
            "output_path": tmp_path / "patch_pca_comparison.svg",
            "columns": None,
            "tile_size": (20, 10),
            "spacing": 3,
            "padding": 5,
            "show_labels": False,
            "background": None,
            "dpi": None,
        }
    ]


def test_loaded_patch_pca_projection_does_not_require_fit_settings(
    monkeypatch,
    tmp_path,
):
    from vision_lens.pipeline import image as image_pipeline

    projection_path = tmp_path / "projection.npz"
    projection_path.touch()
    raw = {
        "input": {"files": ["examples/1.jpg"]},
        "model": {
            "architecture": "vit",
            "backend": "timm",
            "name": "mock_model",
        },
        "analysis": {
            "method": "patch_pca",
            "projection": "load",
            "projection_path": str(projection_path),
        },
        "output": {"directory": str(tmp_path / "outputs")},
    }
    config = parse_config(raw)

    def reached_projection_loader(_path):
        raise RuntimeError("projection loader reached")

    monkeypatch.setattr(
        image_pipeline,
        "load_patch_pca_projection",
        reached_projection_loader,
    )

    with pytest.raises(RuntimeError, match="projection loader reached"):
        run_patch_pca_from_config(config)


def test_patch_pca_single_image_run_exports_heatmap_and_comparison(tmp_path):
    patch_pca = PatchPCAResult(
        patch_embeddings=torch.rand(1, 4, 3),
        foreground_mask=torch.ones(1, 4, dtype=torch.bool),
        images=(Image.new("RGB", (4, 4), "red"),),
        patch_grid=(2, 2),
        image_size=(4, 4),
    )

    paths = export_patch_pca_outputs(
        patch_pca,
        image_paths=(Path("horse.jpg"),),
        output_dir=tmp_path,
    )

    assert [path.name for path in paths] == [
        "horse_patch_pca.png",
        "patch_pca_comparison.png",
    ]


def test_patch_pca_single_image_grids_only_exports_comparison(tmp_path):
    patch_pca = PatchPCAResult(
        patch_embeddings=torch.rand(1, 4, 3),
        foreground_mask=torch.ones(1, 4, dtype=torch.bool),
        images=(Image.new("RGB", (4, 4), "red"),),
        patch_grid=(2, 2),
        image_size=(4, 4),
    )
    output = OutputConfig(
        tmp_path,
        heatmaps=False,
        overlays=False,
        grids=True,
        raw_arrays=False,
    )

    paths = export_patch_pca_outputs(
        patch_pca,
        image_paths=(Path("horse.jpg"),),
        output_dir=tmp_path,
        output_config=output,
    )

    assert [path.name for path in paths] == ["patch_pca_comparison.png"]


def test_projection_only_image_pca_skips_projection_and_render_pass(
    monkeypatch, tmp_path
):
    from vision_lens.pipeline import image as image_pipeline

    projection_path = tmp_path / "run-manifest.json.tmp"
    raw = {
        "input": {"files": ["examples/1.jpg"]},
        "model": {
            "architecture": "vit",
            "backend": "timm",
            "name": "mock_vit",
            "pretrained": False,
        },
        "analysis": {"method": "patch_pca", "save_projection": str(projection_path)},
        "output": {
            "directory": str(tmp_path),
            "heatmaps": False,
            "grids": False,
            "raw_arrays": False,
        },
    }
    config = parse_config(raw)
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
            data_config={},
        ),
    )
    fitted_projection = object()

    monkeypatch.setattr(image_pipeline, "load_model", lambda _config: loaded_model)
    monkeypatch.setattr(
        image_pipeline,
        "build_batch_preprocessor",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        image_pipeline,
        "_fit_image_pca_projection",
        lambda *_args, **_kwargs: fitted_projection,
    )

    def write_projection(projection, path, _policy):
        assert projection is fitted_projection
        path.touch()
        return path

    monkeypatch.setattr(image_pipeline, "_write_projection", write_projection)
    monkeypatch.setattr(
        image_pipeline,
        "project_patch_embeddings",
        lambda *_args, **_kwargs: pytest.fail("projection pass should be skipped"),
    )

    result = run_patch_pca_from_config(config)

    assert result.patch_pca is None
    assert result.output_paths == (projection_path,)
    assert projection_path.is_file()
    assert not (tmp_path / "patch_pca_comparison.png").exists()


def test_patch_pca_keeps_single_image_page_in_larger_run(tmp_path):
    patch_pca = PatchPCAResult(
        patch_embeddings=torch.rand(1, 4, 3),
        foreground_mask=torch.ones(1, 4, dtype=torch.bool),
        images=(Image.new("RGB", (4, 4), "red"),),
        patch_grid=(2, 2),
        image_size=(4, 4),
    )

    paths = export_patch_pca_outputs(
        patch_pca,
        image_paths=(Path("horse.jpg"),),
        output_dir=tmp_path,
        grid_page_offset=1,
        total_grid_pages=2,
    )

    assert [path.name for path in paths] == [
        "horse_patch_pca.png",
        "patch_pca_comparison_part-002.png",
    ]


def test_patch_pca_pipeline_combines_grid_across_bounded_batches(
    monkeypatch,
    tmp_path,
):
    from vision_lens.media import processing
    from vision_lens.pipeline import image as image_pipeline

    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    input_paths = []
    for index in range(3):
        path = input_dir / f"image-{index}.jpg"
        Image.new("RGB", (4, 4), index * 40).save(path)
        input_paths.append(path)
    config = parse_config(
        {
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
                "pretrained": False,
            },
            "input": {"files": [str(path) for path in input_paths]},
            "output": {
                "directory": str(output_dir),
                "grids": True,
            },
            "preprocessing": {"image_size": 4},
            "analysis": {
                "method": "patch_pca",
                "foreground_separation": True,
            },
            "runtime": {"device": "cpu", "batch_size": 2},
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
    extracted_batch_sizes = []

    def fake_embeddings(_model, inputs, _patch_grid):
        extracted_batch_sizes.append(len(inputs))
        offset = sum(extracted_batch_sizes)
        return torch.arange(
            offset,
            offset + len(inputs) * 4 * 5,
            dtype=torch.float32,
        ).reshape(len(inputs), 4, 5)

    monkeypatch.setattr(image_pipeline, "load_model", lambda _config: loaded_model)
    monkeypatch.setattr(
        processing,
        "load_images",
        lambda paths, workers=0: [Image.new("RGB", (4, 4), "white") for _ in paths],
    )
    monkeypatch.setattr(
        image_pipeline,
        "build_batch_preprocessor",
        lambda *_args, **_kwargs: lambda _image: torch.ones(3, 4, 4),
    )
    monkeypatch.setattr(image_pipeline, "extract_patch_embeddings", fake_embeddings)

    result = run_patch_pca_from_config(config)

    assert extracted_batch_sizes == [2, 1]
    assert result.patch_pca is None
    assert result.processed_inputs == 3
    assert {path.name for path in result.output_paths} == {
        "image-0_patch_pca.png",
        "image-1_patch_pca.png",
        "image-2_patch_pca.png",
        "patch_pca_comparison.png",
    }
    assert not list(output_dir.glob("patch_pca_comparison_part-*.png"))
    assert (output_dir / "run-manifest.json").is_file()


def test_items_per_grid_splits_gradcam_comparison_files(tmp_path):
    from vision_lens.analysis.attention import GradCamResult

    images = [Image.new("RGB", (4, 4), "white")] * 5
    gradcam = GradCamResult(
        logits=torch.zeros(5, 2),
        maps=torch.rand(5, 1, 4, 4),
        target_classes=(0, 0, 0, 0, 0),
        target_layer="layer4",
        image_size=(4, 4),
    )

    paths = export_gradcam_outputs(
        images=images,
        image_paths=tuple(Path(f"image-{index}.jpg") for index in range(5)),
        gradcam=gradcam,
        output_dir=tmp_path,
        alpha=0.8,
        cmap="viridis",
        grid_format="pdf",
        output_config=OutputConfig(
            tmp_path,
            heatmaps=False,
            overlays=False,
            grids=True,
        ),
        visualization_config=VisualizationConfig(
            columns=2,
            items_per_grid=2,
            grid_format="pdf",
        ),
    )

    assert [path.name for path in paths] == [
        "gradcam_images_part-001.pdf",
        "gradcam_images_part-002.pdf",
        "gradcam_images_part-003.pdf",
    ]


def test_attention_pipeline_combines_cross_image_grid_across_batches(
    monkeypatch,
    tmp_path,
):
    from vision_lens.media import processing
    from vision_lens.pipeline import image as image_pipeline

    config = parse_config(
        {
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
                "pretrained": False,
            },
            "input": {
                "files": [
                    "examples/1.jpg",
                    "examples/2.jpg",
                    "examples/3.jpg",
                ]
            },
            "output": {
                "directory": str(tmp_path),
                "heatmaps": False,
                "overlays": False,
                "grids": True,
                "raw_arrays": True,
            },
            "preprocessing": {"image_size": 4},
            "analysis": {"method": "attention", "layers": [0]},
            "runtime": {"device": "cpu", "batch_size": 2},
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
    batch_sizes = []

    def extract_batch(*, inputs, **_kwargs):
        batch_sizes.append(len(inputs))
        return AttentionExtractionResult(
            logits=torch.zeros(len(inputs), 2),
            layers=(
                LayerAttentionMaps(
                    layer_index=0,
                    maps=torch.rand(len(inputs), 1, 4, 4),
                    head_indices=None,
                    head_fusion="mean",
                    patch_grid=(2, 2),
                ),
            ),
            image_size=(4, 4),
        )

    monkeypatch.setattr(image_pipeline, "load_model", lambda _config: loaded_model)
    monkeypatch.setattr(
        processing,
        "load_images",
        lambda paths, workers=0: [Image.new("RGB", (4, 4), "white") for _ in paths],
    )
    monkeypatch.setattr(
        image_pipeline,
        "build_batch_preprocessor",
        lambda *_args, **_kwargs: lambda _image: torch.ones(3, 4, 4),
    )
    monkeypatch.setattr(image_pipeline, "_extract_attention", extract_batch)

    result = run_vit_attention_from_config(config)

    assert batch_sizes == [2, 1]
    assert result.attention is None
    assert result.processed_inputs == 3
    assert {path.name for path in result.output_paths} == {
        "1_layer-0_heads-mean.npy",
        "1_layers_heads-mean.png",
        "2_layer-0_heads-mean.npy",
        "2_layers_heads-mean.png",
        "3_layer-0_heads-mean.npy",
        "3_layers_heads-mean.png",
        "layer-0_images_heads-mean.png",
    }
    assert not list(tmp_path.glob("layer-0_images_heads-mean_part-*.png"))


def test_shared_normalization_is_fitted_across_all_batches(monkeypatch, tmp_path):
    from vision_lens.media import processing
    from vision_lens.pipeline import image as image_pipeline

    config = parse_config(
        {
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
                "pretrained": False,
            },
            "input": {
                "files": [
                    "examples/1.jpg",
                    "examples/2.jpg",
                    "examples/3.jpg",
                ]
            },
            "output": {"directory": str(tmp_path)},
            "preprocessing": {"image_size": 4},
            "analysis": {"method": "attention", "layers": [0]},
            "runtime": {"device": "cpu", "batch_size": 2},
            "visualization": {"normalization": "shared"},
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
    extraction_index = 0
    load_count = 0
    rendering_configs = []

    def fake_load_model(_config):
        nonlocal load_count
        load_count += 1
        return loaded_model

    def fake_attention(*, inputs, **_kwargs):
        nonlocal extraction_index
        values = ([0.0, 2.0], [10.0], [0.0, 2.0], [10.0])[extraction_index]
        extraction_index += 1
        maps = torch.tensor(values).reshape(len(inputs), 1, 1, 1).expand(-1, 1, 2, 2)
        return AttentionExtractionResult(
            logits=torch.zeros(len(inputs), 2),
            layers=(
                LayerAttentionMaps(
                    layer_index=0,
                    maps=maps,
                    head_indices=None,
                    head_fusion="mean",
                    patch_grid=(2, 2),
                ),
            ),
            image_size=(4, 4),
        )

    def fake_export(**kwargs):
        rendering_configs.append(kwargs["visualization_config"])
        return ()

    monkeypatch.setattr(image_pipeline, "load_model", fake_load_model)
    monkeypatch.setattr(
        processing,
        "load_images",
        lambda paths, workers=0: [Image.new("RGB", (4, 4), "white") for _ in paths],
    )
    monkeypatch.setattr(
        image_pipeline,
        "build_batch_preprocessor",
        lambda *_args, **_kwargs: lambda _image: torch.ones(3, 4, 4),
    )
    monkeypatch.setattr(image_pipeline, "_extract_attention", fake_attention)
    monkeypatch.setattr(image_pipeline, "export_attention_outputs", fake_export)

    result = run_vit_attention_from_config(config)

    assert load_count == 1
    assert result.processed_inputs == 3
    assert len(rendering_configs) == 2
    assert all(item.normalization == "fixed" for item in rendering_configs)
    assert all(item.normalization_range == (0.0, 10.0) for item in rendering_configs)


def test_output_overwrite_error_and_skip_policies(tmp_path):
    from vision_lens.analysis.attention import GradCamResult

    existing = tmp_path / "first_gradcam_heatmap.png"
    existing.touch()
    gradcam = GradCamResult(
        logits=torch.zeros(1, 2),
        maps=torch.rand(1, 1, 4, 4),
        target_classes=(0,),
        target_layer="layer4",
        image_size=(4, 4),
    )
    common = {
        "images": [Image.new("RGB", (4, 4), "white")],
        "image_paths": (Path("first.jpg"),),
        "gradcam": gradcam,
        "output_dir": tmp_path,
        "alpha": 0.8,
        "cmap": "viridis",
        "grid_format": "png",
    }

    with pytest.raises(FileExistsError, match="Output already exists"):
        export_gradcam_outputs(
            **common,
            output_config=OutputConfig(
                tmp_path,
                overlays=False,
                grids=False,
                overwrite="error",
            ),
        )

    paths = export_gradcam_outputs(
        **common,
        output_config=OutputConfig(
            tmp_path,
            overlays=False,
            grids=False,
            overwrite="skip",
        ),
    )
    assert paths == ()


def test_existing_manifest_fails_before_model_loading(monkeypatch, tmp_path):
    from vision_lens.pipeline import image as image_pipeline

    (tmp_path / "run-manifest.json").write_text("{}")
    config = parse_config(
        {
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
                "pretrained": False,
            },
            "input": {"files": ["examples/1.jpg"]},
            "output": {"directory": str(tmp_path), "overwrite": "error"},
            "preprocessing": {"image_size": 4},
            "analysis": {"method": "attention", "layers": [0]},
            "runtime": {"device": "cpu"},
        }
    )
    loaded = False

    def fail_if_loaded(_config):
        nonlocal loaded
        loaded = True
        raise AssertionError("model should not be loaded")

    monkeypatch.setattr(image_pipeline, "load_model", fail_if_loaded)

    with pytest.raises(FileExistsError, match="Output artifact.*run-manifest.json"):
        run_vit_attention_from_config(config)

    assert not loaded


def test_existing_saved_projection_fails_before_model_loading(monkeypatch, tmp_path):
    from vision_lens.pipeline import image as image_pipeline

    projection_path = tmp_path / "projection.npz"
    projection_path.touch()
    config = parse_config(
        {
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
                "pretrained": False,
            },
            "input": {"files": ["examples/1.jpg"]},
            "analysis": {
                "method": "patch_pca",
                "save_projection": str(projection_path),
            },
            "output": {
                "directory": str(tmp_path / "outputs"),
                "heatmaps": False,
                "grids": False,
                "raw_arrays": False,
                "overwrite": "error",
            },
        }
    )
    loaded = False

    def fail_if_loaded(_config):
        nonlocal loaded
        loaded = True
        raise AssertionError("model should not be loaded")

    monkeypatch.setattr(image_pipeline, "load_model", fail_if_loaded)

    with pytest.raises(FileExistsError, match="Output artifact.*projection.npz"):
        run_patch_pca_from_config(config)

    assert not loaded


def test_gradcam_pipeline_passes_fixed_class_workers_and_batch_size(
    monkeypatch,
    tmp_path,
):
    from vision_lens.analysis.attention import GradCamResult
    from vision_lens.media import processing
    from vision_lens.pipeline import image as image_pipeline

    config = parse_config(
        {
            "model": {
                "architecture": "cnn",
                "backend": "torchvision",
                "name": "mock_cnn",
                "pretrained": False,
            },
            "input": {"files": ["examples/1.jpg", "examples/2.jpg"]},
            "output": {
                "directory": str(tmp_path),
                "heatmaps": False,
                "overlays": False,
                "grids": True,
                "raw_arrays": True,
            },
            "preprocessing": {"image_size": 4},
            "analysis": {
                "method": "gradcam",
                "target_layer": "features.0",
                "target_class": 1,
            },
            "runtime": {"device": "cpu", "workers": 2, "batch_size": 1},
            "visualization": {
                "output_size": [6, 2],
                "interpolation": "anyup",
                "anyup_query_chunk_size": 7,
                "padding": 7,
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
    received = []
    worker_counts = []

    def fake_gradcam(_model, inputs, _metadata, **kwargs):
        received.append(kwargs)
        return GradCamResult(
            logits=torch.zeros(len(inputs), 2),
            maps=torch.rand(len(inputs), 1, 4, 4),
            target_classes=tuple(kwargs["target_classes"]),
            target_layer=kwargs["target_layer"],
            image_size=(4, 4),
        )

    monkeypatch.setattr(image_pipeline, "load_model", lambda _config: loaded_model)
    monkeypatch.setattr(
        processing,
        "load_images",
        lambda paths, workers: (
            worker_counts.append(workers)
            or [Image.new("RGB", (4, 4), "white") for _ in paths]
        ),
    )
    monkeypatch.setattr(
        image_pipeline,
        "build_batch_preprocessor",
        lambda *_args, **_kwargs: lambda _image: torch.ones(3, 4, 4),
    )
    monkeypatch.setattr(image_pipeline, "extract_gradcam", fake_gradcam)

    result = run_gradcam_from_config(config)

    assert worker_counts == [2, 2]
    assert [call["target_classes"] for call in received] == [[1], [1]]
    assert all(call["target_layer"] == "features.0" for call in received)
    assert all(call["output_size"] == (2, 6) for call in received)
    assert all(call["anyup_query_chunk_size"] == 7 for call in received)
    assert {path.name for path in result.output_paths} == {
        "1_gradcam.npy",
        "2_gradcam.npy",
        "gradcam_images.png",
    }
    assert not list(tmp_path.glob("gradcam_images_part-*.png"))
    with Image.open(tmp_path / "gradcam_images.png") as grid:
        assert grid.size == (474, 270)


def _attention_result(layer_indices: tuple[int, ...]) -> AttentionExtractionResult:
    return AttentionExtractionResult(
        logits=torch.zeros(1, 2),
        layers=tuple(
            LayerAttentionMaps(
                layer_index=layer_index,
                maps=torch.rand(1, 1, 4, 4),
                head_indices=None,
                head_fusion="mean",
                patch_grid=(2, 2),
            )
            for layer_index in layer_indices
        ),
        image_size=(4, 4),
    )
