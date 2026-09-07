import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from vision_lens.attention import AttentionExtractionResult, LayerAttentionMaps
from vision_lens.config import (
    OutputConfig,
    VisualizationConfig,
    load_preset,
    parse_config,
)
from vision_lens.feature_pca import PatchPCAResult
from vision_lens.models import LoadedModel, ModelMetadata
from vision_lens.pipeline import (
    export_gradcam_outputs,
    run_gradcam_from_config,
    run_patch_pca_from_config,
    run_pipeline_from_config,
    run_vit_attention_from_config,
)


def test_vit_pipeline_exports_figures_with_mocked_model(monkeypatch, tmp_path):
    from vision_lens import pipeline, processing

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
                "paths": ["data/examples/1.jpg"],
            },
            "output": {
                "directory": str(tmp_path),
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
                "overlay_alpha": 0.35,
                "cmap": "viridis",
                "grid_format": "svg",
            },
        }
    )
    config = replace(config, output=OutputConfig(tmp_path))

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

    monkeypatch.setattr(pipeline, "load_model", lambda _config: loaded_model)
    monkeypatch.setattr(
        processing,
        "load_images",
        lambda _paths, workers=0: [Image.new("RGB", (4, 4), "white")],
    )
    monkeypatch.setattr(
        pipeline,
        "build_batch_preprocessor",
        lambda *_args, **_kwargs: lambda _image: torch.ones(3, 4, 4),
    )
    monkeypatch.setattr(
        pipeline,
        "_extract_attention",
        lambda **_kwargs: attention,
    )

    result = run_vit_attention_from_config(config)

    assert len(result.output_paths) == 4
    assert {path.name for path in result.output_paths} == {
        "1_layer-0_heads-mean_heatmap.png",
        "1_layer-0_heads-mean_overlay.png",
        "1_layers_heads-mean.svg",
        "layer-0_images_heads-mean.svg",
    }
    assert all(Path(path).is_file() for path in result.output_paths)
    manifest = json.loads((tmp_path / "run-manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["model"]["name"] == "mock_vit"
    assert manifest["inputs"][0]["id"] == "1"
    assert len(manifest["outputs"]) == 4


def test_rollout_grid_items_place_layer_row_above_rollout_row():
    from vision_lens.pipeline import _rollout_grid_items

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
    from vision_lens.pipeline import _rollout_grid_items

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


def test_rollout_preset_dispatches_to_rollout_pipeline(monkeypatch):
    from vision_lens import pipeline

    config = load_preset("dinov2-reg4-rollout")
    sentinel = object()
    monkeypatch.setattr(
        pipeline,
        "run_vit_rollout_comparison_from_config",
        lambda received: sentinel if received is config else None,
    )

    assert run_pipeline_from_config(config) is sentinel


def test_gradcam_export_preserves_heatmap_overlay_and_grid_layout(tmp_path):
    from vision_lens.attention import GradCamResult

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


def test_patch_pca_pipeline_exports_reference_style_images(monkeypatch, tmp_path):
    from vision_lens import pipeline, processing

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
            "input": {"paths": [str(horse_a), str(horse_b)]},
            "output": {"directory": str(tmp_path)},
            "preprocessing": {"image_size": 4},
            "runtime": {"device": "cpu"},
            "analysis": {
                "method": "patch_pca",
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

    monkeypatch.setattr(pipeline, "load_model", lambda _config: loaded_model)
    monkeypatch.setattr(
        processing,
        "load_images",
        lambda _paths, workers=0: [Image.new("RGB", (4, 4), "white")] * 2,
    )
    monkeypatch.setattr(
        pipeline,
        "build_batch_preprocessor",
        lambda *_args, **_kwargs: lambda _image: torch.ones(3, 4, 4),
    )
    monkeypatch.setattr(
        pipeline,
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
        assert comparison.size == (56, 36)


def test_patch_pca_pipeline_fits_and_transforms_multiple_bounded_batches(
    monkeypatch,
    tmp_path,
):
    from vision_lens import pipeline, processing

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
                "grids": False,
            },
            "preprocessing": {"image_size": 4},
            "analysis": {"method": "patch_pca"},
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

    monkeypatch.setattr(pipeline, "load_model", lambda _config: loaded_model)
    monkeypatch.setattr(
        processing,
        "load_images",
        lambda paths, workers=0: [Image.new("RGB", (4, 4), "white") for _ in paths],
    )
    monkeypatch.setattr(
        pipeline,
        "build_batch_preprocessor",
        lambda *_args, **_kwargs: lambda _image: torch.ones(3, 4, 4),
    )
    monkeypatch.setattr(pipeline, "extract_patch_embeddings", fake_embeddings)

    result = run_patch_pca_from_config(config)

    assert extracted_batch_sizes == [2, 1]
    assert result.patch_pca is None
    assert result.processed_inputs == 3
    assert {path.name for path in result.output_paths} == {
        "image-0_patch_pca.png",
        "image-1_patch_pca.png",
        "image-2_patch_pca.png",
    }
    assert (output_dir / "run-manifest.json").is_file()


def test_items_per_grid_splits_gradcam_comparison_files(tmp_path):
    from vision_lens.attention import GradCamResult

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


def test_attention_pipeline_honors_batch_size_and_raw_only_output(
    monkeypatch,
    tmp_path,
):
    from vision_lens import pipeline, processing

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
                    "data/examples/1.jpg",
                    "data/examples/2.jpg",
                    "data/examples/3.jpg",
                ]
            },
            "output": {
                "directory": str(tmp_path),
                "heatmaps": False,
                "overlays": False,
                "grids": False,
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

    monkeypatch.setattr(pipeline, "load_model", lambda _config: loaded_model)
    monkeypatch.setattr(
        processing,
        "load_images",
        lambda paths, workers=0: [Image.new("RGB", (4, 4), "white") for _ in paths],
    )
    monkeypatch.setattr(
        pipeline,
        "build_batch_preprocessor",
        lambda *_args, **_kwargs: lambda _image: torch.ones(3, 4, 4),
    )
    monkeypatch.setattr(pipeline, "_extract_attention", extract_batch)

    result = run_vit_attention_from_config(config)

    assert batch_sizes == [2, 1]
    assert result.attention is None
    assert result.processed_inputs == 3
    assert len(result.output_paths) == 3
    assert all(path.suffix == ".npy" for path in result.output_paths)


def test_shared_normalization_is_fitted_across_all_batches(monkeypatch, tmp_path):
    from vision_lens import pipeline, processing

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
                    "data/examples/1.jpg",
                    "data/examples/2.jpg",
                    "data/examples/3.jpg",
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

    monkeypatch.setattr(pipeline, "load_model", fake_load_model)
    monkeypatch.setattr(
        processing,
        "load_images",
        lambda paths, workers=0: [Image.new("RGB", (4, 4), "white") for _ in paths],
    )
    monkeypatch.setattr(
        pipeline,
        "build_batch_preprocessor",
        lambda *_args, **_kwargs: lambda _image: torch.ones(3, 4, 4),
    )
    monkeypatch.setattr(pipeline, "_extract_attention", fake_attention)
    monkeypatch.setattr(pipeline, "export_attention_outputs", fake_export)

    result = run_vit_attention_from_config(config)

    assert load_count == 1
    assert result.processed_inputs == 3
    assert len(rendering_configs) == 2
    assert all(item.normalization == "fixed" for item in rendering_configs)
    assert all(item.normalization_range == (0.0, 10.0) for item in rendering_configs)


def test_output_overwrite_error_and_skip_policies(tmp_path):
    from vision_lens.attention import GradCamResult

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
    from vision_lens import pipeline

    (tmp_path / "run-manifest.json").write_text("{}")
    config = parse_config(
        {
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
                "pretrained": False,
            },
            "input": {"files": ["data/examples/1.jpg"]},
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

    monkeypatch.setattr(pipeline, "load_model", fail_if_loaded)

    with pytest.raises(FileExistsError, match="Run manifest already exists"):
        run_vit_attention_from_config(config)

    assert not loaded


def test_gradcam_pipeline_passes_fixed_class_workers_and_batch_size(
    monkeypatch,
    tmp_path,
):
    from vision_lens import pipeline, processing
    from vision_lens.attention import GradCamResult

    config = parse_config(
        {
            "model": {
                "architecture": "cnn",
                "backend": "torchvision",
                "name": "mock_cnn",
                "pretrained": False,
            },
            "input": {"files": ["data/examples/1.jpg", "data/examples/2.jpg"]},
            "output": {
                "directory": str(tmp_path),
                "heatmaps": False,
                "overlays": False,
                "grids": False,
                "raw_arrays": True,
            },
            "preprocessing": {"image_size": 4},
            "analysis": {
                "method": "gradcam",
                "target_layer": "features.0",
                "target_class": 1,
            },
            "runtime": {"device": "cpu", "workers": 2, "batch_size": 1},
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

    monkeypatch.setattr(pipeline, "load_model", lambda _config: loaded_model)
    monkeypatch.setattr(
        processing,
        "load_images",
        lambda paths, workers: (
            worker_counts.append(workers)
            or [Image.new("RGB", (4, 4), "white") for _ in paths]
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_batch_preprocessor",
        lambda *_args, **_kwargs: lambda _image: torch.ones(3, 4, 4),
    )
    monkeypatch.setattr(pipeline, "extract_gradcam", fake_gradcam)

    result = run_gradcam_from_config(config)

    assert worker_counts == [2, 2]
    assert [call["target_classes"] for call in received] == [[1], [1]]
    assert all(call["target_layer"] == "features.0" for call in received)
    assert len(result.output_paths) == 2


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
