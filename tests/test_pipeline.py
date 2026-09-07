from pathlib import Path

import torch
from PIL import Image

from vision_lens.attention import AttentionExtractionResult, LayerAttentionMaps
from vision_lens.config import OutputConfig, load_preset, parse_config
from vision_lens.feature_pca import PatchPCAResult
from vision_lens.models import LoadedModel, ModelMetadata
from vision_lens.pipeline import (
    export_gradcam_outputs,
    run_patch_pca_from_config,
    run_pipeline_from_config,
    run_vit_attention_from_config,
)


def test_vit_pipeline_exports_figures_with_mocked_model(monkeypatch, tmp_path):
    from vision_lens import pipeline

    config = parse_config(
        {
            "task": "vit_attention",
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
                "pretrained": False,
                "options": {},
            },
            "images": {
                "paths": ["data/examples/1.jpg"],
            },
            "output": {
                "directory": str(tmp_path),
            },
            "attention": {
                "layers": [0],
                "heads": None,
                "head_fusion": "mean",
            },
            "runtime": {
                "device": "cpu",
                "image_size": 4,
            },
            "visualization": {
                "overlay_alpha": 0.35,
                "cmap": "viridis",
                "grid_format": "svg",
            },
        }
    )
    config = config.__class__(
        task=config.task,
        model=config.model,
        images=config.images,
        output=OutputConfig(tmp_path),
        attention=config.attention,
        runtime=config.runtime,
        visualization=config.visualization,
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

    monkeypatch.setattr(pipeline, "load_model", lambda _config: loaded_model)
    monkeypatch.setattr(
        pipeline,
        "load_images",
        lambda _paths: [Image.new("RGB", (4, 4), "white")],
    )
    monkeypatch.setattr(
        pipeline,
        "build_preprocess",
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
    from vision_lens import pipeline

    config = parse_config(
        {
            "task": "patch_pca",
            "model": {
                "architecture": "vit",
                "backend": "timm",
                "name": "mock_vit",
                "pretrained": False,
                "options": {},
            },
            "images": {"paths": ["horse-a.jpg", "horse-b.jpg"]},
            "output": {"directory": str(tmp_path)},
            "runtime": {"device": "cpu", "image_size": 4},
            "patch_pca": {
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
        pipeline,
        "load_images",
        lambda _paths: [Image.new("RGB", (4, 4), "white")] * 2,
    )
    monkeypatch.setattr(
        pipeline,
        "build_preprocess",
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
