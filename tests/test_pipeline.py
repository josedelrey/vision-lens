from pathlib import Path

import torch
from PIL import Image

from vision_lens.attention import AttentionExtractionResult, LayerAttentionMaps
from vision_lens.config import OutputConfig, parse_config
from vision_lens.models import LoadedModel, ModelMetadata
from vision_lens.pipeline import run_vit_attention_from_config


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
