import os
from pathlib import Path

import pytest

from vision_lens.config import parse_config
from vision_lens.pipeline import run_pipeline_from_config

RUN_REAL_MODELS = os.environ.get("VISION_LENS_RUN_REAL_MODELS") == "1"
IMAGE = Path(__file__).parents[1] / "examples/1.jpg"

SMOKE_CASES = [
    (
        "dino_attention",
        "vit_small_patch8_224.dino",
        {"method": "attention", "layers": [11]},
        "_heatmap.png",
    ),
    (
        "dinov2_attention",
        "hf_hub:timm/vit_small_patch14_reg4_dinov2.lvd142m",
        {"method": "attention", "layers": [11]},
        "_heatmap.png",
    ),
    (
        "dinov2_rollout",
        "hf_hub:timm/vit_small_patch14_reg4_dinov2.lvd142m",
        {"method": "rollout", "layers": [11]},
        "_heatmap.png",
    ),
    (
        "resnet_gradcam",
        "resnet50",
        {"method": "gradcam", "target_layer": "layer4"},
        "_heatmap.png",
    ),
    (
        "dinov2_pca",
        "hf_hub:timm/vit_base_patch14_dinov2.lvd142m",
        {"method": "patch_pca"},
        "_patch_pca.png",
    ),
]


@pytest.mark.parametrize(
    ("case_name", "model_name", "analysis", "expected_suffix"), SMOKE_CASES
)
@pytest.mark.real_model
@pytest.mark.skipif(
    not RUN_REAL_MODELS,
    reason="set VISION_LENS_RUN_REAL_MODELS=1 to download pretrained weights",
)
def test_pretrained_model_pipeline_smoke(
    case_name, model_name, analysis, expected_suffix, tmp_path
):
    config = _smoke_config(case_name, model_name, analysis, tmp_path)

    result = run_pipeline_from_config(config)

    assert result.loaded_model.metadata.pretrained is True
    assert any(path.name.endswith(expected_suffix) for path in result.output_paths)
    assert all(path.is_file() for path in result.output_paths)
    assert (config.output.directory / "run-manifest.json").is_file()


@pytest.mark.parametrize(
    ("case_name", "model_name", "analysis"),
    [(name, model, analysis) for name, model, analysis, _ in SMOKE_CASES],
)
def test_pretrained_model_smoke_configuration_is_valid(
    case_name, model_name, analysis, tmp_path
):
    _smoke_config(case_name, model_name, analysis, tmp_path)


def _smoke_config(case_name, model_name, analysis, output_root):
    is_gradcam = analysis["method"] == "gradcam"
    return parse_config(
        {
            "input": {"files": [str(IMAGE)]},
            "model": {
                "architecture": "cnn" if is_gradcam else "vit",
                "backend": "torchvision" if is_gradcam else "timm",
                "name": model_name,
            },
            "preprocessing": {"image_size": 224},
            "analysis": analysis,
            "runtime": {"batch_size": 1, "device": "cpu"},
            "output": {
                "directory": str(output_root / case_name),
                "heatmaps": True,
                "grids": False,
                "raw_arrays": False,
            },
        }
    )
