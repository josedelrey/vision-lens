import os
from pathlib import Path

import pytest

from vision_lens.config import load_preset
from vision_lens.pipeline import run_pipeline_from_config

REPO_ROOT = Path(__file__).parents[1]
RUN_REAL_MODELS = os.environ.get("VISION_LENS_RUN_REAL_MODELS") == "1"

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        not RUN_REAL_MODELS,
        reason="set VISION_LENS_RUN_REAL_MODELS=1 to download pretrained weights",
    ),
]


@pytest.mark.parametrize(
    ("preset", "analysis_overrides", "expected_suffix"),
    [
        (
            "dino-vits8-attention",
            {"layers": [11], "heads": None, "head_fusion": "mean"},
            "_heatmap.png",
        ),
        (
            "dinov2-reg4-attention",
            {"layers": [11], "heads": None, "head_fusion": "mean"},
            "_heatmap.png",
        ),
        (
            "dinov2-reg4-rollout",
            {"layers": [11], "heads": None, "head_fusion": "mean"},
            "_heatmap.png",
        ),
        (
            "resnet50-gradcam",
            {"target_layer": "layer4", "target_class": None},
            "_heatmap.png",
        ),
        (
            "dinov2-pca",
            {
                "foreground_threshold": 0.5,
                "foreground_side": "low",
                "projection": "fit",
                "projection_path": None,
                "save_projection": None,
            },
            "_patch_pca.png",
        ),
    ],
)
def test_pretrained_model_pipeline_smoke(
    preset,
    analysis_overrides,
    expected_suffix,
    tmp_path,
):
    config = load_preset(
        preset,
        base_dir=REPO_ROOT,
        overrides={
            "input": {"files": ["data/examples/1.jpg"], "limit": 1},
            "preprocessing": {"image_size": 224},
            "analysis": analysis_overrides,
            "runtime": {"batch_size": 1, "device": "cpu"},
            "output": {
                "directory": str(tmp_path / preset),
                "heatmaps": True,
                "overlays": False,
                "grids": False,
                "raw_arrays": False,
                "overwrite": "error",
            },
        },
    )

    result = run_pipeline_from_config(config)

    assert result.loaded_model.metadata.pretrained is True
    assert any(path.name.endswith(expected_suffix) for path in result.output_paths)
    assert all(path.is_file() for path in result.output_paths)
    assert (config.output.directory / "run-manifest.json").is_file()
