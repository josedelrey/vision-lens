import os
from pathlib import Path

import pytest

from vision_lens.config import load_config
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
    ("config_name", "analysis_overrides", "expected_suffix"),
    [
        (
            "vit_attention.example.yaml",
            {"layers": [11], "heads": None, "head_fusion": "mean"},
            "_heatmap.png",
        ),
        (
            "vit_attention.dinov2_reg4.example.yaml",
            {"layers": [11], "heads": None, "head_fusion": "mean"},
            "_heatmap.png",
        ),
        (
            "vit_rollout.dinov2_reg4.example.yaml",
            {"layers": [11], "heads": None, "head_fusion": "mean"},
            "_heatmap.png",
        ),
        (
            "gradcam.example.yaml",
            {"target_layer": "layer4", "target_class": None},
            "_heatmap.png",
        ),
        (
            "patch_pca.dinov2.example.yaml",
            {
                "foreground_threshold": 0.5,
                "foreground_side": "high",
                "projection": "fit",
                "projection_path": None,
                "save_projection": None,
            },
            "_patch_pca.png",
        ),
    ],
)
def test_pretrained_model_pipeline_smoke(
    config_name,
    analysis_overrides,
    expected_suffix,
    tmp_path,
):
    config = load_config(
        REPO_ROOT / "configs" / config_name,
        overrides={
            "input": {"files": ["examples/1.jpg"], "limit": 1},
            "preprocessing": {"image_size": 224},
            "analysis": analysis_overrides,
            "runtime": {"batch_size": 1, "device": "cpu"},
            "output": {
                "directory": str(tmp_path / config_name),
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
