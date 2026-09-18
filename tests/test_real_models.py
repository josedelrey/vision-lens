import os
from pathlib import Path

import pytest

from vision_lens.config import load_config
from vision_lens.pipeline import run_pipeline_from_config

REPO_ROOT = Path(__file__).parents[1]
RUN_REAL_MODELS = os.environ.get("VISION_LENS_RUN_REAL_MODELS") == "1"

SMOKE_CASES = [
    (
        "vit_attention.yaml",
        {"layers": [11], "heads": None, "head_fusion": "mean"},
        "_heatmap.png",
    ),
    (
        "vit_attention.dinov2_reg4.yaml",
        {"layers": [11], "heads": None, "head_fusion": "mean"},
        "_heatmap.png",
    ),
    (
        "vit_rollout.dinov2_reg4.yaml",
        {"layers": [11]},
        "_heatmap.png",
    ),
    (
        "gradcam.yaml",
        {"target_layer": "layer4", "target_class": None},
        "_heatmap.png",
    ),
    (
        "patch_pca.dinov2.yaml",
        {
            "foreground_separation": True,
            "foreground_threshold": 0.5,
            "foreground_side": "high",
            "projection": "fit",
            "save_projection": None,
        },
        "_patch_pca.png",
    ),
]


@pytest.mark.parametrize(
    ("config_name", "analysis_overrides", "expected_suffix"),
    SMOKE_CASES,
)
@pytest.mark.real_model
@pytest.mark.skipif(
    not RUN_REAL_MODELS,
    reason="set VISION_LENS_RUN_REAL_MODELS=1 to download pretrained weights",
)
def test_pretrained_model_pipeline_smoke(
    config_name,
    analysis_overrides,
    expected_suffix,
    tmp_path,
):
    config = _smoke_config(config_name, analysis_overrides, tmp_path)

    result = run_pipeline_from_config(config)

    assert result.loaded_model.metadata.pretrained is True
    assert any(path.name.endswith(expected_suffix) for path in result.output_paths)
    assert all(path.is_file() for path in result.output_paths)
    assert (config.output.directory / "run-manifest.json").is_file()


@pytest.mark.parametrize(
    ("config_name", "analysis_overrides"),
    [(name, overrides) for name, overrides, _ in SMOKE_CASES],
)
def test_pretrained_model_smoke_configuration_is_valid(
    config_name,
    analysis_overrides,
    tmp_path,
):
    _smoke_config(config_name, analysis_overrides, tmp_path)


def _smoke_config(config_name, analysis_overrides, output_root):
    output_overrides = {
        "directory": str(output_root / config_name),
        "heatmaps": True,
        "grids": False,
        "raw_arrays": False,
        "overwrite": "error",
    }
    return load_config(
        REPO_ROOT / "configs" / config_name,
        overrides={
            "input": {"files": ["examples/1.jpg"], "limit": 1},
            "preprocessing": {"image_size": 224},
            "analysis": analysis_overrides,
            "runtime": {"batch_size": 1, "device": "cpu"},
            "output": output_overrides,
        },
    )
