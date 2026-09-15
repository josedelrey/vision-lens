from pathlib import Path

import pytest

from vision_lens.config import load_config

REPO_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("config_name", "method"),
    [
        ("vit_attention.video.yaml", "attention"),
        ("vit_attention.dinov2_reg4.video.yaml", "attention"),
        ("vit_rollout.dinov2_reg4.video.yaml", "rollout"),
        ("gradcam.video.yaml", "gradcam"),
        ("patch_pca.dinov2.video.yaml", "patch_pca"),
    ],
)
def test_video_example_config_uses_expected_video_mode(tmp_path, config_name, method):
    source = tmp_path / "sample.mp4"
    source.touch()
    output = tmp_path / "outputs"
    example = REPO_ROOT / "configs" / config_name

    config = load_config(
        example,
        overrides={
            "input": {"files": [str(source)], "folders": []},
            "output": {"directory": str(output)},
        },
    )

    assert config.input.paths == (source,)
    assert config.analysis.method == method
    assert config.video is not None
    assert config.video.sampling_rate == "auto"
    assert config.runtime.batch_size == 1
    assert config.output.directory == output
    assert config.visualization.output_size == "match"
    assert config.visualization.interpolation == "anyup_soft"
    assert config.visualization.anyup_query_chunk_size == 4096
    assert config.output.grids
    if method == "patch_pca":
        assert config.output.heatmaps
        assert not config.output.overlays
        assert config.video.pca_fit_frames == 32
    else:
        assert not config.output.heatmaps
        assert config.output.overlays


@pytest.mark.parametrize(
    "config_name",
    [
        "vit_attention.yaml",
        "vit_attention.dinov2_reg4.yaml",
        "vit_rollout.dinov2_reg4.yaml",
        "gradcam.yaml",
        "patch_pca.dinov2.yaml",
    ],
)
def test_image_example_config_selects_every_example_image(config_name):
    config = load_config(REPO_ROOT / "configs" / config_name)
    expected = (
        (REPO_ROOT / "examples/5.jpg", REPO_ROOT / "examples/6.jpg")
        if config_name == "patch_pca.dinov2.yaml"
        else tuple(sorted((REPO_ROOT / "examples").glob("*.jpg")))
    )
    assert config.input.paths == expected
    assert config.visualization.output_size == "match"
    assert config.visualization.interpolation == "anyup_soft"
    assert config.visualization.anyup_query_chunk_size == 4096
