from pathlib import Path

import pytest
import yaml

from vision_lens.config import load_config

REPO_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("config_name", "method"),
    [
        ("vit_attention.video.example.yaml", "attention"),
        ("vit_attention.dinov2_reg4.video.example.yaml", "attention"),
        ("vit_rollout.dinov2_reg4.video.example.yaml", "rollout"),
        ("gradcam.video.example.yaml", "gradcam"),
        ("patch_pca.dinov2.video.example.yaml", "patch_pca"),
    ],
)
def test_video_example_config_selects_all_mp4_inputs_and_video_mode(
    tmp_path, config_name, method
):
    sources = tuple(tmp_path / f"{index}.mp4" for index in range(1, 3))
    for source in sources:
        source.touch()
    (tmp_path / "ignore.txt").touch()
    output = tmp_path / "outputs"
    example = REPO_ROOT / "configs" / config_name
    raw = yaml.safe_load(example.read_text())
    assert raw["input"]["folders"] == ["videos"]
    assert raw["input"]["patterns"] == ["*.mp4"]

    config = load_config(
        example,
        overrides={
            "input": {"folders": [str(tmp_path)]},
            "output": {"directory": str(output)},
        },
    )

    assert config.input.paths == sources
    assert config.analysis.method == method
    assert config.video is not None
    assert config.video.sampling_rate == "auto"
    assert config.runtime.batch_size == 1
    assert config.output.directory == output
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
        "vit_attention.example.yaml",
        "vit_attention.dinov2_reg4.example.yaml",
        "vit_rollout.dinov2_reg4.example.yaml",
        "gradcam.example.yaml",
        "patch_pca.dinov2.example.yaml",
    ],
)
def test_image_example_config_selects_every_example_image(config_name):
    config = load_config(REPO_ROOT / "configs" / config_name)
    expected = tuple(sorted((REPO_ROOT / "examples").glob("*.jpg")))
    assert config.input.paths == expected
