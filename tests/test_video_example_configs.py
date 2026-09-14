from pathlib import Path

import pytest

from vision_lens.config import load_config

REPO_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("config_name", "method", "scene"),
    [
        ("vit_attention.video.example.yaml", "attention", "lego"),
        ("vit_attention.dinov2_reg4.video.example.yaml", "attention", "fern"),
        ("vit_rollout.dinov2_reg4.video.example.yaml", "rollout", "fern"),
        ("gradcam.video.example.yaml", "gradcam", "lego"),
        ("patch_pca.dinov2.video.example.yaml", "patch_pca", "fern"),
    ],
)
def test_video_example_config_selects_one_input_and_video_mode(
    tmp_path, config_name, method, scene
):
    source = tmp_path / f"{scene}.mp4"
    source.touch()
    output = tmp_path / "outputs"

    config = load_config(
        REPO_ROOT / "configs" / config_name,
        overrides={
            "input": {"files": [str(source)]},
            "output": {"directory": str(output)},
        },
    )

    assert config.input.paths == (source,)
    assert config.analysis.method == method
    assert config.video is not None
    assert config.video.sampling_rate == 10
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
