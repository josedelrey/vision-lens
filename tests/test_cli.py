import pytest

from vision_lens.cli import _parse_overrides, main


def test_cli_parses_repeated_nested_overrides_as_yaml_values():
    assert _parse_overrides(
        [
            "runtime.image_size=672",
            "attention.layers=[2, 5, 8, 11]",
            "model.pretrained=false",
        ]
    ) == {
        "runtime": {"image_size": 672},
        "attention": {"layers": [2, 5, 8, 11]},
        "model": {"pretrained": False},
    }


def test_cli_rejects_malformed_override():
    with pytest.raises(ValueError, match="expected SECTION.KEY=VALUE"):
        _parse_overrides(["runtime.image_size"])


def test_cli_lists_presets_without_running_a_model(capsys):
    assert main(["--list-presets"]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "dino-vits8-attention",
        "dinov2-reg4-attention",
        "dinov2-reg4-rollout",
        "resnet50-gradcam",
        "dinov2-pca",
    ]
