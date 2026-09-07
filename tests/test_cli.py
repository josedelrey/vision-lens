from pathlib import Path
from types import SimpleNamespace

import pytest

from vision_lens.cli import _parse_overrides, main


def test_cli_parses_repeated_nested_overrides_as_yaml_values():
    assert _parse_overrides(
        [
            "preprocessing.image_size=672",
            "analysis.layers=[2, 5, 8, 11]",
            "model.pretrained=false",
        ]
    ) == {
        "preprocessing": {"image_size": 672},
        "analysis": {"layers": [2, 5, 8, 11]},
        "model": {"pretrained": False},
    }


def test_cli_rejects_malformed_override():
    with pytest.raises(ValueError, match="expected SECTION.KEY=VALUE"):
        _parse_overrides(["preprocessing.image_size"])


def test_cli_lists_presets_without_running_a_model(capsys):
    assert main(["--list-presets"]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "dino-vits8-attention",
        "dinov2-reg4-attention",
        "dinov2-reg4-rollout",
        "resnet50-gradcam",
        "dinov2-pca",
    ]


def test_cli_hides_individual_output_paths_by_default(monkeypatch, capsys):
    from vision_lens import cli

    monkeypatch.setattr(cli, "run_pipeline_from_config", lambda _config: _result())

    assert main(["--preset", "dinov2-pca"]) == 0

    output_dir = Path("outputs/example")
    assert capsys.readouterr().out.splitlines() == [f"saved 2 files to {output_dir}"]


def test_cli_validates_without_running_a_model(monkeypatch, capsys):
    from vision_lens import cli

    monkeypatch.setattr(
        cli,
        "run_pipeline_from_config",
        lambda _config: pytest.fail("model pipeline should not run"),
    )

    assert main(["validate", "--preset", "dinov2-pca"]) == 0
    assert capsys.readouterr().out == "configuration is valid\n"


def test_cli_prints_resolved_configuration(capsys):
    assert (
        main(
            [
                "resolve",
                "--preset",
                "dinov2-pca",
                "--set",
                "runtime.device=cpu",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "method: patch_pca" in output
    assert "image_size: 672" in output
    assert "device: cpu" in output


def test_cli_reports_configuration_errors_without_a_traceback(capsys):
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "validate",
                "--preset",
                "dinov2-pca",
                "--set",
                "visualization.colrmap=viridis",
            ]
        )

    error = capsys.readouterr().err
    assert "Unknown key(s) in visualization: colrmap" in error
    assert "Traceback" not in error


def _result():
    output_dir = Path("outputs/example")
    return SimpleNamespace(
        output_paths=(output_dir / "first.png", output_dir / "second.png"),
        config=SimpleNamespace(output=SimpleNamespace(directory=output_dir)),
    )
