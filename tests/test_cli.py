import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from vision_lens import load_config, resolved_config_yaml
from vision_lens.cli import CONFIG_OPTION_PATHS, _build_parser, main

BUNDLED_CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*.yaml"))


@pytest.fixture
def config_path(tmp_path):
    source = tmp_path / "input.jpg"
    source.touch()
    path = tmp_path / "workflow.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "input": {"files": [str(source)]},
                "model": {
                    "architecture": "vit",
                    "backend": "timm",
                    "name": "mock_vit",
                },
                "analysis": {"method": "patch_pca"},
                "output": {"directory": str(tmp_path / "results")},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_without_config_validates_the_supplied_mapping(capsys):
    with pytest.raises(SystemExit, match="2"):
        main(["validate", "--runtime-device", "cpu"])

    error = capsys.readouterr().err
    assert "input must select at least one existing file" in error


def test_cli_supports_named_flags_without_config(capsys, tmp_path):
    source = tmp_path / "input.jpg"
    source.touch()

    assert (
        main(
            [
                "resolve",
                "--input-files",
                f"[{source}]",
                "--model-architecture",
                "vit",
                "--model-backend",
                "timm",
                "--model-name",
                "mock_vit",
                "--preprocessing-interpolation",
                "bilinear",
                "--analysis-method",
                "attention",
                "--analysis-layers",
                "[0]",
                "--runtime-device",
                "cpu",
                "--visualization-output-size",
                "match",
                "--visualization-interpolation",
                "nearest",
                "--visualization-cmap-black",
                "{threshold: 20, blend_width: 35, transparent: true}",
                "--output-directory",
                str(tmp_path / "results"),
                "--output-transparent-overlays",
                "true",
                "--output-overwrite",
                "replace",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "architecture: vit" in output
    assert "method: attention" in output
    assert "device: cpu" in output
    assert "transparent_overlays: true" in output


def test_named_flags_override_yaml_values(capsys, config_path):
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["runtime"] = {"device": "cpu"}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert (
        main(
            [
                "resolve",
                "--config",
                str(config_path),
                "--runtime-device",
                "cuda",
            ]
        )
        == 0
    )

    assert "device: cuda" in capsys.readouterr().out


def test_video_flag_adds_default_video_section(capsys, tmp_path):
    source = tmp_path / "input.mp4"
    source.touch()

    assert (
        main(
            [
                "resolve",
                "--video",
                "--input-files",
                f"[{source}]",
                "--model-architecture",
                "vit",
                "--model-backend",
                "timm",
                "--model-name",
                "mock_vit",
                "--analysis-method",
                "attention",
                "--analysis-layers",
                "[0]",
                "--output-directory",
                str(tmp_path / "results"),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert output.startswith("# detected media: 0 images, 1 video\n")
    assert "video:" in output
    assert "sampling_rate: 5.0" in output


def test_video_input_adds_default_video_settings_automatically(capsys, tmp_path):
    source = tmp_path / "input.mp4"
    source.touch()

    assert (
        main(
            [
                "resolve",
                "--input-files",
                f"[{source}]",
                "--model-architecture",
                "vit",
                "--model-backend",
                "timm",
                "--model-name",
                "mock_vit",
                "--analysis-method",
                "attention",
                "--analysis-layers",
                "[0]",
                "--output-directory",
                str(tmp_path / "results"),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert output.startswith("# detected media: 0 images, 1 video\n")
    assert "video:" in output
    assert "sampling_rate: 5.0" in output


def test_every_config_setting_has_a_named_cli_flag():
    parser = _build_parser()
    available = {
        option for action in parser._actions for option in action.option_strings
    }
    expected = {
        f"--{section}-{key.replace('_', '-')}" for section, key in CONFIG_OPTION_PATHS
    }

    assert expected <= available
    assert "--video" in available


@pytest.mark.parametrize("config_path", BUNDLED_CONFIGS, ids=lambda path: path.stem)
def test_bundled_configs_match_their_cli_only_forms(capsys, config_path):
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    arguments = ["resolve"]
    for section, values in raw.items():
        if section == "video" and not values:
            arguments.append("--video")
        for key, value in values.items():
            arguments.extend(
                [
                    f"--{section}-{key.replace('_', '-')}",
                    json.dumps(value),
                ]
            )

    assert main(arguments) == 0

    expected = resolved_config_yaml(load_config(config_path))
    assert capsys.readouterr().out == expected


@pytest.mark.parametrize(
    "arguments",
    [
        ["validate", "--runtime-device", "["],
        ["validate", "--output-directory", "{}"],
    ],
)
def test_malformed_cli_config_fails_without_a_traceback(capsys, arguments):
    with pytest.raises(SystemExit, match="2"):
        main(arguments)

    captured = capsys.readouterr()
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("contents", [None, "input: [unterminated"])
def test_unreadable_config_fails_without_a_traceback(capsys, tmp_path, contents):
    config_path = tmp_path / "workflow.yaml"
    if contents is not None:
        config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main(["validate", "--config", str(config_path)])

    assert "Traceback" not in capsys.readouterr().err


def test_cli_hides_individual_output_paths_by_default(monkeypatch, capsys, config_path):
    from vision_lens import pipeline

    monkeypatch.setattr(
        pipeline,
        "run_pipeline_from_config",
        lambda _config, **_kwargs: _result(),
    )

    assert main(["--config", str(config_path)]) == 0

    output_dir = Path("outputs/example")
    assert capsys.readouterr().out.splitlines() == [
        f"saved 2 outputs plus run manifest to {output_dir}"
    ]


def test_cli_validates_without_running_a_model(capsys, config_path):
    assert main(["validate", "--config", str(config_path)]) == 0
    assert capsys.readouterr().out == ("configuration is valid (1 image, 0 videos)\n")


def test_cli_validation_does_not_import_pipeline_modules(
    monkeypatch, capsys, config_path
):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("vision_lens.pipeline"):
            pytest.fail(f"validation imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert main(["validate", "--config", str(config_path)]) == 0
    assert capsys.readouterr().out == ("configuration is valid (1 image, 0 videos)\n")


def test_cli_prints_resolved_configuration(capsys, config_path):
    assert (
        main(
            [
                "resolve",
                "--config",
                str(config_path),
                "--runtime-device",
                "cpu",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "method: patch_pca" in output
    assert "image_size: 672" in output
    assert "device: cpu" in output


def test_cli_reports_unknown_flags_without_a_traceback(capsys, config_path):
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "validate",
                "--config",
                str(config_path),
                "--visualization-colrmap",
                "viridis",
            ]
        )

    error = capsys.readouterr().err
    assert "unrecognized arguments: --visualization-colrmap viridis" in error
    assert "Traceback" not in error


def _result():
    output_dir = Path("outputs/example")
    return SimpleNamespace(
        output_paths=(output_dir / "first.png", output_dir / "second.png"),
        config=SimpleNamespace(output=SimpleNamespace(directory=output_dir)),
    )
