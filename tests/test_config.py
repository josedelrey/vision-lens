from pathlib import Path

import pytest
import yaml

from vision_lens.config import (
    config_to_dict,
    load_config,
    load_preset,
    parse_config,
    resolved_config_yaml,
)
from vision_lens.presets import available_presets, get_preset


def _minimal_config(
    visualization=None,
    *,
    method="attention",
    architecture="vit",
    backend="timm",
):
    analysis = {"method": method}
    if method in {"attention", "rollout"}:
        analysis["layers"] = [0]
    return {
        "input": {"paths": ["data/examples/1.jpg"]},
        "model": {
            "architecture": architecture,
            "backend": backend,
            "name": "mock_model",
        },
        "preprocessing": {},
        "analysis": analysis,
        "runtime": {},
        "visualization": visualization or {},
        "output": {"directory": "outputs"},
    }


def test_parse_config_applies_documented_defaults():
    config = parse_config(_minimal_config())

    assert config.model.pretrained is True
    assert config.model.options is None
    assert config.preprocessing.image_size == 672
    assert config.analysis.method == "attention"
    assert config.analysis.heads is None
    assert config.analysis.head_fusion == "mean"
    assert config.runtime.device == "auto"
    assert config.runtime.batch_size is None
    assert config.runtime.workers == 0
    assert config.runtime.precision == "float32"
    assert config.runtime.seed is None
    assert config.preprocessing.resize == "stretch"
    assert config.preprocessing.crop == "none"
    assert config.preprocessing.pad == "none"
    assert config.preprocessing.interpolation is None
    assert config.preprocessing.normalize is True
    assert config.visualization.overlay_alpha == 0.45
    assert config.visualization.cmap == "viridis"
    assert config.visualization.grid_format == "png"
    assert config.visualization.columns is None
    assert config.visualization.items_per_grid is None
    assert config.visualization.normalization == "per_map"
    assert config.output.heatmaps is True
    assert config.output.overlays is True
    assert config.output.grids is True
    assert config.output.raw_arrays is False
    assert config.output.overwrite == "replace"


def test_parse_config_reads_visualization_values():
    config = parse_config(
        _minimal_config(
            {
                "overlay_alpha": 0.35,
                "cmap": "magma",
                "grid_format": "svg",
            }
        )
    )

    assert config.visualization.overlay_alpha == 0.35
    assert config.visualization.cmap == "magma"
    assert config.visualization.grid_format == "svg"


def test_patch_pca_defaults_and_overrides_are_in_analysis_section():
    raw_config = _minimal_config(method="patch_pca")
    config = parse_config(raw_config)

    assert config.analysis.foreground_threshold == 0.5
    assert config.analysis.foreground_side == "high"
    assert config.output.overlays is False

    raw_config["analysis"].update(
        {"foreground_threshold": 0.65, "foreground_side": "low"}
    )
    overridden = parse_config(raw_config)
    assert overridden.analysis.foreground_threshold == 0.65
    assert overridden.analysis.foreground_side == "low"


def test_load_config_resolves_all_paths_from_config_file(tmp_path):
    config_dir = tmp_path / "nested" / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "experiment.yaml"
    raw_config = _minimal_config()
    raw_config["input"]["paths"] = ["../images/cat.jpg"]
    raw_config["output"]["directory"] = "../figures"
    image_path = tmp_path / "nested" / "images" / "cat.jpg"
    image_path.parent.mkdir()
    image_path.touch()
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    config = load_config(config_path)

    assert config.input.paths == (image_path,)
    assert config.output.directory == tmp_path / "nested" / "figures"


@pytest.mark.parametrize(
    ("preset_name", "config_name"),
    [
        ("dino-vits8-attention", "vit_attention.example.yaml"),
        (
            "dinov2-reg4-attention",
            "vit_attention.dinov2_reg4.example.yaml",
        ),
        ("dinov2-reg4-rollout", "vit_rollout.dinov2_reg4.example.yaml"),
        ("resnet50-gradcam", "gradcam.example.yaml"),
        ("dinov2-pca", "patch_pca.dinov2.example.yaml"),
    ],
)
def test_presets_resolve_to_the_existing_example_workflows(preset_name, config_name):
    repo_root = Path(__file__).parents[1]

    preset_config = load_preset(preset_name, base_dir=repo_root)
    example_config = load_config(repo_root / "configs" / config_name)

    assert preset_config == example_config


def test_user_config_then_cli_values_override_preset_at_setting_level():
    config = parse_config(
        {
            "preset": "dino-vits8-attention",
            "analysis": {"heads": [1, 3], "head_fusion": "none"},
        },
        overrides={
            "runtime": {"device": "cpu"},
            "visualization": {"overlay_alpha": 0.25},
        },
    )

    assert config.preset == "dino-vits8-attention"
    assert config.analysis.layers == (2, 5, 8, 11)
    assert config.analysis.heads == (1, 3)
    assert config.analysis.head_fusion == "none"
    assert config.preprocessing.image_size == 224
    assert config.runtime.device == "cpu"
    assert config.visualization.overlay_alpha == 0.25
    assert config.visualization.cmap == "viridis"


def test_cli_selected_preset_replaces_config_selected_preset():
    config = parse_config(
        {"preset": "resnet50-gradcam"},
        preset="dino-vits8-attention",
    )

    assert config.preset == "dino-vits8-attention"
    assert config.analysis.method == "attention"


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (None, "imagse"),
        ("input", "recusrive"),
        ("model", "weights"),
        ("preprocessing", "cropping"),
        ("runtime", "worker_count"),
        ("visualization", "column_count"),
        ("output", "format"),
    ],
)
def test_unknown_keys_are_rejected(section, key):
    raw_config = _minimal_config()
    if section is None:
        raw_config[key] = True
    else:
        raw_config[section][key] = True

    with pytest.raises(ValueError, match=key):
        parse_config(raw_config)


def test_keys_for_a_different_analysis_method_are_rejected():
    raw_config = _minimal_config(
        method="gradcam",
        architecture="cnn",
        backend="torchvision",
    )
    raw_config["analysis"]["layers"] = [0]

    with pytest.raises(ValueError, match="Unknown key.*layers"):
        parse_config(raw_config)


@pytest.mark.parametrize(
    ("method", "architecture", "backend"),
    [
        ("attention", "cnn", "torchvision"),
        ("rollout", "cnn", "torchvision"),
        ("patch_pca", "cnn", "torchvision"),
        ("gradcam", "vit", "timm"),
    ],
)
def test_invalid_analysis_model_combinations_are_rejected(
    method,
    architecture,
    backend,
):
    with pytest.raises(ValueError, match=f"analysis.method='{method}' requires"):
        parse_config(
            _minimal_config(
                method=method,
                architecture=architecture,
                backend=backend,
            )
        )


def test_fixed_model_image_size_is_validated_before_loading():
    raw_config = _minimal_config()
    raw_config["model"]["name"] = "vit_small_patch8_224.dino"
    raw_config["preprocessing"]["image_size"] = 672

    with pytest.raises(ValueError, match="requires preprocessing.image_size=224"):
        parse_config(raw_config)


def test_model_img_size_option_is_rejected_in_favor_of_authoritative_setting():
    raw_config = _minimal_config()
    raw_config["model"]["options"] = {"img_size": 672}

    with pytest.raises(ValueError, match="preprocessing.image_size"):
        parse_config(raw_config)


def test_known_model_layer_and_head_constraints_are_validated():
    raw_config = _minimal_config()
    raw_config["model"]["name"] = "hf_hub:timm/vit_small_patch14_reg4_dinov2.lvd142m"
    raw_config["analysis"]["layers"] = [12]

    with pytest.raises(ValueError, match="layers 0 through 11"):
        parse_config(raw_config)

    raw_config["analysis"]["layers"] = [0]
    raw_config["analysis"]["heads"] = [6]
    with pytest.raises(ValueError, match="heads 0 through 5"):
        parse_config(raw_config)


def test_resolved_config_contains_defaults_and_absolute_paths():
    config = parse_config(_minimal_config())

    resolved = config_to_dict(config)
    rendered = yaml.safe_load(resolved_config_yaml(config))

    assert resolved == rendered
    assert Path(resolved["input"]["files"][0]).is_absolute()
    assert Path(resolved["output"]["directory"]).is_absolute()
    assert resolved["preprocessing"]["image_size"] == 672
    assert resolved["preprocessing"]["resize"] == "stretch"
    assert resolved["analysis"]["head_fusion"] == "mean"


def test_missing_input_is_rejected_before_model_loading(tmp_path):
    raw_config = _minimal_config()
    raw_config["input"]["paths"] = ["missing.jpg"]

    with pytest.raises(ValueError, match="Input file.*missing.jpg"):
        parse_config(raw_config, base_dir=tmp_path)


def test_unknown_preset_lists_available_choices():
    with pytest.raises(ValueError, match="Unknown preset 'missing'") as error:
        get_preset("missing")

    assert all(name in str(error.value) for name in available_presets())


def test_input_folders_patterns_recursion_and_limit_are_resolved(tmp_path):
    image_dir = tmp_path / "images"
    nested = image_dir / "nested"
    nested.mkdir(parents=True)
    (image_dir / "a.jpg").touch()
    (image_dir / "ignored.txt").touch()
    (nested / "b.jpg").touch()
    raw_config = _minimal_config()
    raw_config["input"] = {
        "folders": ["images"],
        "patterns": ["*.jpg"],
        "recursive": True,
        "limit": 1,
    }

    config = parse_config(raw_config, base_dir=tmp_path)

    assert config.input.paths == (image_dir / "a.jpg",)
    assert config.input.folders == (image_dir,)
    assert config.input.patterns == ("*.jpg",)
    assert config.input.recursive is True
    assert config.input.limit == 1


def test_all_new_controls_are_parsed_and_resolved(tmp_path):
    raw_config = _minimal_config(
        method="gradcam",
        architecture="cnn",
        backend="torchvision",
    )
    raw_config["analysis"].update({"target_layer": "layer3", "target_class": 7})
    raw_config["input"]["paths"] = [str(Path("data/examples/1.jpg").resolve())]
    raw_config["preprocessing"] = {
        "image_size": 224,
        "resize": "longest",
        "crop": "none",
        "pad": "center",
        "interpolation": "bicubic",
        "normalize": True,
        "mean": [0.1, 0.2, 0.3],
        "std": [0.9, 0.8, 0.7],
    }
    raw_config["runtime"] = {
        "batch_size": 2,
        "device": "cpu",
        "workers": 3,
        "precision": "bfloat16",
        "seed": 42,
    }
    raw_config["visualization"] = {
        "tile_size": [320, 240],
        "columns": 2,
        "items_per_grid": 6,
        "spacing": 8,
        "padding": 10,
        "labels": False,
        "background": "#101010",
        "dpi": 150,
        "overlay_alpha": 0.25,
        "cmap": "magma",
        "grid_format": "svg",
        "normalization": "fixed",
        "normalization_range": [-1, 2],
    }
    raw_config["output"].update(
        {
            "heatmaps": False,
            "overlays": True,
            "grids": False,
            "raw_arrays": True,
            "image_format": "webp",
            "raw_format": "npz",
            "overwrite": "error",
        }
    )

    config = parse_config(raw_config, base_dir=tmp_path)

    assert config.analysis.target_class == 7
    assert config.preprocessing.mean == (0.1, 0.2, 0.3)
    assert config.runtime.batch_size == 2
    assert config.runtime.precision == "bfloat16"
    assert config.visualization.tile_size == (320, 240)
    assert config.visualization.items_per_grid == 6
    assert config.visualization.normalization_range == (-1.0, 2.0)
    assert config.output.image_format == "webp"
    assert config.output.raw_format == "npz"


def test_fixed_normalization_requires_a_range():
    raw_config = _minimal_config({"normalization": "fixed"})

    with pytest.raises(ValueError, match="normalization_range is required"):
        parse_config(raw_config)


def test_pca_projection_paths_resolve_from_config_and_load_must_exist(tmp_path):
    raw_config = _minimal_config(method="patch_pca")
    raw_config["input"]["paths"] = [str(Path("data/examples/1.jpg").resolve())]
    raw_config["analysis"].update(
        {
            "projection": "fit",
            "save_projection": "artifacts/pca.npz",
        }
    )
    config = parse_config(raw_config, base_dir=tmp_path)
    assert config.analysis.save_projection == tmp_path / "artifacts" / "pca.npz"

    raw_config["analysis"] = {
        "method": "patch_pca",
        "projection": "load",
        "projection_path": "missing.npz",
    }
    with pytest.raises(ValueError, match="projection file does not exist"):
        parse_config(raw_config, base_dir=tmp_path)


def test_patch_pca_rejects_unsupported_overlay_output():
    raw_config = _minimal_config(method="patch_pca")
    raw_config["output"]["overlays"] = True

    with pytest.raises(ValueError, match="overlays is not supported"):
        parse_config(raw_config)


def test_at_least_one_output_type_must_be_enabled():
    raw_config = _minimal_config()
    raw_config["output"].update(
        {"heatmaps": False, "overlays": False, "grids": False, "raw_arrays": False}
    )

    with pytest.raises(ValueError, match="At least one output type"):
        parse_config(raw_config)
