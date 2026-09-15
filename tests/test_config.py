from pathlib import Path

import pytest
import yaml

from vision_lens.config import (
    config_to_dict,
    load_config,
    parse_config,
    resolved_config_yaml,
)


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
        "input": {"paths": ["examples/1.jpg"]},
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
    assert config.runtime.batch_size == 8
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
    assert config.output.overwrite == "error"
    assert config.video is None


def test_video_settings_are_strict_and_resolved(tmp_path):
    source = tmp_path / "clip.mp4"
    source.touch()
    raw = _minimal_config()
    raw["input"] = {"files": [str(source)]}
    raw["video"] = {
        "start_time": 1.5,
        "end_time": 4,
        "sampling_rate": 2.5,
        "frame_limit": 7,
        "output_resolution": [640, 360],
        "pca_fit_frames": 5,
        "temporal_smoothing": 0.25,
        "codec": "libx264",
    }

    config = parse_config(raw)
    resolved = config_to_dict(config)["video"]

    assert config.video is not None
    assert config.video.start_time == 1.5
    assert config.video.end_time == 4
    assert config.video.sampling_rate == 2.5
    assert config.video.output_resolution == (640, 360)
    assert resolved["temporal_smoothing"] == 0.25

    raw["video"]["unknown"] = True
    with pytest.raises(ValueError, match="Unknown key.*video"):
        parse_config(raw)


def test_video_settings_apply_documented_defaults(tmp_path):
    source = tmp_path / "clip.mp4"
    source.touch()
    raw = _minimal_config()
    raw["input"] = {"files": [str(source)]}
    raw["video"] = {}

    config = parse_config(raw)

    assert config.video is not None
    assert config.video.start_time == 0
    assert config.video.end_time is None
    assert config.video.sampling_rate == 5
    assert config.video.frame_limit is None
    assert config.video.output_resolution is None
    assert config.video.pca_fit_frames == 32
    assert config.video.temporal_smoothing == 0
    assert config.video.codec == "libx264"


def test_video_rejects_crop_and_pad_that_shift_spatial_maps(tmp_path):
    source = tmp_path / "clip.mp4"
    source.touch()
    raw = _minimal_config()
    raw["input"] = {"files": [str(source)]}
    raw["video"] = {}
    for field in ("crop", "pad"):
        raw["preprocessing"] = {"image_size": 672, field: "center"}
        with pytest.raises(
            ValueError, match="preprocessing.crop and preprocessing.pad"
        ):
            parse_config(raw)


def test_video_sampling_rate_accepts_auto_and_rejects_other_strings(tmp_path):
    source = tmp_path / "clip.mp4"
    source.touch()
    raw = _minimal_config()
    raw["input"] = {"files": [str(source)]}
    raw["video"] = {"sampling_rate": "auto"}

    config = parse_config(raw)

    assert config.video is not None
    assert config.video.sampling_rate == "auto"
    assert config_to_dict(config)["video"]["sampling_rate"] == "auto"

    raw["video"]["sampling_rate"] = "original"
    with pytest.raises(ValueError, match="video.sampling_rate"):
        parse_config(raw)


def test_video_rejects_invalid_time_range_and_odd_resolution(tmp_path):
    source = tmp_path / "clip.mp4"
    source.touch()
    raw = _minimal_config()
    raw["input"] = {"files": [str(source)]}
    raw["video"] = {"start_time": 2, "end_time": 1}
    with pytest.raises(ValueError, match="end_time must be greater"):
        parse_config(raw)

    raw["video"] = {"output_resolution": [641, 360]}
    with pytest.raises(ValueError, match="must be even"):
        parse_config(raw)


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


def test_colormap_black_settings_parse_and_round_trip():
    raw = _minimal_config()
    raw["visualization"] = {
        "cmap": "magma",
        "cmap_black": {
            "threshold": 20,
            "blend_width": 35,
            "transparent": True,
        },
    }

    config = parse_config(raw)

    assert config.visualization.cmap_black == (20, 35, True)
    assert config.visualization.render_cmap.name == "magma"
    assert config.visualization.render_cmap.black_threshold == 20
    assert config.visualization.render_cmap.black_blend_width == 35
    assert config.visualization.render_cmap.black_transparent is True
    assert config_to_dict(config)["visualization"]["cmap_black"] == {
        "threshold": 20,
        "blend_width": 35,
        "transparent": True,
    }


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"threshold": 20},
        {"threshold": -1, "blend_width": 20},
        {"threshold": 255, "blend_width": 1},
        {"threshold": 20, "blend_width": 0},
        {"threshold": 250, "blend_width": 6},
        {"threshold": 20, "blend_width": 35, "transparent": "yes"},
    ],
)
def test_colormap_black_rejects_invalid_settings(value):
    raw = _minimal_config()
    raw["visualization"] = {"cmap_black": value}

    with pytest.raises(ValueError, match="visualization.cmap_black"):
        parse_config(raw)


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


def test_patch_pca_accepts_auto_threshold_and_rejects_other_strings():
    raw_config = _minimal_config(method="patch_pca")
    raw_config["analysis"]["foreground_threshold"] = "auto"

    config = parse_config(raw_config)

    assert config.patch_pca.foreground_threshold == "auto"
    assert config_to_dict(config)["analysis"]["foreground_threshold"] == "auto"

    raw_config["analysis"]["foreground_threshold"] = "otsu"
    with pytest.raises(ValueError, match="analysis.foreground_threshold"):
        parse_config(raw_config)


def test_load_config_resolves_yaml_and_overrides_from_project_root(
    tmp_path, monkeypatch
):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n")
    config_dir = tmp_path / "nested" / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "experiment.yaml"
    raw_config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/vit_attention.example.yaml").read_text()
    )
    raw_config["input"]["files"] = ["images/cat.jpg"]
    raw_config["input"]["folders"] = []
    raw_config["output"]["directory"] = "figures"
    image_path = tmp_path / "images" / "cat.jpg"
    image_path.parent.mkdir()
    image_path.touch()
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    config = load_config(config_path)

    assert config.input.paths == (image_path,)
    assert config.output.directory == tmp_path / "figures"

    other_image = image_path.with_name("other.jpg")
    other_image.touch()
    overridden = load_config(
        config_path,
        overrides={
            "input": {"files": ["images/other.jpg"]},
            "output": {"directory": "other-figures"},
        },
    )
    assert overridden.input.paths == (other_image,)
    assert overridden.output.directory == tmp_path / "other-figures"


def test_paths_fall_back_to_working_directory_without_project(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "workflow.yaml"
    image_path = tmp_path / "photo.jpg"
    image_path.touch()
    raw_config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/vit_attention.example.yaml").read_text()
    )
    raw_config["input"]["files"] = ["photo.jpg"]
    raw_config["input"]["folders"] = []
    raw_config["output"]["directory"] = "outputs"
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = load_config(config_path)

    assert config.input.paths == (image_path,)
    assert config.output.directory == tmp_path / "outputs"


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("input", "limit"),
        ("model", "pretrained"),
        ("preprocessing", "mean"),
        ("analysis", "heads"),
        ("runtime", "seed"),
        ("visualization", "padding"),
        ("output", "raw_format"),
    ],
)
def test_yaml_requires_every_setting_even_when_overridden(tmp_path, section, key):
    raw = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/vit_attention.example.yaml").read_text()
    )
    raw[section].pop(key)
    path = tmp_path / "incomplete.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match=f"Missing setting.*{section}:.*{key}"):
        load_config(path, overrides={section: {key: None}})


def test_yaml_requires_all_video_settings(tmp_path):
    raw = yaml.safe_load(
        (
            Path(__file__).parents[1] / "configs/vit_attention.video.example.yaml"
        ).read_text()
    )
    raw["video"].pop("codec")
    path = tmp_path / "incomplete-video.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="Missing setting.*video: codec"):
        load_config(path)


def test_yaml_rejects_removed_preset_key(tmp_path):
    raw = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/vit_attention.example.yaml").read_text()
    )
    raw["preset"] = "dino-vits8-attention"
    path = tmp_path / "old-config.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="Unknown key.*preset"):
        load_config(path)


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
    raw_config["input"]["paths"] = [str(Path("examples/1.jpg").resolve())]
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
    raw_config["input"]["paths"] = [str(Path("examples/1.jpg").resolve())]
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
