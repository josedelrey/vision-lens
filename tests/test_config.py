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
        "input": {"files": ["examples/1.jpg"]},
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
    assert config.visualization.overlay_alpha_curve is None
    assert config.visualization.output_size is None
    assert config.visualization.interpolation == "bilinear"
    assert config.visualization.anyup_query_chunk_size is None
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
    raw["visualization"] = {"output_size": [640, 360]}
    raw["video"] = {
        "start_time": 1.5,
        "end_time": 4,
        "sampling_rate": 2.5,
        "frame_limit": 7,
        "temporal_smoothing": 0.25,
        "codec": "libx264",
    }

    config = parse_config(raw)
    resolved = config_to_dict(config)["video"]

    assert config.video is not None
    assert config.video.start_time == 1.5
    assert config.video.end_time == 4
    assert config.video.sampling_rate == 2.5
    assert config.visualization.output_size == (640, 360)
    assert config.output.grids is False
    assert "grids" not in config_to_dict(config)["output"]
    assert resolved["temporal_smoothing"] == 0.25

    raw["video"]["unknown"] = True
    with pytest.raises(ValueError, match="Unknown key.*video"):
        parse_config(raw)


def test_video_rejects_image_grid_output(tmp_path):
    source = tmp_path / "clip.mp4"
    source.touch()
    raw = _minimal_config()
    raw["input"] = {"files": [str(source)]}
    raw["output"]["grids"] = False
    raw["video"] = {}

    with pytest.raises(ValueError, match="Unknown key.*output: grids"):
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
    assert config.video.pca_fit_frames == 32
    assert config.video.temporal_smoothing == 0
    assert config.video.codec == "libx264"


def test_video_patch_pca_has_no_image_foreground_settings(tmp_path):
    source = tmp_path / "clip.mp4"
    source.touch()
    raw = _minimal_config(method="patch_pca")
    raw["input"] = {"files": [str(source)]}
    raw["video"] = {}

    config = parse_config(raw)
    resolved = config_to_dict(config)["analysis"]

    assert config.analysis.foreground_separation is None
    assert config.analysis.foreground_threshold is None
    assert config.analysis.foreground_side is None
    assert config.analysis.rgb_fit_scope is None
    assert "foreground_separation" not in resolved
    assert "foreground_threshold" not in resolved
    assert "foreground_side" not in resolved
    assert "rgb_fit_scope" not in resolved

    raw["analysis"]["foreground_threshold"] = 0.2
    with pytest.raises(
        ValueError, match="analysis.foreground_threshold.*not applicable"
    ):
        parse_config(raw)


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


def test_video_rejects_invalid_time_range_and_odd_output_size(tmp_path):
    source = tmp_path / "clip.mp4"
    source.touch()
    raw = _minimal_config()
    raw["input"] = {"files": [str(source)]}
    raw["video"] = {"start_time": 2, "end_time": 1}
    with pytest.raises(ValueError, match="end_time must be greater"):
        parse_config(raw)

    raw["video"] = {}
    raw["visualization"] = {"output_size": [641, 360]}
    with pytest.raises(ValueError, match="must be even"):
        parse_config(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_time", float("nan")),
        ("end_time", float("inf")),
        ("sampling_rate", float("nan")),
        ("sampling_rate", float("inf")),
    ],
)
def test_video_numeric_settings_must_be_finite(tmp_path, field, value):
    source = tmp_path / "clip.mp4"
    source.touch()
    raw = _minimal_config()
    raw["input"] = {"files": [str(source)]}
    raw["video"] = {field: value}

    with pytest.raises(ValueError, match=f"video.{field}.*finite"):
        parse_config(raw)


def test_parse_config_reads_visualization_values():
    config = parse_config(
        _minimal_config(
            {
                "output_size": "match",
                "interpolation": "anyup",
                "anyup_query_chunk_size": 4096,
                "overlay_alpha": 0.35,
                "cmap": "magma",
                "grid_format": "svg",
            }
        )
    )

    assert config.visualization.overlay_alpha == 0.35
    assert config.visualization.output_size == "match"
    assert config.visualization.anyup_query_chunk_size == 4096
    assert config.visualization.cmap == "magma"
    assert config.visualization.grid_format == "svg"
    assert config_to_dict(config)["visualization"]["output_size"] == "match"
    assert config_to_dict(config)["visualization"]["anyup_query_chunk_size"] == 4096


@pytest.mark.parametrize(
    ("value", "expected"),
    [([320, 180], (320, 180)), (256, (256, 256)), (None, None)],
)
def test_visualization_output_size_accepts_fixed_square_and_null(value, expected):
    config = parse_config(_minimal_config({"output_size": value}))

    assert config.visualization.output_size == expected
    serialized = config_to_dict(config)["visualization"]["output_size"]
    assert serialized == (list(expected) if isinstance(expected, tuple) else expected)


def test_removed_match_input_size_is_rejected():
    with pytest.raises(ValueError, match="Unknown key.*match_input_size"):
        parse_config(_minimal_config({"match_input_size": True}))


def test_anyup_query_chunk_size_requires_anyup_interpolation():
    with pytest.raises(ValueError, match="requires.*interpolation"):
        parse_config(_minimal_config({"anyup_query_chunk_size": 4096}))

    with pytest.raises(ValueError, match="positive integer"):
        parse_config(
            _minimal_config({"interpolation": "anyup", "anyup_query_chunk_size": 0})
        )


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


def test_overlay_alpha_curve_parses_and_round_trips():
    raw = _minimal_config(
        {
            "overlay_alpha": 0.35,
            "overlay_alpha_curve": {"steepness": 12, "midpoint": 0.25},
        }
    )

    config = parse_config(raw)

    assert config.visualization.overlay_alpha_curve is not None
    assert config.visualization.overlay_alpha_curve.steepness == 12
    assert config.visualization.overlay_alpha_curve.midpoint == 0.25
    assert config_to_dict(config)["visualization"]["overlay_alpha_curve"] == {
        "steepness": 12,
        "midpoint": 0.25,
    }


def test_overlay_alpha_curve_midpoint_defaults_to_half():
    config = parse_config(_minimal_config({"overlay_alpha_curve": {"steepness": 12}}))

    assert config.visualization.overlay_alpha_curve is not None
    assert config.visualization.overlay_alpha_curve.midpoint == 0.5


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"steepness": 0},
        {"steepness": -1},
        {"steepness": True},
        {"steepness": float("inf")},
        {"steepness": float("nan")},
        {"steepness": 10, "midpoint": -0.1},
        {"steepness": 10, "midpoint": 1.1},
        {"steepness": 10, "midpoint": True},
        {"steepness": 10, "unknown": 0.5},
    ],
)
def test_overlay_alpha_curve_rejects_invalid_settings(value):
    raw = _minimal_config({"overlay_alpha_curve": value})

    with pytest.raises(ValueError, match="visualization.overlay_alpha_curve"):
        parse_config(raw)


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

    assert config.analysis.foreground_separation is True
    assert config.analysis.foreground_threshold == 0.5
    assert config.analysis.foreground_side == "high"
    assert config.analysis.rgb_fit_scope == "foreground"
    assert config.output.overlays is False

    raw_config["analysis"].update(
        {
            "foreground_separation": False,
            "foreground_threshold": 0.65,
            "foreground_side": "low",
            "rgb_fit_scope": "all",
        }
    )
    with pytest.raises(ValueError, match="foreground_threshold.*not applicable"):
        parse_config(raw_config)

    for key in ("foreground_threshold", "foreground_side", "rgb_fit_scope"):
        raw_config["analysis"].pop(key)
    overridden = parse_config(raw_config)
    assert overridden.analysis.foreground_separation is False
    assert overridden.analysis.foreground_threshold is None
    assert overridden.analysis.foreground_side is None
    assert overridden.analysis.rgb_fit_scope is None
    assert "rgb_fit_scope" not in config_to_dict(overridden)["analysis"]
    assert config_to_dict(overridden)["analysis"]["foreground_separation"] is False


def test_patch_pca_rejects_unknown_rgb_fit_scope():
    raw_config = _minimal_config(method="patch_pca")
    raw_config["analysis"]["rgb_fit_scope"] = "background"

    with pytest.raises(ValueError, match="analysis.rgb_fit_scope"):
        parse_config(raw_config)


def test_patch_pca_accepts_auto_threshold_and_rejects_other_strings():
    raw_config = _minimal_config(method="patch_pca")
    raw_config["analysis"]["foreground_threshold"] = "auto"

    config = parse_config(raw_config)

    assert config.analysis.foreground_threshold == "auto"
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
        (Path(__file__).parents[1] / "configs/vit_attention.yaml").read_text()
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
        (Path(__file__).parents[1] / "configs/vit_attention.yaml").read_text()
    )
    raw_config["input"]["files"] = ["photo.jpg"]
    raw_config["input"]["folders"] = []
    raw_config["output"]["directory"] = "outputs"
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = load_config(config_path)

    assert config.input.paths == (image_path,)
    assert config.output.directory == tmp_path / "outputs"


def test_yaml_uses_parser_defaults_for_omitted_settings(tmp_path):
    raw = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/vit_attention.yaml").read_text()
    )
    path = tmp_path / "minimal.yaml"
    path.write_text(yaml.safe_dump(raw))

    config = load_config(path)

    assert config.model.pretrained is True
    assert config.runtime.batch_size == 8
    assert config.visualization.grid_format == "png"
    assert config.output.raw_arrays is False


def test_yaml_rejects_removed_preset_key(tmp_path):
    raw = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/vit_attention.yaml").read_text()
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

    with pytest.raises(ValueError, match="managed loader argument.*img_size"):
        parse_config(raw_config)


@pytest.mark.parametrize(
    ("backend", "architecture", "option"),
    [("timm", "vit", "pretrained"), ("torchvision", "cnn", "weights")],
)
def test_model_loader_managed_options_are_rejected(backend, architecture, option):
    method = "attention" if backend == "timm" else "gradcam"
    raw_config = _minimal_config(
        method=method,
        architecture=architecture,
        backend=backend,
    )
    raw_config["model"]["options"] = {option: None}

    with pytest.raises(ValueError, match=f"managed loader argument.*{option}"):
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
    raw_config["input"]["files"] = ["missing.jpg"]

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
    assert config_to_dict(config)["input"] == {"files": [str(image_dir / "a.jpg")]}


@pytest.mark.parametrize("pattern", ["/tmp/*.jpg", "../*.jpg", r"C:\\*.jpg"])
def test_input_glob_patterns_must_be_relative_and_contained(tmp_path, pattern):
    raw_config = _minimal_config()
    raw_config["input"] = {"folders": [str(tmp_path)], "patterns": [pattern]}

    with pytest.raises(ValueError, match="relative glob patterns"):
        parse_config(raw_config)


def test_all_new_controls_are_parsed_and_resolved(tmp_path):
    raw_config = _minimal_config(
        method="gradcam",
        architecture="cnn",
        backend="torchvision",
    )
    raw_config["analysis"].update({"target_layer": "layer3", "target_class": 7})
    raw_config["input"]["files"] = [str(Path("examples/1.jpg").resolve())]
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
        "interpolation": "bilinear_mask",
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
            "raw_arrays": True,
            "grids": True,
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
    assert config.visualization.interpolation == "bilinear_mask"
    assert config_to_dict(config)["visualization"]["interpolation"] == "bilinear_mask"
    assert config.visualization.normalization_range == (-1.0, 2.0)
    assert config.output.image_format == "webp"
    assert config.output.raw_format == "npz"


def test_fixed_normalization_requires_a_range():
    raw_config = _minimal_config({"normalization": "fixed"})

    with pytest.raises(ValueError, match="normalization_range is required"):
        parse_config(raw_config)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("preprocessing", "mean", [float("nan"), 0, 0]),
        ("preprocessing", "std", [1, float("inf"), 1]),
        ("visualization", "normalization_range", [float("nan"), 1]),
        ("visualization", "normalization_range", [0, float("inf")]),
    ],
)
def test_numeric_sequences_must_contain_only_finite_values(section, field, value):
    raw = _minimal_config()
    raw[section][field] = value
    if field == "normalization_range":
        raw["visualization"]["normalization"] = "fixed"

    with pytest.raises(ValueError, match=f"{section}.{field}.*finite"):
        parse_config(raw)


@pytest.mark.parametrize("interpolation", ["mask", "bicubic"])
def test_visualization_interpolation_rejects_unknown_modes(interpolation):
    with pytest.raises(ValueError, match="visualization.interpolation"):
        parse_config(_minimal_config({"interpolation": interpolation}))


@pytest.mark.parametrize(
    "interpolation",
    ["anyup", "anyup_mask", "anyup_soft", "anyup_soft_mask"],
)
def test_visualization_interpolation_accepts_anyup_modes(interpolation):
    values = {"interpolation": interpolation}
    if interpolation in {"anyup_soft", "anyup_soft_mask"}:
        values["anyup_query_chunk_size"] = 4096
    config = parse_config(_minimal_config(values))

    assert config.visualization.interpolation == interpolation


def test_anyup_soft_requires_query_chunking():
    with pytest.raises(ValueError, match="soft AnyUp.*anyup_query_chunk_size"):
        parse_config(_minimal_config({"interpolation": "anyup_soft"}))


def test_pca_projection_paths_resolve_from_config_and_load_must_exist(tmp_path):
    raw_config = _minimal_config(method="patch_pca")
    raw_config["input"]["files"] = [str(Path("examples/1.jpg").resolve())]
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


@pytest.mark.parametrize("collision", ["input", "manifest"])
def test_saved_projection_cannot_collide_with_run_artifacts(tmp_path, collision):
    input_path = Path("examples/1.jpg").resolve()
    output_directory = tmp_path / "outputs"
    raw = _minimal_config(method="patch_pca")
    raw["input"]["files"] = [str(input_path)]
    raw["output"]["directory"] = str(output_directory)
    raw["analysis"]["save_projection"] = (
        str(input_path)
        if collision == "input"
        else str(output_directory / "run-manifest.json")
    )

    with pytest.raises(ValueError, match="must not (overwrite|use)"):
        parse_config(raw)


def test_patch_pca_projection_modes_reject_irrelevant_settings(tmp_path):
    projection_path = tmp_path / "projection.npz"
    projection_path.touch()
    raw = _minimal_config(method="patch_pca")
    raw["input"]["files"] = [str(Path("examples/1.jpg").resolve())]
    raw["analysis"]["projection_path"] = str(projection_path)

    with pytest.raises(ValueError, match="analysis.projection_path.*not applicable"):
        parse_config(raw, base_dir=tmp_path)

    raw["analysis"] = {
        "method": "patch_pca",
        "projection": "load",
        "projection_path": str(projection_path),
        "foreground_threshold": 0.2,
    }
    with pytest.raises(
        ValueError, match="analysis.foreground_threshold.*not applicable"
    ):
        parse_config(raw, base_dir=tmp_path)

    raw["analysis"].pop("foreground_threshold")
    raw["analysis"]["save_projection"] = "copy.npz"
    with pytest.raises(ValueError, match="analysis.save_projection.*not applicable"):
        parse_config(raw, base_dir=tmp_path)


def test_patch_pca_can_fit_only_a_saved_projection(tmp_path):
    raw = _minimal_config(method="patch_pca")
    raw["analysis"]["save_projection"] = str(tmp_path / "projection.npz")
    raw["output"].update({"heatmaps": False, "grids": False, "raw_arrays": False})

    config = parse_config(raw)

    assert config.analysis.save_projection == tmp_path / "projection.npz"


def test_single_image_grids_only_patch_pca_warns():
    raw = _minimal_config(method="patch_pca")
    raw["output"].update({"heatmaps": False, "grids": True, "raw_arrays": False})

    with pytest.warns(UserWarning, match="one-tile comparison grid"):
        parse_config(raw)


def test_choice_settings_reject_wrong_shaped_values_cleanly():
    raw = _minimal_config()
    raw["preprocessing"]["resize"] = []

    with pytest.raises(ValueError, match="preprocessing.resize must be one of"):
        parse_config(raw)


def test_attention_heads_must_be_non_empty_when_configured():
    raw = _minimal_config()
    raw["analysis"]["heads"] = []

    with pytest.raises(ValueError, match="analysis.heads must be a non-empty list"):
        parse_config(raw)


def test_model_option_keys_must_be_strings():
    raw = _minimal_config()
    raw["model"]["options"] = {1: "value"}

    with pytest.raises(ValueError, match="model.options keys must be strings"):
        parse_config(raw)


@pytest.mark.parametrize(
    ("location", "message"),
    [("top level", "top level"), ("section", "runtime")],
)
def test_configuration_keys_must_be_strings(location, message):
    raw = _minimal_config()
    if location == "top level":
        raw[1] = {}
    else:
        raw["runtime"][1] = "value"

    with pytest.raises(ValueError, match=f"{message} keys must be strings"):
        parse_config(raw)


@pytest.mark.parametrize(
    ("device", "precision"),
    [("cpu", "float16"), ("mps", "bfloat16")],
)
def test_static_device_precision_incompatibilities_are_rejected(device, precision):
    raw = _minimal_config()
    raw["runtime"].update({"device": device, "precision": precision})

    with pytest.raises(ValueError, match=f"{precision}.*{device.upper()}"):
        parse_config(raw)


@pytest.mark.parametrize("nested", [False, True])
def test_output_directory_cannot_be_a_file_or_descend_from_one(tmp_path, nested):
    file_path = tmp_path / "not-a-directory"
    file_path.touch()
    raw = _minimal_config()
    raw["output"]["directory"] = str(file_path / "child" if nested else file_path)

    with pytest.raises(ValueError, match="output.directory.*not a (file|directory)"):
        parse_config(raw)


def test_runtime_seed_must_fit_numpy_seed_range():
    raw = _minimal_config()
    raw["runtime"]["seed"] = 2**32

    with pytest.raises(ValueError, match=r"runtime.seed must be at most 2\*\*32 - 1"):
        parse_config(raw)


def test_overrides_cannot_switch_conditional_modes():
    raw = _minimal_config(method="patch_pca")

    with pytest.raises(ValueError, match="cannot switch conditional mode.*projection"):
        parse_config(raw, overrides={"analysis": {"projection": "load"}})


def test_video_rejects_managed_dynamic_image_size_option(tmp_path):
    source = tmp_path / "clip.mp4"
    source.touch()
    raw = _minimal_config()
    raw["input"] = {"files": [str(source)]}
    raw["model"]["options"] = {"dynamic_img_size": False}
    raw["video"] = {}

    with pytest.raises(ValueError, match="dynamic_img_size"):
        parse_config(raw)


@pytest.mark.parametrize(
    ("key", "value"),
    (("cmap", "not-a-colormap"), ("background", "not-a-color")),
)
def test_static_matplotlib_values_are_validated(key, value):
    raw = _minimal_config()
    raw["visualization"][key] = value

    with pytest.raises(ValueError, match=f"visualization.{key}"):
        parse_config(raw)


def test_raw_only_maps_reject_rendering_normalization_settings():
    raw = _minimal_config()
    raw["output"].update(
        {
            "heatmaps": False,
            "overlays": False,
            "grids": False,
            "raw_arrays": True,
        }
    )
    raw["visualization"]["normalization"] = "shared"

    with pytest.raises(ValueError, match="visualization.normalization.*not applicable"):
        parse_config(raw)

    raw["visualization"].clear()
    config = parse_config(raw)
    assert "normalization" not in config_to_dict(config)["visualization"]


def test_loaded_patch_pca_projection_uses_only_saved_projection_settings(tmp_path):
    projection_path = tmp_path / "projection.npz"
    projection_path.touch()
    raw = _minimal_config(method="patch_pca")
    raw["analysis"] = {
        "method": "patch_pca",
        "projection": "load",
        "projection_path": str(projection_path),
    }

    config = parse_config(raw)

    assert config.analysis.foreground_separation is None
    assert config_to_dict(config)["analysis"] == {
        "method": "patch_pca",
        "projection": "load",
        "projection_path": str(projection_path),
    }


def test_workflow_specific_no_op_settings_are_rejected(tmp_path):
    source = tmp_path / "clip.mp4"
    source.touch()
    raw = _minimal_config()
    raw["input"] = {"files": [str(source)]}
    raw["video"] = {}

    for section, key, value in (
        ("preprocessing", "resize", "stretch"),
        ("runtime", "workers", 0),
        ("visualization", "grid_format", "png"),
        ("video", "pca_fit_frames", 32),
    ):
        candidate = yaml.safe_load(yaml.safe_dump(raw))
        candidate[section][key] = value
        with pytest.raises(ValueError, match=key):
            parse_config(candidate)

    image = _minimal_config()
    image["preprocessing"].update({"normalize": False, "mean": [0, 0, 0]})
    with pytest.raises(ValueError, match="preprocessing.mean"):
        parse_config(image)

    pca = _minimal_config(method="patch_pca")
    pca["visualization"]["cmap"] = "magma"
    with pytest.raises(ValueError, match="visualization.cmap"):
        parse_config(pca)

    pca = _minimal_config(method="patch_pca")
    pca["output"]["overlays"] = False
    with pytest.raises(ValueError, match="output.overlays"):
        parse_config(pca)

    video_rollout = _minimal_config(method="rollout")
    video_rollout["input"] = {"files": [str(source)]}
    video_rollout["analysis"]["heads"] = [0]
    video_rollout["video"] = {}
    with pytest.raises(ValueError, match="analysis.heads.*not applicable"):
        parse_config(video_rollout)

    gridless_rollout = _minimal_config(method="rollout")
    gridless_rollout["analysis"]["head_fusion"] = "max"
    gridless_rollout["output"]["grids"] = False
    with pytest.raises(ValueError, match="analysis.head_fusion.*not applicable"):
        parse_config(gridless_rollout)


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
