from pathlib import Path

import pytest
import yaml

from vision_lens.config import load_config, load_preset, parse_config
from vision_lens.presets import available_presets, get_preset


def _minimal_config(visualization):
    return {
        "task": "vit_attention",
        "model": {
            "architecture": "vit",
            "backend": "timm",
            "name": "vit_base_patch16_224",
            "pretrained": True,
            "options": {},
        },
        "images": {
            "paths": ["data/examples/1.jpg"],
        },
        "output": {
            "directory": "outputs",
        },
        "attention": {
            "layers": [0],
            "heads": None,
            "head_fusion": "mean",
        },
        "runtime": {
            "device": "cpu",
            "image_size": 224,
        },
        "visualization": visualization,
    }


def test_parse_config_reads_visualization_cmap():
    config = parse_config(
        _minimal_config(
            {
                "overlay_alpha": 0.35,
                "cmap": "viridis",
                "grid_format": "svg",
            }
        )
    )

    assert config.visualization.overlay_alpha == 0.35
    assert config.visualization.cmap == "viridis"
    assert config.visualization.grid_format == "svg"


def test_parse_config_defaults_visualization_cmap_to_viridis():
    config = parse_config(_minimal_config({"overlay_alpha": 0.35}))

    assert config.visualization.cmap == "viridis"
    assert config.visualization.grid_format == "png"


def test_parse_config_defaults_patch_pca_threshold():
    raw_config = _minimal_config({})
    raw_config["task"] = "patch_pca"
    raw_config.pop("attention")

    config = parse_config(raw_config)

    assert config.attention is None
    assert config.patch_pca is not None
    assert config.patch_pca.foreground_threshold == 0.5
    assert config.patch_pca.foreground_side == "high"


def test_parse_config_reads_patch_pca_threshold():
    raw_config = _minimal_config({})
    raw_config["task"] = "patch_pca"
    raw_config.pop("attention")
    raw_config["patch_pca"] = {
        "foreground_threshold": 0.65,
        "foreground_side": "low",
    }

    config = parse_config(raw_config)

    assert config.patch_pca is not None
    assert config.patch_pca.foreground_threshold == 0.65
    assert config.patch_pca.foreground_side == "low"


def test_load_config_resolves_paths_from_config_file(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    raw_config = _minimal_config(
        {
            "overlay_alpha": 0.35,
            "cmap": "viridis",
        }
    )
    raw_config["images"]["paths"] = ["images/cat.jpg"]
    raw_config["output"]["directory"] = "figures"
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    config = load_config(config_path)

    assert config.images.paths == (tmp_path / "images" / "cat.jpg",)
    assert config.output.directory == tmp_path / "figures"


@pytest.mark.parametrize(
    ("preset_name", "config_name"),
    [
        ("dino-vits8-attention", "vit_attention.example.yaml"),
        (
            "dinov2-reg4-attention",
            "vit_attention.dinov2_reg4.example.yaml",
        ),
        ("resnet50-gradcam", "gradcam.example.yaml"),
        ("dinov2-pca", "patch_pca.dinov2.example.yaml"),
    ],
)
def test_presets_resolve_to_the_existing_example_workflows(preset_name, config_name):
    repo_root = Path(__file__).parents[1]

    preset_config = load_preset(preset_name, base_dir=repo_root)
    example_config = load_config(repo_root / "configs" / config_name)

    assert preset_config == example_config


def test_rollout_preset_preserves_current_model_and_rendering_settings():
    config = load_preset("dinov2-reg4-rollout")

    assert config.task == "vit_rollout"
    assert config.model.name == ("hf_hub:timm/vit_small_patch14_reg4_dinov2.lvd142m")
    assert config.model.options == {"img_size": 672}
    assert config.runtime.image_size == 672
    assert config.attention is not None
    assert config.attention.layers == (2, 5, 8, 11)
    assert config.attention.heads is None
    assert config.attention.head_fusion == "mean"
    assert config.visualization.overlay_alpha == 0.8
    assert config.visualization.cmap == "viridis"
    assert config.visualization.grid_format == "pdf"


def test_user_config_and_cli_values_override_only_selected_preset_settings():
    config = parse_config(
        {
            "preset": "dino-vits8-attention",
            "attention": {"heads": [1, 3], "head_fusion": "none"},
        },
        overrides={
            "runtime": {"device": "cpu"},
            "visualization": {"overlay_alpha": 0.25},
        },
    )

    assert config.preset == "dino-vits8-attention"
    assert config.attention is not None
    assert config.attention.layers == (2, 5, 8, 11)
    assert config.attention.heads == (1, 3)
    assert config.attention.head_fusion == "none"
    assert config.runtime.device == "cpu"
    assert config.runtime.image_size == 672
    assert config.visualization.overlay_alpha == 0.25
    assert config.visualization.cmap == "viridis"


def test_cli_selected_preset_takes_precedence_over_config_selected_preset():
    config = parse_config(
        {"preset": "resnet50-gradcam"},
        preset="dino-vits8-attention",
    )

    assert config.preset == "dino-vits8-attention"
    assert config.task == "vit_attention"


def test_unknown_preset_lists_available_choices():
    with pytest.raises(ValueError, match="Unknown preset 'missing'") as error:
        get_preset("missing")

    assert all(name in str(error.value) for name in available_presets())


def test_loading_without_a_preset_preserves_legacy_config_behavior():
    config = parse_config(_minimal_config({"overlay_alpha": 0.35}))

    assert config.preset is None
