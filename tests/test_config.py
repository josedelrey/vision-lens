import yaml

from vision_lens.config import load_config, parse_config


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
            "paths": ["data/samples/cat.jpg"],
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
    assert config.attention is not None
    assert config.attention.rollout_discard_ratio == 0.9


def test_parse_config_reads_rollout_discard_ratio():
    raw_config = _minimal_config({"overlay_alpha": 0.35})
    raw_config["attention"]["rollout_discard_ratio"] = 0.75

    config = parse_config(raw_config)

    assert config.attention is not None
    assert config.attention.rollout_discard_ratio == 0.75


def test_parse_config_defaults_visualization_cmap_to_viridis():
    config = parse_config(_minimal_config({"overlay_alpha": 0.35}))

    assert config.visualization.cmap == "viridis"
    assert config.visualization.grid_format == "png"


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
