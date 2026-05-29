from vision_lens.config import parse_config


def test_parse_config_reads_visualization_cmap():
    config = parse_config(
        {
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
            "visualization": {
                "overlay_alpha": 0.35,
                "cmap": "viridis",
            },
        }
    )

    assert config.visualization.overlay_alpha == 0.35
    assert config.visualization.cmap == "viridis"


def test_parse_config_defaults_visualization_cmap_to_viridis():
    config = parse_config(
        {
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
            "visualization": {
                "overlay_alpha": 0.35,
            },
        }
    )

    assert config.visualization.cmap == "viridis"
