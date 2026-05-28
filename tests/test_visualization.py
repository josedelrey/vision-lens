from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from vision_lens.attention import LayerAttentionMaps
from vision_lens.visualization import (
    attention_map_to_array,
    image_grid,
    make_image_comparison_grid,
    make_layer_comparison_grid,
    overlay_attention,
    render_heatmap,
    save_image,
)


def test_render_heatmap_returns_rgb_image():
    heatmap = render_heatmap(torch.rand(1, 1, 8, 8))

    assert heatmap.mode == "RGB"
    assert heatmap.size == (8, 8)


def test_overlay_attention_matches_input_size():
    image = Image.new("RGB", (32, 24), "white")
    attention_map = torch.rand(1, 1, 24, 32)

    overlay = overlay_attention(image, attention_map)

    assert overlay.mode == "RGB"
    assert overlay.size == image.size


def test_attention_map_to_array_selects_batch_and_head():
    attention_map = torch.zeros(2, 3, 4, 4)
    attention_map[1, 2] = torch.arange(16).reshape(4, 4)

    array = attention_map_to_array(attention_map, batch_index=1, head_index=2)

    assert array.shape == (4, 4)
    assert array.min() == 0
    assert array.max() == 1


def test_make_layer_comparison_grid_can_save():
    image = Image.new("RGB", (24, 24), "white")
    layers = (
        LayerAttentionMaps(
            layer_index=0,
            maps=torch.rand(1, 1, 24, 24),
            head_indices=None,
            head_fusion="mean",
            patch_grid=(2, 2),
        ),
        LayerAttentionMaps(
            layer_index=1,
            maps=torch.rand(1, 1, 24, 24),
            head_indices=None,
            head_fusion="mean",
            patch_grid=(2, 2),
        ),
    )

    with TemporaryDirectory() as directory:
        output_path = Path(directory) / "layers.png"
        grid = make_layer_comparison_grid(
            image,
            layers,
            output_path=output_path,
            columns=2,
        )

        assert output_path.is_file()
        assert grid.size[0] > image.width
        assert grid.size[1] > image.height


def test_make_image_comparison_grid_can_save():
    images = [
        Image.new("RGB", (24, 24), "white"),
        Image.new("RGB", (24, 24), "gray"),
    ]
    attention_maps = [torch.rand(1, 1, 24, 24), torch.rand(1, 1, 24, 24)]

    with TemporaryDirectory() as directory:
        output_path = Path(directory) / "images.png"
        grid = make_image_comparison_grid(
            images,
            attention_maps,
            labels=["cat", "coffee"],
            output_path=output_path,
            columns=2,
        )

        assert output_path.is_file()
        assert grid.size[0] > images[0].width


def test_image_grid_and_save_image():
    images = [
        Image.new("RGB", (16, 16), "white"),
        Image.new("RGB", (16, 16), "black"),
    ]
    grid = image_grid(images, columns=2)

    with TemporaryDirectory() as directory:
        output_path = save_image(grid, Path(directory) / "grid.png")

        assert output_path.is_file()
