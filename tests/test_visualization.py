import torch
from PIL import Image

from vision_lens.visualization import (
    image_grid,
    make_image_comparison_grid,
    overlay_attention,
    render_heatmap,
    save_grid_figure,
)


def test_render_heatmap_accepts_configurable_cmap():
    heatmap = render_heatmap(torch.rand(1, 1, 8, 8), cmap="viridis")

    assert heatmap.mode == "RGB"
    assert heatmap.size == (8, 8)


def test_overlay_attention_matches_input_image_dimensions():
    image = Image.new("RGB", (12, 8), "white")
    overlay = overlay_attention(image, torch.rand(1, 1, 4, 4), cmap="viridis")

    assert overlay.mode == "RGB"
    assert overlay.size == (12, 8)


def test_image_grid_uses_largest_tile_dimensions():
    grid = image_grid(
        [
            Image.new("RGB", (8, 10), "white"),
            Image.new("RGB", (12, 6), "black"),
        ],
        columns=2,
        gap=4,
    )

    assert grid.size == (28, 10)


def test_image_comparison_grid_keeps_expected_output_dimensions(tmp_path):
    output_path = tmp_path / "comparison.svg"
    grid = make_image_comparison_grid(
        images=[
            Image.new("RGB", (8, 8), "white"),
            Image.new("RGB", (8, 8), "black"),
        ],
        attention_maps=[
            torch.rand(1, 1, 4, 4),
            torch.rand(1, 1, 4, 4),
        ],
        labels=["a", "b"],
        output_path=output_path,
        columns=2,
        cmap="viridis",
    )

    assert grid.mode == "RGB"
    assert grid.size == (28, 36)
    assert output_path.is_file()


def test_save_grid_figure_supports_pdf_output(tmp_path):
    output_path = tmp_path / "grid.pdf"

    saved_path = save_grid_figure(
        images=[Image.new("RGB", (8, 8), "white")],
        labels=["example"],
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.is_file()
