import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

from vision_lens.visualization import (
    attention_map_to_array,
    image_grid,
    make_image_comparison_grid,
    overlay_attention,
    render_heatmap,
    save_grid_figure,
    shared_value_range,
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


def test_image_grid_caps_columns_to_the_number_of_items():
    grid = image_grid(
        [Image.new("RGB", (10, 10), "white")] * 2,
        columns=5,
        gap=4,
    )

    assert grid.size == (24, 10)


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


def test_save_grid_figure_uses_fixed_tile_size(tmp_path):
    output_path = tmp_path / "grid.svg"

    save_grid_figure(
        images=[
            Image.new("RGB", (32, 32), "white"),
            Image.new("RGB", (518, 518), "black"),
        ],
        labels=["small", "large"],
        output_path=output_path,
        columns=2,
    )

    svg = output_path.read_text(encoding="utf-8")
    assert 'width="331.2pt"' in svg
    assert 'height="184.32pt"' in svg


def test_save_grid_figure_preserves_exact_raster_dimensions(tmp_path):
    output_path = tmp_path / "grid.png"

    save_grid_figure(
        images=[Image.new("RGB", (4, 4), "white")] * 2,
        labels=["first", "second"],
        output_path=output_path,
        columns=2,
    )

    with Image.open(output_path) as grid:
        assert grid.size == (460, 256)


def test_fixed_normalization_uses_the_configured_range():
    array = attention_map_to_array(
        torch.tensor([[0.0, 5.0, 10.0]]),
        normalization="fixed",
        normalization_range=(0.0, 20.0),
    )

    assert array.tolist() == [[0.0, 0.25, 0.5]]


def test_shared_normalization_uses_one_range_for_multiple_maps():
    value_range = shared_value_range(
        [torch.tensor([[0.0, 1.0]]), torch.tensor([[10.0, 20.0]])]
    )
    first = attention_map_to_array(
        torch.tensor([[0.0, 1.0]]),
        normalization="shared",
        normalization_range=value_range,
    )

    assert value_range == (0.0, 20.0)
    assert np.allclose(first, [[0.0, 0.05]])


def test_torch_then_matplotlib_import_needs_no_global_openmp_workaround():
    environment = os.environ.copy()
    environment.pop("KMP_DUPLICATE_LIB_OK", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, torch; "
                "import vision_lens.visualization; "
                "assert 'KMP_DUPLICATE_LIB_OK' not in os.environ"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


def test_grid_style_controls_tile_size_spacing_padding_and_labels(tmp_path):
    output_path = tmp_path / "custom.svg"
    save_grid_figure(
        images=[Image.new("RGB", (8, 8), "white")] * 2,
        labels=["first", "second"],
        output_path=output_path,
        columns=1,
        tile_size=(100, 50),
        spacing=5,
        padding=10,
        show_labels=False,
        background="#101010",
        dpi=200,
    )

    svg = output_path.read_text(encoding="utf-8")
    assert 'width="43.2pt"' in svg
    assert 'height="45pt"' in svg
    assert "first" not in svg
