import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from vision_lens.config import ColormapSpec
from vision_lens.visualization import (
    attention_map_to_array,
    image_grid,
    make_image_comparison_grid,
    overlay_attention,
    render_heatmap,
    save_grid_figure,
    save_image,
    shared_value_range,
)


def test_render_heatmap_accepts_configurable_cmap():
    heatmap = render_heatmap(torch.rand(1, 1, 8, 8), cmap="viridis")

    assert heatmap.mode == "RGB"
    assert heatmap.size == (8, 8)


def test_render_heatmap_can_hold_low_palette_values_at_black_then_blend():
    values = torch.tensor([[0.0, 19 / 255, 20 / 255, 30 / 255, 40 / 255, 1.0]])
    cmap = ColormapSpec(
        "viridis",
        black_threshold=20,
        black_blend_width=20,
        black_transparent=False,
    )

    pixels = np.asarray(
        render_heatmap(
            values, cmap=cmap, normalization="fixed", normalization_range=(0, 1)
        )
    )[0]
    original = np.asarray(
        render_heatmap(
            values, cmap="viridis", normalization="fixed", normalization_range=(0, 1)
        )
    )[0]

    assert np.all(pixels[:3] == 0)
    assert np.allclose(pixels[3], original[3] * 0.5, atol=1)
    assert np.array_equal(pixels[4:], original[4:])


def test_black_colormap_transparency_does_not_change_standalone_heatmap(tmp_path):
    values = torch.tensor([[0.0, 20 / 255, 21 / 255, 1.0]])
    cmap = ColormapSpec(
        "viridis",
        black_threshold=20,
        black_blend_width=20,
        black_transparent=True,
    )

    heatmap = render_heatmap(
        values, cmap=cmap, normalization="fixed", normalization_range=(0, 1)
    )
    pixels = np.asarray(heatmap)

    assert heatmap.mode == "RGB"
    assert np.all(pixels[0, 0] == 0)
    assert np.all(pixels[0, 1] == 0)
    assert np.any(pixels[0, 2] != 0)
    assert np.any(pixels[0, 3] != 0)

    path = tmp_path / "transparent.png"
    save_image(heatmap, path)
    with Image.open(path) as saved:
        assert saved.mode == "RGB"
        assert np.all(np.asarray(saved)[0, 0] == 0)


def test_overlay_reveals_original_beneath_transparent_black():
    values = torch.tensor([[0.0, 1.0]])
    image = Image.new("RGB", (2, 1), (200, 100, 50))
    cmap = ColormapSpec(
        "viridis",
        black_threshold=20,
        black_blend_width=20,
        black_transparent=True,
    )

    overlay = overlay_attention(
        image,
        values,
        alpha=1,
        cmap=cmap,
        normalization="fixed",
        normalization_range=(0, 1),
    )

    assert np.array_equal(np.asarray(overlay)[0, 0], [200, 100, 50])
    assert not np.array_equal(np.asarray(overlay)[0, 1], [200, 100, 50])


@pytest.mark.parametrize(
    "cmap",
    [
        "gray",
        ColormapSpec(
            "gray",
            black_threshold=20,
            black_blend_width=20,
            black_transparent=False,
        ),
        ColormapSpec(
            "gray",
            black_threshold=20,
            black_blend_width=20,
            black_transparent=True,
        ),
    ],
)
def test_sigmoid_alpha_curve_endpoints_are_scaled_by_overlay_alpha(cmap):
    values = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
    image = Image.new("RGB", (5, 1), (200, 100, 50))

    overlay = overlay_attention(
        image,
        values,
        alpha=0.1,
        alpha_curve_steepness=10,
        cmap=cmap,
        normalization="fixed",
        normalization_range=(0, 1),
    )
    pixels = np.asarray(overlay)[0]
    expected_high = np.asarray(
        Image.blend(image, Image.new("RGB", image.size, "white"), alpha=0.1)
    )[0, -1]

    assert np.array_equal(pixels[0], [200, 100, 50])
    assert np.allclose(pixels[-1], expected_high, atol=1)


def test_overlay_alpha_uniformly_scales_alpha_curve_opacity():
    values = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
    image = Image.new("RGB", (5, 1), "black")

    full = np.asarray(
        overlay_attention(
            image,
            values,
            alpha=1,
            alpha_curve_steepness=10,
            cmap="gray",
            normalization="fixed",
            normalization_range=(0, 1),
        )
    )
    scaled = np.asarray(
        overlay_attention(
            image,
            values,
            alpha=0.25,
            alpha_curve_steepness=10,
            cmap="gray",
            normalization="fixed",
            normalization_range=(0, 1),
        )
    )

    assert np.allclose(scaled, full * 0.25, atol=1)


def test_alpha_curve_steepness_controls_the_sigmoid_transition():
    values = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
    image = Image.new("RGB", (5, 1), "black")

    gentle = np.asarray(
        overlay_attention(
            image,
            values,
            alpha=1,
            alpha_curve_steepness=2,
            cmap="gray",
            normalization="fixed",
            normalization_range=(0, 1),
        )
    )[0, :, 0]
    steep = np.asarray(
        overlay_attention(
            image,
            values,
            alpha=1,
            alpha_curve_steepness=12,
            cmap="gray",
            normalization="fixed",
            normalization_range=(0, 1),
        )
    )[0, :, 0]

    assert steep[1] < gentle[1]
    assert steep[3] > gentle[3]
    assert abs(int(steep[2]) - int(gentle[2])) <= 1


def test_alpha_curve_midpoint_moves_the_opacity_transition():
    values = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
    image = Image.new("RGB", (5, 1), "black")

    centered = np.asarray(
        overlay_attention(
            image,
            values,
            alpha=1,
            alpha_curve_steepness=10,
            alpha_curve_midpoint=0.5,
            cmap="gray",
            normalization="fixed",
            normalization_range=(0, 1),
        )
    )[0, :, 0]
    shifted = np.asarray(
        overlay_attention(
            image,
            values,
            alpha=1,
            alpha_curve_steepness=10,
            alpha_curve_midpoint=0.25,
            cmap="gray",
            normalization="fixed",
            normalization_range=(0, 1),
        )
    )[0, :, 0]

    assert shifted[1] > centered[1]
    assert shifted[2] > centered[2]
    assert shifted[0] == centered[0] == 0
    assert shifted[-1] == centered[-1] == 255


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


def test_image_grid_automatically_balances_four_items():
    grid = image_grid(
        [Image.new("RGB", (10, 10), "white")] * 4,
        gap=4,
    )

    assert grid.size == (24, 24)


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


@pytest.mark.parametrize(
    ("background", "expected_rgb", "label_is_lighter"),
    [(None, (255, 255, 255), False), ("#101010", (16, 16, 16), True)],
)
def test_labeled_grid_png_has_opaque_contrasting_background(
    tmp_path, background, expected_rgb, label_is_lighter
):
    output_path = tmp_path / "grid.png"
    save_grid_figure(
        images=[Image.new("RGB", (8, 8), "purple")],
        labels=["layer 2"],
        output_path=output_path,
        tile_size=(100, 50),
        padding=10,
        background=background,
    )

    with Image.open(output_path) as grid:
        pixels = np.asarray(grid.convert("RGBA"))
    assert np.all(pixels[:, :, 3] == 255)
    assert tuple(pixels[0, 0, :3]) == expected_rgb
    label_region = pixels[10:42, 10:110, :3]
    if label_is_lighter:
        assert label_region.max() > 200
    else:
        assert label_region.min() < 80


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
