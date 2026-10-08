from dataclasses import replace

import numpy as np
import pytest
import torch

from vision_lens.analysis.patch_pca import (
    PatchPCAProjection,
    fit_patch_pca_projection_batches,
    load_patch_pca_projection,
    project_patch_embeddings,
    save_patch_pca_projection,
)
from vision_lens.analysis.pca_colors import blend_component_scores, component_palette


@pytest.mark.parametrize("interpolation", ["nearest", "bilinear", "bilinear_mask"])
@pytest.mark.parametrize("foreground", [True, False])
def test_explicit_three_dimensions_preserves_default_pixels(interpolation, foreground):
    embeddings = torch.randn(2, 12, 8, generator=torch.Generator().manual_seed(91))
    settings = {
        "patch_grid": (3, 4),
        "image_size": (9, 12),
        "interpolation": interpolation,
        "foreground_separation": foreground,
    }
    original = project_patch_embeddings(embeddings, **settings)
    explicit = project_patch_embeddings(embeddings, rgb_dimensions=3, **settings)

    assert torch.equal(original.rgb_patches, explicit.rgb_patches)
    assert torch.equal(original.foreground_mask, explicit.foreground_mask)
    assert explicit.component_scores is None
    for first, second in zip(original.images, explicit.images, strict=True):
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_component_palette_extends_distinct_colors_and_blends_scores():
    palette = component_palette(12)
    np.testing.assert_array_equal(palette[:3], np.eye(3))
    np.testing.assert_array_equal(palette[:6], component_palette(6))
    assert np.unique(palette, axis=0).shape[0] == 12
    colors = torch.tensor(palette)
    pure_components = blend_component_scores(torch.eye(12), colors)
    torch.testing.assert_close(pure_components, colors)
    blend = blend_component_scores(
        torch.tensor([[0.0, 0.0, 0.0, 0.25, 0.5]]), colors[:5]
    )
    torch.testing.assert_close(blend[0], 0.25 * colors[3] + 0.5 * colors[4])
    bright = blend_component_scores(torch.ones(1, 12), colors)
    assert bright.max() == 1
    assert bright.min() > 0


@pytest.mark.parametrize("foreground", [True, False])
def test_higher_dimensional_fit_matches_batched_fit_and_replays(tmp_path, foreground):
    embeddings = torch.randn(2, 16, 8, generator=torch.Generator().manual_seed(4))
    settings = {"foreground_separation": foreground, "rgb_dimensions": 6}
    result = project_patch_embeddings(embeddings, (4, 4), (8, 8), **settings)
    projection = fit_patch_pca_projection_batches(
        lambda: (embeddings[:1], embeddings[1:]), **settings
    )
    assert result.projection.rgb_components.shape == (8, 6)
    torch.testing.assert_close(
        projection.rgb_components, result.projection.rgb_components
    )
    assert result.component_scores.shape == (2, 16, 6)
    assert result.component_scores.min() >= 0
    assert result.component_scores.max() <= 1
    assert result.rgb_patches.shape == (2, 16, 3)
    path = save_patch_pca_projection(projection, tmp_path / "pca.npz")
    loaded = load_patch_pca_projection(path)
    torch.testing.assert_close(loaded.component_colors, projection.component_colors)
    replay = project_patch_embeddings(embeddings, (4, 4), (8, 8), projection=loaded)
    assert torch.equal(result.foreground_mask, replay.foreground_mask)
    torch.testing.assert_close(result.rgb_patches, replay.rgb_patches)
    torch.testing.assert_close(result.component_scores, replay.component_scores)
    for first, second in zip(result.images, replay.images, strict=True):
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    with pytest.raises(ValueError, match="does not match the saved projection"):
        project_patch_embeddings(
            embeddings, (4, 4), (8, 8), projection=loaded, rgb_dimensions=5
        )


def test_legacy_projection_without_palette_keeps_rgb_mapping(tmp_path):
    projection = replace(_six_feature_projection(), rgb_components=torch.eye(6)[:, :3])
    projection = replace(
        projection, rgb_minimum=torch.zeros(3), rgb_maximum=torch.ones(3)
    )
    path = save_patch_pca_projection(projection, tmp_path / "legacy.npz")
    with np.load(path) as values:
        legacy = {
            name: values[name] for name in values.files if name != "component_colors"
        }
    np.savez(path, **legacy)
    loaded = load_patch_pca_projection(path)
    embeddings = torch.tensor([[[0.1, 0.3, 0.9, 0.0, 0.0, 1.0]]])
    result = project_patch_embeddings(embeddings, (1, 1), (1, 1), projection=loaded)
    assert torch.equal(result.rgb_patches, embeddings[:, :, :3])


@pytest.mark.parametrize("dimensions", [2, 0, -1, 3.5, True, "6"])
def test_pca_rejects_invalid_dimension_counts(dimensions):
    embeddings = torch.ones(1, 4, 8)
    with pytest.raises(ValueError, match="rgb_dimensions"):
        project_patch_embeddings(embeddings, (2, 2), (2, 2), rgb_dimensions=dimensions)
    with pytest.raises(ValueError, match="rgb_dimensions"):
        fit_patch_pca_projection_batches(
            lambda: (embeddings,), rgb_dimensions=dimensions
        )


def test_pca_checks_feature_count_and_pads_small_fitting_sets():
    with pytest.raises(ValueError, match="feature count"):
        project_patch_embeddings(torch.ones(1, 4, 4), (2, 2), (2, 2), rgb_dimensions=5)
    for embeddings in (torch.ones(1, 1, 8), torch.zeros(1, 4, 8)):
        result = project_patch_embeddings(
            embeddings, (1, embeddings.shape[1]), (1, 4), rgb_dimensions=6
        )
        assert result.component_scores.shape[-1] == 6
        assert torch.isfinite(result.rgb_patches).all()
        assert not result.rgb_patches.any()


@pytest.mark.parametrize(
    ("interpolation", "chunk_size"),
    [
        ("anyup", None),
        ("anyup_mask", None),
        ("anyup", 2),
        ("anyup_mask", 2),
        ("anyup_soft", 2),
        ("anyup_soft_mask", 2),
    ],
)
def test_anyup_preserves_extra_colors_and_separate_foreground_axis(
    monkeypatch, interpolation, chunk_size
):
    from vision_lens.analysis import patch_pca

    projection = _six_feature_projection()
    embeddings = torch.tensor(
        [[[0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 1], [1, 0, 0, 0, 0, 0]]],
        dtype=torch.float32,
    )
    calls = []

    def upsample(_guidance, features, output_size, **kwargs):
        values = kwargs.get("values", features)
        calls.append(values.shape[1])
        return torch.nn.functional.interpolate(values, size=output_size, mode="nearest")

    monkeypatch.setattr(patch_pca, "upsample_features", upsample)
    monkeypatch.setattr(patch_pca, "upsample_values_streaming", upsample)
    result = project_patch_embeddings(
        embeddings,
        (1, 3),
        (2, 6),
        projection=projection,
        interpolation=interpolation,
        guidance_image=torch.zeros(1, 3, 2, 6),
        anyup_query_chunk_size=chunk_size,
    )
    palette = component_palette(5)
    expected = np.zeros((2, 6, 3), dtype=np.uint8)
    expected[:, :2] = np.round(palette[3] * 255).astype(np.uint8)
    expected[:, 2:4] = np.round(palette[4] * 255).astype(np.uint8)
    np.testing.assert_array_equal(np.asarray(result.images[0]), expected)
    assert result.foreground_mask[0, :, :4].all()
    assert not result.foreground_mask[0, :, 4:].any()
    assert calls == [
        6 if chunk_size is None or interpolation in {"anyup", "anyup_soft"} else 5
    ]


@pytest.mark.parametrize(
    "colors", [torch.zeros(4, 3), torch.full((5, 3), float("nan"))]
)
def test_projection_rejects_malformed_component_palettes(colors):
    projection = replace(_six_feature_projection(), component_colors=colors)
    with pytest.raises(ValueError, match="component colors"):
        project_patch_embeddings(
            torch.ones(1, 4, 6), (2, 2), (2, 2), projection=projection
        )


def _six_feature_projection():
    return PatchPCAProjection(
        foreground_components=torch.eye(6)[:, 5:],
        foreground_minimum=torch.zeros(1),
        foreground_maximum=torch.ones(1),
        rgb_components=torch.eye(6)[:, :5],
        rgb_minimum=torch.zeros(5),
        rgb_maximum=torch.ones(5),
        foreground_threshold=0.5,
        foreground_side="high",
        rgb_fit_scope="foreground",
        foreground_separation=True,
    )
