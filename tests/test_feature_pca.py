import numpy as np
import pytest
import torch

from vision_lens.feature_pca import (
    _patch_tokens_from_features,
    fit_patch_pca_projection_batches,
    load_patch_pca_projection,
    project_patch_embeddings,
    save_patch_pca_projection,
)


def test_streaming_pca_projection_is_independent_of_embedding_batch_size():
    generator = torch.Generator().manual_seed(12)
    embeddings = torch.rand(7, 4, 6, generator=generator)

    def batches(size):
        return lambda: (
            embeddings[start : start + size]
            for start in range(0, len(embeddings), size)
        )

    fitted_two = fit_patch_pca_projection_batches(
        batches(2),
        foreground_threshold=0.4,
        foreground_side="low",
    )
    fitted_three = fit_patch_pca_projection_batches(
        batches(3),
        foreground_threshold=0.4,
        foreground_side="low",
    )

    result_two = project_patch_embeddings(
        embeddings,
        patch_grid=(2, 2),
        image_size=(8, 8),
        projection=fitted_two,
    )
    result_three = project_patch_embeddings(
        embeddings,
        patch_grid=(2, 2),
        image_size=(8, 8),
        projection=fitted_three,
    )

    assert torch.equal(result_two.foreground_mask, result_three.foreground_mask)
    assert all(
        np.array_equal(np.asarray(first), np.asarray(second))
        for first, second in zip(result_two.images, result_three.images, strict=True)
    )


def test_patch_pca_returns_rgb_images_and_a_shared_mask():
    embeddings = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.0, 0.1], [3.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [2.0, 0.1, 0.0], [4.0, 1.0, 1.0]],
        ]
    )

    result = project_patch_embeddings(
        embeddings,
        patch_grid=(2, 2),
        image_size=(8, 8),
        foreground_threshold=0.5,
    )

    assert result.patch_grid == (2, 2)
    assert result.foreground_mask.shape == (2, 4)
    assert len(result.images) == 2
    assert all(image.mode == "RGB" and image.size == (8, 8) for image in result.images)


def test_patch_pca_keeps_rejected_patches_black():
    result = project_patch_embeddings(
        torch.rand(1, 4, 5),
        patch_grid=(2, 2),
        image_size=(4, 4),
        foreground_threshold=1.0,
    )

    assert not result.foreground_mask.any()
    assert np.asarray(result.images[0]).max() == 0


def test_patch_pca_can_select_the_low_side_as_foreground():
    embeddings = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])

    high = project_patch_embeddings(
        embeddings,
        patch_grid=(2, 2),
        image_size=(2, 2),
        foreground_threshold=0.5,
        foreground_side="high",
    )
    low = project_patch_embeddings(
        embeddings,
        patch_grid=(2, 2),
        image_size=(2, 2),
        foreground_threshold=0.5,
        foreground_side="low",
    )

    assert torch.equal(low.foreground_mask, ~high.foreground_mask)


def test_patch_pca_validates_patch_count():
    with pytest.raises(ValueError, match="patch_grid describes 4 patches"):
        project_patch_embeddings(
            torch.rand(1, 3, 5),
            patch_grid=(2, 2),
            image_size=(4, 4),
        )


def test_patch_token_extraction_removes_prefix_and_register_tokens():
    features = torch.rand(2, 8, 6)

    patches = _patch_tokens_from_features(features, patch_count=4)

    assert torch.equal(patches, features[:, 4:])


def test_patch_token_extraction_prefers_dinov2_patch_tokens():
    patch_tokens = torch.rand(2, 4, 6)

    patches = _patch_tokens_from_features(
        {
            "x_norm_clstoken": torch.rand(2, 6),
            "x_norm_patchtokens": patch_tokens,
        },
        patch_count=4,
    )

    assert patches is patch_tokens


def test_patch_pca_projection_can_be_saved_and_reused_with_frozen_scaling(tmp_path):
    embeddings = torch.rand(2, 4, 6)
    fitted = project_patch_embeddings(
        embeddings,
        patch_grid=(2, 2),
        image_size=(8, 8),
        foreground_threshold=0.4,
        foreground_side="low",
    )
    projection_path = tmp_path / "projection.npz"
    assert fitted.projection is not None
    save_patch_pca_projection(fitted.projection, projection_path)

    loaded = load_patch_pca_projection(projection_path)
    reused = project_patch_embeddings(
        embeddings,
        patch_grid=(2, 2),
        image_size=(8, 8),
        foreground_threshold=0.9,
        foreground_side="high",
        projection=loaded,
    )

    assert torch.equal(reused.foreground_mask, fitted.foreground_mask)
    assert all(
        np.array_equal(np.asarray(actual), np.asarray(expected))
        for actual, expected in zip(reused.images, fitted.images, strict=True)
    )
