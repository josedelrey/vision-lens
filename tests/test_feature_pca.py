import numpy as np
import pytest
import torch

from vision_lens.feature_pca import (
    _patch_tokens_from_features,
    project_patch_embeddings,
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
    embeddings = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]
    )

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
