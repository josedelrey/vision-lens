import pytest
import torch

from vision_lens.attention import (
    class_token_attention_to_map,
    compute_attention_rollout,
    infer_patch_grid,
    token_attention_to_map,
)


def test_infer_patch_grid_from_patch_size():
    assert infer_patch_grid(
        num_patches=196,
        image_size=(224, 224),
        patch_size=(16, 16),
    ) == (14, 14)


def test_class_token_attention_to_map_fuses_and_resizes_heads():
    attention = torch.zeros(1, 2, 5, 5)
    attention[:, 0, 0, 1:] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    attention[:, 1, 0, 1:] = torch.tensor([0.4, 0.3, 0.2, 0.1])

    maps = class_token_attention_to_map(
        attention,
        image_size=(4, 4),
        patch_size=(2, 2),
        heads=None,
        head_fusion="mean",
    )

    assert maps.shape == (1, 1, 4, 4)
    assert torch.all(maps >= 0)
    assert torch.all(maps <= 1)


def test_class_token_attention_to_map_can_keep_individual_heads():
    attention = torch.zeros(1, 3, 5, 5)
    attention[:, :, 0, 1:] = torch.rand(1, 3, 4)

    maps = class_token_attention_to_map(
        attention,
        image_size=(4, 4),
        patch_size=(2, 2),
        heads=[0, 2],
        head_fusion="none",
    )

    assert maps.shape == (1, 2, 4, 4)


def test_infer_patch_grid_rejects_non_square_unknown_grid():
    with pytest.raises(ValueError):
        infer_patch_grid(
            num_patches=6,
            image_size=(224, 224),
            patch_size=None,
        )


def test_compute_attention_rollout_returns_one_map_per_layer():
    attention = torch.eye(5).reshape(1, 1, 5, 5)
    rollout = compute_attention_rollout([attention, attention])

    assert len(rollout) == 2
    assert rollout[0].shape == (1, 5, 5)
    assert rollout[1].shape == (1, 5, 5)


def test_token_attention_to_map_resizes_rollout_attention():
    token_attention = torch.zeros(1, 5, 5)
    token_attention[:, 0, 1:] = torch.tensor([0.1, 0.2, 0.3, 0.4])

    maps = token_attention_to_map(
        token_attention,
        image_size=(4, 4),
        patch_size=(2, 2),
    )

    assert maps.shape == (1, 1, 4, 4)
    assert torch.all(maps >= 0)
    assert torch.all(maps <= 1)
