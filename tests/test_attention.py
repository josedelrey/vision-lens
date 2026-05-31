import pytest
import torch

from vision_lens.attention import (
    class_token_attention_to_map,
    compute_attention_rollout,
    infer_patch_grid,
    infer_prefix_tokens_and_patch_grid,
    token_attention_to_map,
)


def test_infer_patch_grid_from_patch_size():
    assert infer_patch_grid(
        num_patches=196,
        image_size=(224, 224),
        patch_size=(16, 16),
    ) == (14, 14)


def test_infer_prefix_tokens_for_standard_vit():
    prefix_tokens, grid = infer_prefix_tokens_and_patch_grid(
        token_count=197,
        image_size=(224, 224),
        patch_size=(16, 16),
    )

    assert prefix_tokens == 1
    assert grid == (14, 14)


def test_infer_prefix_tokens_for_dinov2_register_vit():
    prefix_tokens, grid = infer_prefix_tokens_and_patch_grid(
        token_count=1374,
        image_size=(518, 518),
        patch_size=(14, 14),
    )

    assert prefix_tokens == 5
    assert grid == (37, 37)


def test_infer_prefix_tokens_preserves_square_fallback_without_patch_size():
    prefix_tokens, grid = infer_prefix_tokens_and_patch_grid(
        token_count=197,
        image_size=(224, 224),
        patch_size=None,
    )

    assert prefix_tokens == 1
    assert grid == (14, 14)


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


def test_class_token_attention_to_map_mean_fusion_values():
    attention = torch.zeros(1, 2, 5, 5)
    attention[:, 0, 0, 1:] = torch.tensor([1.0, 3.0, 5.0, 7.0])
    attention[:, 1, 0, 1:] = torch.tensor([3.0, 5.0, 7.0, 9.0])

    maps = class_token_attention_to_map(
        attention,
        image_size=(2, 2),
        patch_size=(1, 1),
        head_fusion="mean",
        normalize=False,
    )

    assert torch.equal(maps, torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]]))


def test_class_token_attention_to_map_max_fusion_values():
    attention = torch.zeros(1, 2, 5, 5)
    attention[:, 0, 0, 1:] = torch.tensor([1.0, 8.0, 5.0, 2.0])
    attention[:, 1, 0, 1:] = torch.tensor([3.0, 5.0, 7.0, 9.0])

    maps = class_token_attention_to_map(
        attention,
        image_size=(2, 2),
        patch_size=(1, 1),
        head_fusion="max",
        normalize=False,
    )

    assert torch.equal(maps, torch.tensor([[[[3.0, 8.0], [7.0, 9.0]]]]))


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


def test_class_token_attention_to_map_skips_register_tokens():
    attention = torch.zeros(1, 2, 9, 9)
    attention[:, :, 0, 5:] = torch.rand(1, 2, 4)

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


def test_compute_attention_rollout_uses_max_head_fusion():
    attention = torch.tensor(
        [
            [
                [
                    [0.1, 0.2, 0.7],
                    [0.3, 0.4, 0.3],
                    [0.1, 0.8, 0.1],
                ],
                [
                    [0.9, 0.05, 0.05],
                    [0.1, 0.8, 0.1],
                    [0.4, 0.2, 0.4],
                ],
            ]
        ]
    )

    rollout = compute_attention_rollout(
        [attention],
        head_fusion="max",
        discard_ratio=0,
    )

    expected = torch.tensor(
        [
            [
                [1.9 / 2.8, 0.2 / 2.8, 0.7 / 2.8],
                [0.3 / 2.4, 1.8 / 2.4, 0.3 / 2.4],
                [0.4 / 2.6, 0.8 / 2.6, 1.4 / 2.6],
            ]
        ]
    )
    assert torch.allclose(rollout[0], expected)


def test_compute_attention_rollout_discards_lowest_attention_values():
    attention = torch.tensor([[[[0.4, 0.1], [0.2, 0.3]]]])

    rollout = compute_attention_rollout(
        [attention],
        head_fusion="max",
        discard_ratio=0.25,
    )

    expected = torch.tensor([[[1.0, 0.0], [0.2 / 1.5, 1.3 / 1.5]]])
    assert torch.allclose(rollout[0], expected)


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


def test_token_attention_to_map_skips_register_tokens():
    token_attention = torch.zeros(1, 9, 9)
    token_attention[:, 0, 5:] = torch.tensor([0.1, 0.2, 0.3, 0.4])

    maps = token_attention_to_map(
        token_attention,
        image_size=(4, 4),
        patch_size=(2, 2),
    )

    assert maps.shape == (1, 1, 4, 4)
    assert torch.all(maps >= 0)
    assert torch.all(maps <= 1)
