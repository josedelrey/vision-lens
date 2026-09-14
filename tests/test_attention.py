import pytest
import torch
from torch import nn

from vision_lens.attention import (
    class_token_attention_to_map,
    compute_attention_rollout,
    extract_gradcam,
    infer_patch_grid,
    infer_prefix_tokens_and_patch_grid,
    token_attention_to_map,
)
from vision_lens.models import ModelMetadata


def test_gradcam_preserves_rectangular_input_geometry():
    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(4, 2)

        def forward(self, inputs):
            return self.classifier(self.pool(self.features(inputs)).flatten(1))

    metadata = ModelMetadata(
        architecture="cnn",
        backend="torchvision",
        name="tiny",
        pretrained=False,
        device="cpu",
        input_size=(3, 12, 20),
        image_size=(12, 20),
        patch_size=None,
        num_classes=2,
        data_config={},
    )
    result = extract_gradcam(
        TinyCNN(),
        torch.rand(1, 3, 12, 20, requires_grad=True),
        metadata,
        target_layer="features.0",
        target_classes=[1],
    )

    assert result.maps.shape == (1, 1, 12, 20)


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
