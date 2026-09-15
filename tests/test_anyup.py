import torch
import torch.nn.functional as functional

from vision_lens import anyup


class _FakeAnyUp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.call = None

    def forward(
        self,
        image,
        features,
        *,
        output_size,
        q_chunk_size,
    ):
        self.call = (image, features, output_size, q_chunk_size)
        return functional.interpolate(features, size=output_size, mode="nearest")


def test_prepare_anyup_image_replaces_source_normalization():
    pixels = torch.tensor([[[[0.2]], [[0.4]], [[0.8]]]])
    source_mean = (0.1, 0.2, 0.3)
    source_std = (0.5, 0.25, 0.1)
    source_normalized = (
        pixels - torch.tensor(source_mean)[None, :, None, None]
    ) / torch.tensor(source_std)[None, :, None, None]

    prepared = anyup.prepare_anyup_image(
        source_normalized,
        source_mean=source_mean,
        source_std=source_std,
    )
    expected = (
        pixels - torch.tensor(anyup.IMAGENET_MEAN)[None, :, None, None]
    ) / torch.tensor(anyup.IMAGENET_STD)[None, :, None, None]

    assert torch.allclose(prepared, expected)


def test_load_anyup_uses_official_multi_backbone_model_and_caches(monkeypatch):
    loaded = []

    def fake_load(repository, model_name, **options):
        loaded.append((repository, model_name, options))
        return _FakeAnyUp()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    anyup.load_anyup.cache_clear()

    first = anyup.load_anyup("cpu")
    second = anyup.load_anyup("cpu")

    assert first is second
    assert loaded == [
        (
            "wimmerth/anyup:checkpoint_v2",
            "anyup_multi_backbone",
            {
                "pretrained": True,
                "use_natten": False,
                "device": "cpu",
                "trust_repo": True,
            },
        )
    ]
    assert not first.training
    anyup.load_anyup.cache_clear()


def test_upsample_features_delegates_to_anyup():
    model = _FakeAnyUp()
    image = torch.rand(1, 3, 8, 10)
    features = torch.rand(1, 5, 2, 2)

    result = anyup.upsample_features(
        image,
        features,
        (6, 7),
        model=model,
        q_chunk_size=3,
    )

    assert result.shape == (1, 5, 6, 7)
    assert model.call is not None
    called_image, called_features, output_size, q_chunk_size = model.call
    assert torch.equal(called_image, image)
    assert torch.equal(called_features, features)
    assert output_size == (6, 7)
    assert q_chunk_size == 3
