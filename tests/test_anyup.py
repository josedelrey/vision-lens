import torch
import torch.nn.functional as functional

from vision_lens.analysis import anyup


class _FakeAnyUp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.cross_decode = torch.nn.Module()
        self.cross_decode.window_ratio = 0.1
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


class _IdentityRope(torch.nn.Module):
    def forward(self, values, _coordinates):
        return values


class _FakeCrossAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_q = torch.nn.RMSNorm(4)
        self.norm_k = torch.nn.RMSNorm(4)
        self.attention = torch.nn.MultiheadAttention(
            4,
            1,
            dropout=0.0,
            batch_first=True,
        )


class _FakeDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.cross_attn = _FakeCrossAttention()
        self.window_ratio = 0.25


class _FakeStreamingAnyUp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = torch.nn.Conv2d(3, 4, 1, bias=False)
        self.query_encoder = torch.nn.Identity()
        self.key_encoder = torch.nn.Identity()
        self.key_features_encoder = torch.nn.Identity()
        self.aggregation = torch.nn.Conv2d(8, 4, 1, bias=False)
        self.cross_decode = _FakeDecoder()
        self.rope = _IdentityRope()


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
    assert first.cross_decode.window_ratio == 0.1
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
    )

    assert result.shape == (1, 5, 6, 7)
    assert model.call is not None
    called_image, called_features, output_size, q_chunk_size = model.call
    assert torch.equal(called_image, image)
    assert torch.equal(called_features, features)
    assert output_size == (6, 7)
    assert q_chunk_size is None


def test_upsample_features_uses_streaming_when_chunked(monkeypatch):
    model = _FakeAnyUp()
    image = torch.rand(1, 3, 4, 5)
    features = torch.rand(1, 2, 2, 3)
    expected = torch.rand(1, 2, 6, 7)
    calls = []

    def fake_stream(
        image,
        features,
        output_size,
        *,
        model,
        q_chunk_size,
        attention_mode,
    ):
        calls.append(
            (image, features, output_size, model, q_chunk_size, attention_mode)
        )
        return expected

    monkeypatch.setattr(anyup, "upsample_values_streaming", fake_stream)

    result = anyup.upsample_features(
        image,
        features,
        (6, 7),
        model=model,
        q_chunk_size=3,
        attention_mode="soft",
    )

    assert result is expected
    assert calls == [(image, features, (6, 7), model, 3, "soft")]


def test_soft_attention_bias_tapers_before_blocking_distant_keys():
    bias = anyup._soft_attention_bias_rows(
        (8, 8),
        (4, 4),
        0,
        8,
        0.2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert bias.shape == (64, 16)
    assert torch.any(bias == 0)
    assert torch.any(torch.isfinite(bias) & (bias < 0))
    assert torch.any(torch.isneginf(bias))


def test_local_attention_tiles_match_slices_of_full_masks():
    output_size = (13, 17)
    feature_size = (6, 8)
    query_bounds = (3, 10, 4, 14)
    key_bounds = anyup._local_key_bounds(
        output_size,
        feature_size,
        *query_bounds,
        0.2,
    )
    full_hard = anyup._attention_mask_rows(
        output_size,
        feature_size,
        query_bounds[0],
        query_bounds[1],
        0.2,
        device=torch.device("cpu"),
    ).reshape(
        query_bounds[1] - query_bounds[0],
        output_size[1],
        feature_size[0],
        feature_size[1],
    )
    tiled_hard = anyup._attention_mask_tile(
        output_size,
        feature_size,
        *query_bounds,
        *key_bounds,
        0.2,
        device=torch.device("cpu"),
    )
    row_start, row_end, column_start, column_end = key_bounds
    expected_hard = full_hard[
        :,
        query_bounds[2] : query_bounds[3],
        row_start:row_end,
        column_start:column_end,
    ].reshape_as(tiled_hard)

    full_soft = anyup._soft_attention_bias_rows(
        output_size,
        feature_size,
        query_bounds[0],
        query_bounds[1],
        0.2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).reshape_as(full_hard)
    tiled_soft = anyup._soft_attention_bias_tile(
        output_size,
        feature_size,
        *query_bounds,
        *key_bounds,
        0.2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_soft = full_soft[
        :,
        query_bounds[2] : query_bounds[3],
        row_start:row_end,
        column_start:column_end,
    ].reshape_as(tiled_soft)

    assert torch.equal(tiled_hard, expected_hard)
    assert torch.equal(tiled_soft, expected_soft)


def test_query_tiles_respect_chunk_budget():
    assert anyup._query_tile_size((672, 672), 4096) == (64, 64)
    assert anyup._query_tile_size((7, 9), 18) == (3, 6)
    for output_size, chunk_size in [((3, 100), 5), ((100, 3), 5), ((8, 8), 1)]:
        tile_height, tile_width = anyup._query_tile_size(output_size, chunk_size)
        assert tile_height * tile_width <= chunk_size


def test_streaming_anyup_matches_full_attention_with_compact_values():
    torch.manual_seed(3)
    model = _FakeStreamingAnyUp().eval()
    image = torch.rand(1, 3, 4, 5)
    features = torch.rand(1, 4, 2, 3)
    values = torch.rand(1, 3, 2, 3)
    output_size = (7, 9)

    expected = _full_anyup_values(model, image, features, values, output_size)
    actual = anyup.upsample_values_streaming(
        image,
        features,
        output_size,
        values=values,
        model=model,
        q_chunk_size=18,
    )

    assert actual.shape == (1, 3, 7, 9)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_streaming_soft_anyup_matches_full_attention():
    torch.manual_seed(5)
    model = _FakeStreamingAnyUp().eval()
    image = torch.rand(1, 3, 4, 5)
    features = torch.rand(1, 4, 2, 3)
    values = torch.rand(1, 2, 2, 3)
    output_size = (7, 9)

    expected = _full_anyup_values(
        model,
        image,
        features,
        values,
        output_size,
        attention_mode="soft",
    )
    actual = anyup.upsample_values_streaming(
        image,
        features,
        output_size,
        values=values,
        model=model,
        q_chunk_size=18,
        attention_mode="soft",
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_memory_efficient_attention_matches_multihead_attention():
    torch.manual_seed(7)
    cross_attention = _FakeCrossAttention()
    cross_attention.attention = torch.nn.MultiheadAttention(
        4,
        2,
        dropout=0.0,
        batch_first=True,
    )
    queries = cross_attention.norm_q(torch.rand(2, 5, 4))
    keys = cross_attention.norm_k(torch.rand(2, 6, 4))
    values = torch.rand(2, 6, 3)
    hard_mask = torch.tensor(
        [
            [False, False, True, True, True, True],
            [False, False, False, True, True, True],
            [True, False, False, False, True, True],
            [True, True, False, False, False, True],
            [True, True, True, False, False, False],
        ]
    )

    expected = anyup._materialized_attention_values(
        cross_attention.attention,
        queries,
        keys,
        values,
        hard_mask,
    )
    actual = anyup._attention_weighted_values(
        cross_attention,
        queries,
        keys,
        values,
        hard_mask,
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_memory_efficient_attention_matches_soft_attention_bias():
    torch.manual_seed(11)
    cross_attention = _FakeCrossAttention()
    queries = cross_attention.norm_q(torch.rand(1, 4, 4))
    keys = cross_attention.norm_k(torch.rand(1, 3, 4))
    values = torch.rand(1, 3, 2)
    soft_bias = torch.tensor(
        [
            [0.0, -0.5, float("-inf")],
            [-0.2, 0.0, -1.0],
            [-1.0, -0.2, 0.0],
            [float("-inf"), -0.5, 0.0],
        ]
    )

    expected = anyup._materialized_attention_values(
        cross_attention.attention,
        queries,
        keys,
        values,
        soft_bias,
    )
    actual = anyup._attention_weighted_values(
        cross_attention,
        queries,
        keys,
        values,
        soft_bias,
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def _full_anyup_values(
    model,
    image,
    features,
    values,
    output_size,
    *,
    attention_mode="hard",
):
    encoded = model.image_encoder(image)
    height, width = encoded.shape[-2:]
    coordinates = anyup._coordinates(
        height,
        width,
        device=encoded.device,
        dtype=encoded.dtype,
    )
    encoded = encoded.permute(0, 2, 3, 1).reshape(1, height * width, 4)
    encoded = model.rope(encoded, coordinates)
    encoded = encoded.reshape(1, height, width, 4).permute(0, 3, 1, 2)
    queries = functional.adaptive_avg_pool2d(
        model.query_encoder(encoded),
        output_size,
    )
    queries = model.cross_decode.conv2d(queries)
    keys = functional.adaptive_avg_pool2d(
        model.key_encoder(encoded),
        features.shape[-2:],
    )
    feature_keys = model.key_features_encoder(functional.normalize(features, dim=1))
    keys = model.aggregation(torch.cat((keys, feature_keys), dim=1))
    query_sequence = queries.permute(0, 2, 3, 1).reshape(1, -1, 4)
    key_sequence = keys.permute(0, 2, 3, 1).reshape(1, -1, 4)
    value_sequence = values.permute(0, 2, 3, 1).reshape(1, -1, values.shape[1])
    mask_arguments = (
        output_size,
        features.shape[-2:],
        0,
        output_size[0],
        model.cross_decode.window_ratio,
    )
    if attention_mode == "soft":
        mask = anyup._soft_attention_bias_rows(
            *mask_arguments,
            device=image.device,
            dtype=image.dtype,
        )
    else:
        mask = anyup._attention_mask_rows(
            *mask_arguments,
            device=image.device,
        )
    _ignored, attention = model.cross_decode.cross_attn.attention(
        model.cross_decode.cross_attn.norm_q(query_sequence),
        model.cross_decode.cross_attn.norm_k(key_sequence),
        key_sequence,
        average_attn_weights=True,
        attn_mask=mask,
    )
    result = torch.einsum("bij,bjd->bid", attention, value_sequence)
    return result.reshape(1, *output_size, values.shape[1]).permute(0, 3, 1, 2)
