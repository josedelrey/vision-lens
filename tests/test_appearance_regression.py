import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vision_lens.feature_pca import project_patch_embeddings
from vision_lens.visualization import image_grid, overlay_attention, render_heatmap

BASELINE_PATH = Path(__file__).parent / "baselines" / "appearance.json"


def test_attention_gradcam_and_rollout_renderer_matches_recorded_pixels():
    baseline = _baseline()["renderer"]
    base = Image.fromarray(
        np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3),
        mode="RGB",
    )
    attention_map = torch.tensor([[[[0.0, 0.25], [0.75, 1.0]]]])

    heatmap = render_heatmap(attention_map, cmap="viridis")
    overlay = overlay_attention(
        base,
        attention_map,
        alpha=0.8,
        cmap="viridis",
    )

    _assert_image_matches(heatmap, baseline["heatmap"])
    _assert_image_matches(overlay, baseline["overlay"])


def test_dinov2_pca_projection_and_clean_grid_match_recorded_pixels():
    baseline = _baseline()["pca"]
    embeddings = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.2, 0.0, 0.1],
                [2.0, 0.1, 0.5, 0.0],
                [3.0, 1.0, 0.2, 0.4],
            ],
            [
                [0.2, 0.0, 0.1, 0.0],
                [1.2, 0.3, 0.0, 0.2],
                [2.2, 0.0, 0.7, 0.1],
                [4.0, 1.1, 0.4, 0.8],
            ],
        ]
    )

    result = project_patch_embeddings(
        embeddings,
        patch_grid=(2, 2),
        image_size=(8, 8),
        foreground_threshold=0.5,
        foreground_side="low",
    )
    comparison = image_grid(
        result.images,
        columns=2,
        background="white",
        gap=16,
        padding=16,
    )

    assert result.foreground_mask.tolist() == baseline["foreground_mask"]
    for image, expected in zip(result.images, baseline["images"], strict=True):
        _assert_image_matches(image, expected)
    _assert_image_matches(comparison, baseline["comparison_grid"])

    for image, mask in zip(result.images, result.foreground_mask, strict=True):
        array = np.asarray(image)
        for row, column in np.argwhere(~mask.reshape(2, 2).numpy()):
            y = row * (array.shape[0] - 1)
            x = column * (array.shape[1] - 1)
            assert np.all(array[y, x] == 0)


def _baseline():
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _assert_image_matches(image, expected):
    assert image.mode == expected["mode"]
    assert list(image.size) == expected["size"]
    assert hashlib.sha256(image.tobytes()).hexdigest() == expected["sha256"]
