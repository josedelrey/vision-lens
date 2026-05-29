import torch

from vision_lens.visualization import render_heatmap


def test_render_heatmap_accepts_configurable_cmap():
    heatmap = render_heatmap(torch.rand(1, 1, 8, 8), cmap="viridis")

    assert heatmap.mode == "RGB"
    assert heatmap.size == (8, 8)
