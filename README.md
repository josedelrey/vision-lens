# vision-lens

`vision-lens` is a small Python package for a computer vision exercise on
visualizing attention maps from pretrained Vision Transformers.

The first target workflow is:

1. Load a pretrained ViT model.
2. Run one or more ImageNet-like images through it.
3. Extract attention maps from selected layers and heads.
4. Save heatmaps and overlays that make the model attention easier to compare.

## Installation

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

## Intended usage

Run the ViT exercise figures from an example script:

```bash
python examples/run_vit_attention.py
```

That script exports fused-head attention maps, individual-head maps, and
attention rollout comparisons under `outputs/vit_attention/`.

Run the CNN Grad-CAM baseline separately:

```bash
python examples/run_cnn_gradcam.py
```

Or with the CLI:

```bash
python -m vision_lens.cli --config configs/vit_attention.example.yaml
```

If the package is installed, the console command is also available:

```bash
vision-lens --config configs/vit_attention.example.yaml
```

The eight bundled example photographs live in `data/examples/`. They are
center-cropped to 672 × 672 pixels, encoded as metadata-free JPEGs, and covered
by the attribution details in `data/examples/ATTRIBUTION.md`. Generated
attention maps and overlays go in `outputs/`.

Optional helpers are also available in Python for ViT attention rollout and a
simple torchvision CNN Grad-CAM baseline:

```python
from vision_lens.pipeline import run_gradcam, run_vit_rollout_comparison
```
