# vision-lens

`vision-lens` is a small Python package for a computer vision exercise on
visualizing attention maps and patch features from pretrained vision models.

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

## Appearance-preserving presets

The current workflows are available as named presets:

- `dino-vits8-attention`
- `dinov2-reg4-attention`
- `dinov2-reg4-rollout`
- `resnet50-gradcam`
- `dinov2-pca`

Each preset also has a directly runnable script:

```bash
python scripts/run_dino_vits8_attention.py
python scripts/run_dinov2_reg4_attention.py
python scripts/run_dinov2_reg4_rollout.py
python scripts/run_resnet50_gradcam.py
python scripts/run_dinov2_pca.py
```

The scripts accept the same one-off overrides as the CLI, for example
`python scripts/run_dinov2_pca.py --set runtime.device=cpu`.

List them or run one directly:

```bash
vision-lens --list-presets
vision-lens --preset dinov2-reg4-rollout
```

A config can select a preset and override only the settings that should change:

```yaml
preset: dino-vits8-attention

images:
  paths:
    - data/examples/1.jpg

visualization:
  overlay_alpha: 0.6
```

One-off CLI overrides use YAML values and may be repeated:

```bash
vision-lens --preset dino-vits8-attention \
  --set attention.heads='[0, 1, 2]' \
  --set attention.head_fusion=none
```

Preset settings are applied first, followed by values from the config file and
then `--set` overrides. Omitting a preset retains the previous config behavior.

The eight bundled example photographs live in `data/examples/`. They are
center-cropped to 672 × 672 pixels, encoded as metadata-free JPEGs, and covered
by the attribution details in `data/examples/ATTRIBUTION.md`. Generated
attention maps and overlays go in `outputs/`.

Optional helpers are also available in Python for ViT attention rollout and a
simple torchvision CNN Grad-CAM baseline:

```python
from vision_lens.pipeline import run_gradcam, run_vit_rollout_comparison
```

## DINOv2 patch-feature PCA

To produce the black-background RGB patch-feature visualization shown in the
exercise reference, run:

```bash
vision-lens --config configs/patch_pca.dinov2.example.yaml
```

The pipeline extracts normalized DINOv2 patch tokens, fits the foreground mask
from the first shared principal component, and maps the foreground through the
next three shared PCA channels. It saves one PNG per input plus a clean,
unlabelled comparison grid at `outputs/dinov2_patch_pca/`.

The shared PCA basis is important: colors remain comparable across every image
listed in the same config. Adjust `patch_pca.foreground_threshold` if the
default `0.5` mask includes too much background or hides part of the subject.
PCA component signs are arbitrary, so switch `patch_pca.foreground_side`
between `low` and `high` if the subject and background appear reversed.

The same workflow is available from Python:

```python
from vision_lens.pipeline import run_patch_pca

result = run_patch_pca("configs/patch_pca.dinov2.example.yaml")
```
