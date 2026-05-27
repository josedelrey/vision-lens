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

Run the exercise pipeline from an example script:

```bash
python examples/run_vit_attention.py
```

Or, once the CLI is implemented:

```bash
vision-lens --config configs/vit_attention.example.yaml
```

Input images should go in `data/samples/`. Generated attention maps and
overlays should go in `outputs/`.
