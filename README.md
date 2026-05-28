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

Or with the CLI:

```bash
python -m vision_lens.cli --config configs/vit_attention.example.yaml
```

If the package is installed, the console command is also available:

```bash
vision-lens --config configs/vit_attention.example.yaml
```

Input images live in `data/samples/`. Generated attention maps and overlays go
in `outputs/`.

The example config uses curated sample images from `scikit-image`. Extra local
images placed in `data/samples/` are ignored by Git by default.
