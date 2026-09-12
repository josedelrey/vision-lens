# Vision Lens

[![CI](https://github.com/josedelrey/vision-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/josedelrey/vision-lens/actions/workflows/ci.yml)

Vision Lens turns pretrained vision-model internals into clear, reproducible
attention, rollout, Grad-CAM, and patch-feature PCA visualizations for images
and video frames.

> **Results gallery placeholder**
>
> Final attention, rollout, Grad-CAM, and PCA examples will be added here after
> visual review.

## Install

Vision Lens targets Linux and uses [uv](https://docs.astral.sh/uv/) for Python,
dependency, and virtual-environment management. Clone the repository, install
uv, then create the complete development environment from the committed lock:

```bash
uv sync --locked --all-extras
```

`uv` creates `.venv` automatically. Run project commands through `uv run`, so
shell activation is not required. For a smaller runtime-only environment:

```bash
uv sync --locked --no-dev
```

Add `--extra video` to include video support. The development command above
uses `--all-extras`, so it already includes video support.

Dependency declarations and development tools live in `pyproject.toml`; exact
versions are recorded in `uv.lock`. After intentionally changing dependency
constraints, refresh the lock with `uv lock` (or `uv lock --upgrade` to upgrade
all dependencies), then commit both files.

## Quick starts

Run one image through the DINO attention preset:

```bash
uv run vision-lens --preset dino-vits8-attention --set input.limit=1
```

Reproduce the DINOv2 PCA example:

```bash
uv run python scripts/run_dinov2_pca.py
```

The DINOv2 PCA preset fits one shared projection across both example images,
so foreground selection and colors remain comparable:

> **PCA result placeholder**
>
> The final shared-projection comparison will be added here.

> **Video demo placeholder**
>
> A reviewed frame-based video example and command will be added here.

Outputs are written below `outputs/`. Each run also creates a manifest with the
resolved configuration, model identity, versions, inputs, and generated files.

## Configure a workflow

Use a preset as a stable starting point and override only what changes:

```yaml
preset: dinov2-reg4-attention

input:
  files: [photo.jpg]

runtime:
  device: auto
  batch_size: 4

visualization:
  overlay_alpha: 0.6

output:
  directory: results
  overwrite: error
```

Validate or inspect the resolved configuration before loading a model:

```bash
uv run vision-lens validate --config workflow.yaml
uv run vision-lens resolve --config workflow.yaml
uv run vision-lens run --config workflow.yaml
```

Resolution order is **defaults → preset → YAML → CLI overrides**. Relative
paths in YAML files are resolved from the YAML file's directory.

## Included workflows

| Preset | Analysis | Model | Input size |
|---|---|---|---:|
| `dino-vits8-attention` | attention | DINO ViT-S/8 | 224 |
| `dinov2-reg4-attention` | attention | DINOv2 ViT-S/14 + registers | 672 |
| `dinov2-reg4-rollout` | rollout | DINOv2 ViT-S/14 + registers | 672 |
| `resnet50-gradcam` | Grad-CAM | ResNet-50 | 672 |
| `dinov2-pca` | patch PCA | DINOv2 ViT-B/14 | 672 |

Presets preserve the established rendering, including no-crop resizing,
per-map normalization, opacity, colors, grids, and PCA projection behavior.

The [configuration reference](https://github.com/josedelrey/vision-lens/blob/main/docs/configuration.md)
documents every setting, default, validation rule, and CLI override.

Attention and Grad-CAM visualizations are diagnostic views, not causal
explanations. Video processing analyzes sampled frames independently; source
audio is not included in exports. Pretrained weights require network access on
first use, and an optional `HF_TOKEN` only improves Hugging Face download rate
limits. Custom timm and torchvision models are not guaranteed to expose the
internals required by each analysis.

## License

Vision Lens is released under the [MIT License](https://github.com/josedelrey/vision-lens/blob/main/LICENSE).
Bundled example-image provenance is recorded in the
[attribution file](https://github.com/josedelrey/vision-lens/blob/main/data/examples/ATTRIBUTION.md).
