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

Run one image through the DINO attention configuration:

```bash
uv run vision-lens --config configs/vit_attention.yaml --set input.limit=1
```

Reproduce the DINOv2 PCA example:

```bash
uv run vision-lens --config configs/patch_pca.dinov2.yaml
```

Runs print model-loading status and show `tqdm` progress by image or sampled
video frame in an interactive terminal. Multi-pass work has separate bars for
fitting and rendering; non-interactive logs keep only the status lines.

The DINOv2 PCA configuration fits one shared projection across the two horse images,
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

Copy an example from `configs/` and edit it for your run. Every setting must
appear in the YAML, including settings set to `null`. For patch PCA, put only
the images that should share one fit in a configuration file. Images needing
different thresholds or `foreground_side` values need separate configurations.

Validate or inspect the resolved configuration before loading a model:

```bash
uv run vision-lens validate --config workflow.yaml
uv run vision-lens resolve --config workflow.yaml
uv run vision-lens run --config workflow.yaml
```

CLI `--set` values override the complete YAML configuration. Relative
paths in YAML files and CLI overrides are resolved from the project
root (the nearest ancestor containing `pyproject.toml`), not the config file's
directory. If no project root is found, they use the current working directory.

## Included workflows

| Example configuration | Analysis | Model | Input size |
|---|---|---|---:|
| `vit_attention.yaml` | attention | DINO ViT-S/8 | 224 |
| `vit_attention.dinov2_reg4.yaml` | attention | DINOv2 ViT-S/14 + registers | 672 |
| `vit_rollout.dinov2_reg4.yaml` | rollout | DINOv2 ViT-S/14 + registers | 672 |
| `gradcam.yaml` | Grad-CAM | ResNet-50 | 672 |
| `patch_pca.dinov2.yaml` | patch PCA | DINOv2 ViT-B/14 | 672 |

The [configuration reference](https://github.com/josedelrey/vision-lens/blob/main/docs/configuration.md)
documents every setting, validation rule, and CLI override.

Attention and Grad-CAM visualizations are diagnostic views, not causal
explanations. Video processing analyzes sampled frames independently; source
audio is not included in exports. Pretrained weights require network access on
first use, and an optional `HF_TOKEN` only improves Hugging Face download rate
limits. The AnyUp visualization modes additionally download
the official multi-backbone AnyUp model and checkpoint through PyTorch Hub on
first use. The video examples use query chunking to reduce AnyUp's peak VRAM
usage while retaining their requested output size. Custom timm and torchvision
models are not guaranteed to expose the
internals required by each analysis.

## License

Vision Lens is released under the [MIT License](https://github.com/josedelrey/vision-lens/blob/main/LICENSE).
Bundled example-image provenance is recorded in the
[attribution file](https://github.com/josedelrey/vision-lens/blob/main/examples/ATTRIBUTION.md).
