# Configuration

Vision Lens configuration uses seven sections: `input`, `model`,
`preprocessing`, `analysis`, `runtime`, `visualization`, and `output`. A
top-level `preset` is optional. Unknown keys and settings belonging to a
different analysis method are rejected before a model is loaded.

## Complete example

Paths are relative to the YAML file, not the current working directory.

```yaml
preset: dinov2-reg4-attention

input:
  paths:
    - ../data/examples/1.jpg

model:
  architecture: vit
  backend: timm
  name: hf_hub:timm/vit_small_patch14_reg4_dinov2.lvd142m
  pretrained: true
  options: {}

preprocessing:
  image_size: 672

analysis:
  method: attention
  layers: [2, 5, 8, 11]
  heads: null
  head_fusion: mean

runtime:
  device: auto

visualization:
  overlay_alpha: 0.8
  cmap: viridis
  grid_format: pdf

output:
  directory: ../outputs/custom_attention
```

## Settings

### Top level

| Setting | Default | Description | Example |
|---|---|---|---|
| `preset` | none | Named settings applied before this file. | `dinov2-pca` |

### `input`

| Setting | Default | Description | Example |
|---|---|---|---|
| `paths` | required | Non-empty list of input image files. | `[../data/examples/1.jpg]` |

Every path must identify an existing file. Relative paths are resolved from the
directory containing the YAML file and normalized to absolute paths.

### `model`

| Setting | Default | Description | Example |
|---|---|---|---|
| `architecture` | required | Model family: currently `vit` or `cnn`. | `vit` |
| `backend` | required | Loader: currently `timm` or `torchvision`. | `timm` |
| `name` | required | Backend model identifier. | `resnet50` |
| `pretrained` | `true` | Load pretrained weights. | `false` |
| `options` | `null` | Extra backend model-loader keywords. | `{drop_rate: 0.1}` |

`model.options.img_size` is forbidden. Use `preprocessing.image_size`, which is
the single authoritative model-input size.

### `preprocessing`

| Setting | Default | Description | Example |
|---|---|---|---|
| `image_size` | `672` | Square model-input width and height. | `224` |

Images are resized directly to this size without cropping. Known fixed-size
models reject incompatible values. For example, `vit_small_patch8_224.dino`
requires `224`.

### `analysis`

| Setting | Default | Applies to | Description | Example |
|---|---|---|---|---|
| `method` | `attention` | all | Analysis to run. | `rollout` |
| `layers` | required | attention, rollout | Layer indices or `all`. | `[2, 5, 8, 11]` |
| `heads` | `null` | attention, rollout | Head indices; `null` selects all before fusion. | `[0, 1]` |
| `head_fusion` | `mean` | attention, rollout | `mean`, `max`, or `none`. | `none` |
| `target_layer` | `null` | Grad-CAM | Module path; `null` selects automatically. | `layer4` |
| `foreground_threshold` | `0.5` | patch PCA | First-component foreground cutoff. | `0.6` |
| `foreground_side` | `high` | patch PCA | Keep the `high` or `low` side. | `low` |

Allowed methods are `attention`, `rollout`, `gradcam`, and `patch_pca`.
Attention, rollout, and patch PCA require a `vit` model with the `timm`
backend. Grad-CAM requires a `cnn` model with the `torchvision` backend. Known
layer and head limits are validated before model loading.

### `runtime`

| Setting | Default | Description | Example |
|---|---|---|---|
| `device` | `auto` | `auto`, `cpu`, `cuda`, or `mps`. | `cpu` |

### `visualization`

| Setting | Default | Description | Example |
|---|---|---|---|
| `overlay_alpha` | `0.45` | Heatmap opacity from 0 to 1. | `0.8` |
| `cmap` | `viridis` | Matplotlib color-map name. | `magma` |
| `grid_format` | `png` | Grid format: `png`, `pdf`, or `svg`. | `pdf` |

These settings currently affect attention, rollout, and Grad-CAM rendering.
Patch PCA retains its appearance-preserving rendering until the additional
rendering controls are introduced.

### `output`

| Setting | Default | Description | Example |
|---|---|---|---|
| `directory` | required | Destination for generated files. | `../outputs/run-1` |

## Precedence and overrides

Settings are merged in this order, with later values winning:

1. Built-in field defaults.
2. The selected preset.
3. Values in the user YAML file.
4. Repeated CLI `--set` overrides.

CLI override values use YAML syntax:

```bash
vision-lens --config configs/vit_attention.example.yaml \
  --set runtime.device=cpu \
  --set analysis.heads='[0, 1]' \
  --set analysis.head_fusion=none
```

## Validation and resolved configuration

Validate without loading weights or processing images:

```bash
vision-lens validate --config configs/vit_attention.example.yaml
```

Print the final merged configuration, including defaults and absolute paths:

```bash
vision-lens resolve --config configs/vit_attention.example.yaml
```

Both commands accept `--preset` and repeated `--set` options. Normal `run`
execution performs the same validation before loading a model:

```bash
vision-lens run --preset dinov2-pca --set runtime.device=cpu
```
