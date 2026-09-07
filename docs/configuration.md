# Configuration

Vision Lens uses seven YAML sections: `input`, `model`, `preprocessing`,
`analysis`, `runtime`, `visualization`, and `output`. A top-level `preset` is
optional. Unknown keys and incompatible settings are rejected before model
weights are loaded.

All relative paths are resolved from the YAML file's directory. Settings are
merged in this order, with later values winning:

1. Built-in defaults.
2. The selected preset.
3. Values in the YAML file.
4. Repeated CLI `--set` overrides.

## Input

```yaml
input:
  files: [../data/examples/1.jpg]
  folders: [../photos]
  patterns: ["*.jpg", "*.png"]
  recursive: true
  limit: 100
```

| Setting | Default | Description | Example |
|---|---|---|---|
| `files` | `[]` | Explicit image files, kept in the listed order. | `[cat.jpg]` |
| `folders` | `[]` | Folders searched for matching files. | `[../photos]` |
| `patterns` | `['*.jpg', '*.jpeg', '*.png', '*.webp']` | Glob patterns applied to every folder. | `["*.jpg"]` |
| `recursive` | `false` | Search inside nested folders. | `true` |
| `limit` | `null` | Maximum inputs after expansion; `null` means all. | `50` |

At least one file must be selected. Duplicate paths are removed. The legacy
`input.paths` spelling remains accepted as an alias for `input.files`, but the
two cannot be used together.

## Model

```yaml
model:
  architecture: vit
  backend: timm
  name: hf_hub:timm/vit_small_patch14_reg4_dinov2.lvd142m
  pretrained: true
  options: {}
```

| Setting | Default | Description | Example |
|---|---|---|---|
| `architecture` | required | Model family: `vit` or `cnn`. | `vit` |
| `backend` | required | Loader: `timm` or `torchvision`. | `timm` |
| `name` | required | Backend model identifier. | `resnet50` |
| `pretrained` | `true` | Load pretrained weights. | `false` |
| `options` | `null` | Additional backend loader arguments. | `{drop_rate: 0.1}` |

`model.options.img_size` is forbidden. `preprocessing.image_size` is the one
authoritative input size.

## Preprocessing

```yaml
preprocessing:
  image_size: 672
  resize: longest
  crop: none
  pad: center
  interpolation: bicubic
  normalize: true
  mean: null
  std: null
```

| Setting | Default | Description | Example |
|---|---|---|---|
| `image_size` | `672` | Square model input width and height. | `224` |
| `resize` | `stretch` | `stretch`, `shortest`, `longest`, or `none`. | `longest` |
| `crop` | `none` | `none` or a `center` crop to `image_size`. | `center` |
| `pad` | `none` | `none` or `center` padding to `image_size`. | `center` |
| `interpolation` | `null` | Model-derived by default; override with `nearest`, `bilinear`, `bicubic`, or `lanczos`. | `bicubic` |
| `normalize` | `true` | Apply channel normalization. | `false` |
| `mean` | `null` | Model-derived RGB means, or three custom values. | `[0.5, 0.5, 0.5]` |
| `std` | `null` | Model-derived positive RGB standard deviations. | `[0.5, 0.5, 0.5]` |

`stretch` preserves the existing no-crop behavior. To retain aspect ratio, use
`longest` with `pad: center`, or `shortest` with `crop: center`. A transformed
image must end at the model's required size. Known fixed models reject invalid
sizes during configuration validation.

## Analysis

All four analyses run through the same command:

```bash
vision-lens run --config configs/gradcam.example.yaml
```

### Attention and rollout

```yaml
analysis:
  method: attention
  layers: [2, 5, 8, 11]
  heads: null
  head_fusion: mean
```

| Setting | Default | Description | Example |
|---|---|---|---|
| `method` | `attention` | `attention` or `rollout`. | `rollout` |
| `layers` | required | Layer indices or `all`. | `[2, 5, 8, 11]` |
| `heads` | `null` | Head indices; `null` selects all. | `[0, 1]` |
| `head_fusion` | `mean` | `mean`, `max`, or `none`. | `none` |

### Grad-CAM

```yaml
analysis:
  method: gradcam
  target_layer: layer4
  target_class: 207
```

| Setting | Default | Description | Example |
|---|---|---|---|
| `method` | required | Must be `gradcam`. | `gradcam` |
| `target_layer` | `null` | Module path; `null` chooses the last convolution. | `layer4` |
| `target_class` | `null` | Fixed non-negative class index; `null` uses each image's prediction. | `207` |

### Patch PCA

```yaml
analysis:
  method: patch_pca
  foreground_threshold: 0.5
  foreground_side: low
  projection: fit
  projection_path: null
  save_projection: ../outputs/pca-projection.npz
```

| Setting | Default | Description | Example |
|---|---|---|---|
| `method` | required | Must be `patch_pca`. | `patch_pca` |
| `foreground_threshold` | `0.5` | Normalized first-component cutoff. | `0.6` |
| `foreground_side` | `high` | Keep the `high` or `low` side. | `low` |
| `projection` | `fit` | Fit a shared projection or `load` one. | `load` |
| `projection_path` | `null` | Saved `.npz` loaded when `projection: load`. | `pca.npz` |
| `save_projection` | `null` | Save the fitted basis and normalization ranges. | `pca.npz` |

A loaded projection reuses its fitted foreground rule and color ranges, so new
images remain in the same PCA color space.

## Runtime

```yaml
runtime:
  batch_size: 4
  device: auto
  workers: 2
  precision: float32
  seed: 42
```

| Setting | Default | Description | Example |
|---|---|---|---|
| `batch_size` | `8` | Maximum images read, preprocessed, and analyzed together. | `4` |
| `device` | `auto` | `auto`, `cpu`, `cuda`, or `mps`. | `cuda` |
| `workers` | `0` | Threads used to read images; `0` reads sequentially. | `4` |
| `precision` | `float32` | `float32`, `float16`, or `bfloat16`. | `float16` |
| `seed` | `null` | Seed Python, NumPy, and PyTorch; `null` leaves RNG state unchanged. | `42` |

The model and preprocessing transform are created once per run. Vision Lens
then reads, preprocesses, analyzes, renders, and exports no more than
`batch_size` images at a time. Detailed tensors remain available on the Python
result object for single-batch runs; multi-batch runs report `processed_inputs`
without retaining the entire collection in memory.

PCA keeps the historical calculation for a single batch. Larger PCA runs stage
one embedding batch at a time, fit one projection and one set of normalization
bounds over the complete input set, then transform each staged batch. This
keeps colors comparable without holding every image or embedding in memory.
`shared` attention, rollout, and Grad-CAM normalization similarly uses a
bounded fitting pass before rendering. CPU runs reject `float16` (use
`bfloat16` instead), and MPS runs reject `bfloat16`.

## Visualization

```yaml
visualization:
  tile_size: [320, 240]
  columns: 2
  items_per_grid: 6
  spacing: 8
  padding: 12
  labels: true
  background: white
  dpi: 150
  overlay_alpha: 0.6
  cmap: magma
  grid_format: pdf
  normalization: shared
  normalization_range: null
```

| Setting | Default | Description | Example |
|---|---|---|---|
| `tile_size` | `null` | `[width, height]`; `null` retains each workflow's historical size. | `[320, 240]` |
| `columns` | `null` | Grid columns; `null` retains the workflow layout. | `2` |
| `items_per_grid` | `null` | Maximum images/layers per grid file; extra pages receive `_part-001` names. | `6` |
| `spacing` | `null` | Pixels between tiles; `null` retains workflow spacing. | `8` |
| `padding` | `null` | Outer padding in pixels. | `12` |
| `labels` | `null` | Show labels; `null` keeps workflow behavior. | `false` |
| `background` | `null` | Pillow/Matplotlib color. | `"#101010"` |
| `dpi` | `null` | Output DPI; `null` retains workflow behavior. | `150` |
| `overlay_alpha` | `0.45` | Heatmap opacity from 0 to 1. | `0.8` |
| `cmap` | `viridis` | Matplotlib colormap. | `magma` |
| `grid_format` | `png` | `png`, `pdf`, or `svg`. | `pdf` |
| `normalization` | `per_map` | `per_map`, `shared`, or `fixed`. | `shared` |
| `normalization_range` | `null` | Required `[min, max]` for `fixed`; otherwise must be `null`. | `[0, 1]` |

`per_map` is the historical attention and Grad-CAM behavior. `shared` computes
one range across the run. `fixed` clips to an explicit range.

## Output

```yaml
output:
  directory: ../outputs/custom
  heatmaps: true
  overlays: true
  grids: true
  raw_arrays: false
  image_format: png
  raw_format: npy
  overwrite: replace
```

| Setting | Default | Description | Example |
|---|---|---|---|
| `directory` | required | Output folder. | `../outputs/run-1` |
| `heatmaps` | `true` | Export heatmaps or PCA color maps. | `false` |
| `overlays` | `true` (`false` for PCA) | Export overlays where supported. | `false` |
| `grids` | `true` | Export comparison grids. | `false` |
| `raw_arrays` | `false` | Export analysis arrays without rendering. | `true` |
| `image_format` | `png` | `png`, `jpeg`, `tiff`, or `webp`. | `webp` |
| `raw_format` | `npy` | `npy` or compressed `npz`. | `npz` |
| `overwrite` | `error` | `replace`, `error`, or `skip` existing files. Presets explicitly retain `replace`. | `error` |

At least one output type must be enabled.

Every completed run also writes `run-manifest.json` in the output directory.
It records the fully resolved configuration, model identity and input size,
runtime and package versions, input paths with stable collision-safe IDs and
file metadata, output paths, and UTC start/completion times. Custom
configurations default to `overwrite: error`, so an existing manifest stops the
run before the model is loaded. The compatibility presets explicitly use
`overwrite: replace` to retain their earlier rerun behavior.

## Validate, resolve, and override

Validate without loading a model:

```bash
vision-lens validate --config configs/vit_attention.example.yaml
```

Print final values, expanded inputs, and absolute paths:

```bash
vision-lens resolve --config configs/vit_attention.example.yaml
```

Override any leaf setting from the command line using YAML values:

```bash
vision-lens run --config configs/gradcam.example.yaml \
  --set input.limit=4 \
  --set runtime.batch_size=2 \
  --set analysis.target_class=207 \
  --set visualization.items_per_grid=2 \
  --set output.raw_arrays=true
```
