# Configuration

Vision Lens requires seven YAML sections: `input`, `model`, `preprocessing`,
`analysis`, `runtime`, `visualization`, and `output`. A video run also requires
the `video` section. Every setting in each section must be present, even if its
value is `null` or an empty list. Unknown, missing, and incompatible settings
are rejected before model weights are loaded. Copy an example from `configs/`
to start a new workflow.

All relative paths in YAML files and `--set` overrides are resolved
from the project root (the nearest ancestor containing `pyproject.toml`). The
loader searches upward from the config file first, then from the working
directory. If neither is inside a project, paths use the working directory.
Absolute paths remain unchanged. The path given to `--config` itself is a
normal shell path; this rule applies to paths *inside* the configuration.
CLI `--set` values override the YAML values. The YAML must contain every
setting before overrides are applied. The “Example value” column below shows
common values; the “Parser default” column describes the Python parser's
programmatic defaults, not values that a YAML file may omit.

## Input

```yaml
input:
  files: [examples/1.jpg]
  folders: [photos]
  patterns: ["*.jpg", "*.png"]
  recursive: true
  limit: 100
```

| Setting | Parser default | Description | Example |
|---|---|---|---|
| `files` | `[]` | Explicit image files, kept in the listed order. | `[cat.jpg]` |
| `folders` | `[]` | Folders searched for matching files. | `[photos]` |
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

| Setting | Parser default | Description | Example |
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

| Setting | Parser default | Description | Example |
|---|---|---|---|
| `image_size` | `672` | Square model input width and height for images; longest inference side for video. | `224` |
| `resize` | `stretch` | Image resize mode: `stretch`, `shortest`, `longest`, or `none`. Video always uses aspect-ratio resizing. | `longest` |
| `crop` | `none` | `none` or a `center` crop to `image_size`. | `center` |
| `pad` | `none` | `none` or `center` padding to `image_size`. | `center` |
| `interpolation` | `null` | Model-derived by default; override with `nearest`, `bilinear`, `bicubic`, or `lanczos`. | `bicubic` |
| `normalize` | `true` | Apply channel normalization. | `false` |
| `mean` | `null` | Model-derived RGB means, or three custom values. | `[0.5, 0.5, 0.5]` |
| `std` | `null` | Model-derived positive RGB standard deviations. | `[0.5, 0.5, 0.5]` |

For images, `stretch` preserves the existing no-crop behavior. To retain aspect ratio, use
`longest` with `pad: center`, or `shortest` with `crop: center`. A transformed
image must end at the model's required size. Known fixed models reject invalid
sizes during configuration validation.

## Analysis

All four analyses run through the same command:

```bash
uv run vision-lens run --config configs/gradcam.yaml
```

### Attention and rollout

```yaml
analysis:
  method: attention
  layers: [2, 5, 8, 11]
  heads: null
  head_fusion: mean
```

| Setting | Parser default | Description | Example |
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

| Setting | Parser default | Description | Example |
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
  save_projection: null
```

| Setting | Parser default | Description | Example |
|---|---|---|---|
| `method` | required | Must be `patch_pca`. | `patch_pca` |
| `foreground_threshold` | `0.5` | Normalized first-component cutoff, or `auto` to choose an Otsu split from the fit data. | `auto` |
| `foreground_side` | `high` | Keep the `high` or `low` side. | `low` |
| `projection` | `fit` | Fit a shared projection or `load` one. | `load` |
| `projection_path` | `null` | Saved `.npz` loaded when `projection: load`. | `pca.npz` |
| `save_projection` | `null` | Save the fitted basis and normalization ranges. | `pca.npz` |

A loaded projection reuses its fitted foreground rule and color ranges, so new
images remain in the same PCA color space.
With `foreground_threshold: auto`, PCA fits one threshold from the normalized
first component and stores the resulting number in the projection. Video uses
the representative fit frames, so the threshold stays fixed throughout the
clip. A flat component uses `0.5`. `foreground_side` remains explicit because
PCA cannot identify which side of the split is the subject. An automatic split
is most useful when the first-component values form two distinct groups.
PCA components use the approximate low-rank method. All images selected by one
configuration share a PCA fit, threshold, and foreground side. The image example
selects `examples/5.jpg` and `examples/6.jpg` with `foreground_side: low` to
reproduce the earlier horse colors. To fit another image independently, copy
the YAML and select only that image; this also lets you choose its own
`foreground_threshold` and `foreground_side`. Approximate PCA fitting holds the
selected patch embeddings in memory; large groups or long video fit windows
may need more memory.

## Runtime

```yaml
runtime:
  batch_size: 4
  device: auto
  workers: 2
  precision: float32
  seed: 42
```

| Setting | Parser default | Description | Example |
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

PCA fits one projection and one set of normalization bounds across all images
selected by the configuration. Larger runs stage embeddings one batch at a
time before fitting and rendering. This keeps colors comparable without
holding every source image in memory, though the approximate fit still loads
all selected patch embeddings.
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
  match_input_size: true
  interpolation: mask
  overlay_alpha: 0.6
  overlay_alpha_curve:
    steepness: 10
    midpoint: 0.25
  cmap: magma
  cmap_black:
    threshold: 20
    blend_width: 35
    transparent: true
  grid_format: pdf
  normalization: shared
  normalization_range: null
```

| Setting | Parser default | Description | Example |
|---|---|---|---|
| `tile_size` | `null` | `[width, height]`; `null` retains each workflow's historical size. | `[320, 240]` |
| `columns` | `null` | Grid columns; `null` retains the workflow layout. | `2` |
| `items_per_grid` | `null` | Maximum images/layers per grid file; extra pages receive `_part-001` names. | `6` |
| `spacing` | `null` | Pixels between tiles; `null` retains workflow spacing. | `8` |
| `padding` | `null` | Outer padding in pixels. | `12` |
| `labels` | `null` | Show labels; `null` keeps workflow behavior. | `false` |
| `background` | `null` | Pillow/Matplotlib color. | `"#101010"` |
| `dpi` | `null` | Output DPI; `null` retains workflow behavior. | `150` |
| `match_input_size` | `false` | Resize each standalone visualization to its original input dimensions. Video uses the source frame dimensions when enabled. | `true` |
| `interpolation` | `bilinear` | Visualization upscaling mode: `nearest`, `bilinear`, or `mask`. | `mask` |
| `overlay_alpha` | `0.45` | Heatmap opacity from 0 to 1. | `0.8` |
| `overlay_alpha_curve` | `null` | Optional sigmoid-like, value-dependent overlay opacity. `steepness` must be positive; `midpoint` defaults to `0.5` and moves the transition within the normalized 0–1 range. | `{steepness: 10, midpoint: 0.25}` |
| `cmap` | `viridis` | Matplotlib colormap. | `magma` |
| `cmap_black` | `null` | Optional black start using 0–255 palette positions. `threshold` stays black through that position; `blend_width` controls the linear transition; `transparent` reveals the source beneath pure black in overlays. | `{threshold: 20, blend_width: 35, transparent: true}` |
| `grid_format` | `png` | `png`, `pdf`, or `svg`. | `pdf` |
| `normalization` | `per_map` | `per_map`, `shared`, or `fixed`. | `shared` |
| `normalization_range` | `null` | Required `[min, max]` for `fixed`; otherwise must be `null`. | `[0, 1]` |

`per_map` is the historical attention and Grad-CAM behavior. `shared` computes
one range across the run. `fixed` clips to an explicit range.
`nearest` preserves one constant-color block per attention, Grad-CAM activation,
or PCA patch. `bilinear` smoothly interpolates all maps. For attention, rollout,
and Grad-CAM, `mask` is identical to `bilinear`. For patch PCA, `mask` fits the
RGB PCA basis and scaling bounds from foreground patches, projects every patch
through that foreground basis, bilinearly interpolates those colors, and then
applies the foreground mask with nearest-neighbor upscaling. This avoids blending
foreground colors with black while keeping a sharp foreground boundary.

When `match_input_size` is enabled, image heatmaps, overlays, and patch PCA
images use the source image width and height with the configured visualization
interpolation. Composite grid dimensions remain controlled by the
grid layout and `tile_size`. For video, this setting takes precedence over
`video.output_resolution` and uses codec-compatible even source dimensions.
Disable it to retain the model-sized image outputs and configured video
resolution.
Set `overlay_alpha_curve` to `null` to retain the constant `overlay_alpha`.
When enabled, the normalized map value controls opacity through an
endpoint-normalized sigmoid: the lowest value is exactly transparent, the
highest is exactly opaque, and `overlay_alpha` is ignored. The curve is
centered at `midpoint`; lowering it makes smaller values opaque sooner, while
increasing `steepness` makes the transition sharper. The curve is applied
after the colormap and optional `cmap_black` processing. If
`cmap_black.transparent` is also enabled, its pure-black pixels remain fully
transparent; all other pixels use the sigmoid opacity.
Set `cmap_black` to `null` to use the selected colormap unchanged. With a
threshold of `20` and a blend width of `35`, normalized values through palette
position 20 are pure black, values from 20 to 55 blend linearly from black into
the selected colormap, and values from 55 onward use the original colormap.
The threshold must be from 0 to 254. The blend width must be at least 1 and
cannot extend past position 255.
When `transparent: true`, overlays reveal the original image only where the
final heatmap color is exactly pure black. Near-black pixels in the blend still
use the configured overlay opacity. Standalone heatmaps and heatmap videos
remain RGB outputs and keep those pixels solid black.

## Output

```yaml
output:
  directory: outputs/custom
  heatmaps: true
  overlays: true
  grids: true
  raw_arrays: false
  image_format: png
  raw_format: npy
  overwrite: replace
```

| Setting | Parser default | Description | Example |
|---|---|---|---|
| `directory` | required | Output folder. | `outputs/run-1` |
| `heatmaps` | `true` | Export heatmaps or PCA color maps. | `false` |
| `overlays` | `true` (`false` for PCA) | Export overlays where supported. | `false` |
| `grids` | `true` | Export comparison grids. A single-image PCA run skips its redundant one-tile comparison. | `false` |
| `raw_arrays` | `false` | Export analysis arrays without rendering. | `true` |
| `image_format` | `png` | `png`, `jpeg`, `tiff`, or `webp`. | `webp` |
| `raw_format` | `npy` | `npy` or compressed `npz`. | `npz` |
| `overwrite` | `error` | `replace`, `error`, or `skip` existing files. | `error` |

At least one output type must be enabled.

Every completed image or single-video run also writes `run-manifest.json` in the output directory.
It records the fully resolved configuration, model identity and input size,
runtime and package versions, input paths with stable collision-safe IDs and
file metadata, output paths, and UTC start/completion times. Custom
configurations can set `overwrite: error`, so an existing manifest stops the
run before the model is loaded. The included examples use `overwrite: replace`.

## Video

Install the optional dependencies before running a video configuration:

```bash
# From a clean clone, create the locked runtime environment with video support.
uv sync --locked --no-dev --extra video
```

The presence of a `video` section switches the selected image analysis to
timestamp-sampled frame processing. `input` may select one or more video files;
`runtime.batch_size` remains the maximum number of decoded frames held and
analyzed together.

Video inference preserves each source frame's aspect ratio. The longer side is
scaled to `preprocessing.image_size`; ViT dimensions are then rounded to the
nearest multiples of the model patch size. For example, a 1920×1080 video with
`image_size: 672` and 14×14 patches runs at 672×378. CNNs use the proportional
dimensions directly. Video frames are not cropped or padded, so set
`preprocessing.crop` and `preprocessing.pad` to `none`. Image preprocessing
keeps its configured resize behavior. Spatial maps are rendered against the
decoded source frame, and the manifest records the rectangular model input.

When multiple videos are selected, each runs independently in a subdirectory
of `output.directory` named after its source file. Each subdirectory has its
own outputs and `run-manifest.json`. Automatic sampling rates and fitted PCA
projections are resolved separately for each video.

```yaml
video:
  start_time: 0.0
  end_time: null
  sampling_rate: auto
  frame_limit: null
  output_resolution: [1280, 720]
  pca_fit_frames: 32
  temporal_smoothing: 0.0
  codec: libx264
```

| Setting | Parser default | Description | Example |
|---|---|---|---|
| `start_time` | `0.0` | First source timestamp in seconds. | `2.5` |
| `end_time` | `null` | Exclusive ending timestamp in seconds; `null` reads to the end. | `12.0` |
| `sampling_rate` | `5.0` | Frames sampled per second and output playback FPS. Set `auto` to match the source video's reported average FPS. | `auto` |
| `frame_limit` | `null` | Maximum sampled frames after applying the time range. | `120` |
| `output_resolution` | `null` | Even `[width, height]` for single-view output videos; `null` uses the source size, rounded down to even dimensions when needed. Comparison videos use approximately twice the width. | `[1280, 720]` |
| `pca_fit_frames` | `32` | Maximum evenly distributed representative frames used to fit video PCA. | `64` |
| `temporal_smoothing` | `0.0` | Previous-frame blend strength from `0` (off) to `1` (strongest). | `0.35` |
| `codec` | `libx264` | PyAV/FFmpeg encoder name for MP4 outputs. | `libx264` |

Sampling follows decoded presentation timestamps rather than assuming the
source has a constant frame rate. Output frames receive consecutive timestamps
spaced at exactly `1 / sampling_rate`, making the playback duration explicitly
`sampled_frames / sampling_rate`. The run manifest records both values.
If `sampling_rate` is `auto` and the source has no valid reported FPS, the run
stops with an error; set a numeric FPS for that video. For variable-frame-rate
sources, `auto` uses the reported average FPS and still exports constant-FPS
video.

For attention and rollout, Vision Lens writes one stream per selected
layer/head map. `output.heatmaps`, `output.overlays`, and `output.grids` select
heatmap, overlay, and original/visualization comparison videos. Grad-CAM uses
the same controls; set `analysis.target_class` to an integer to freeze its
target across the clip. Patch PCA uses `output.heatmaps` for its RGB PCA video
and `output.grids` for the side-by-side comparison.

Video PCA fits one projection from representative sampled frames, then freezes
the projection, foreground threshold/side, and RGB normalization bounds before
processing the complete clip. A loaded projection remains frozen in the same
way. Temporal smoothing is applied only when its strength is greater than zero
and operates sequentially across batch boundaries.

Generated videos are silent. Source audio is deliberately omitted rather than
copied or time-stretched; this is recorded in `run-manifest.json`. Raw arrays,
when enabled, are written in bounded per-batch files. NPZ batches include their
sample timestamps.

## Validate, resolve, and override

Validate without loading a model:

```bash
uv run vision-lens validate --config configs/vit_attention.yaml
```

Print final values, expanded inputs, and absolute paths:

```bash
uv run vision-lens resolve --config configs/vit_attention.yaml
```

Override any leaf setting from the command line using YAML values:

```bash
uv run vision-lens run --config configs/gradcam.yaml \
  --set input.limit=4 \
  --set runtime.batch_size=2 \
  --set analysis.target_class=207 \
  --set visualization.items_per_grid=2 \
  --set output.raw_arrays=true
```
