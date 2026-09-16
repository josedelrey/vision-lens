# Configuration

Vision Lens configurations use `input`, `model`, `preprocessing`, `analysis`,
`runtime`, `visualization`, and `output` sections. The presence of a `video`
section selects a video workflow. Settings with parser defaults may be omitted;
the resolved manifest records the applicable defaults actually used. Unknown,
missing required, incompatible, and workflow-inapplicable settings are rejected
before model weights are loaded. Copy an example from `configs/` to start a new
workflow.

All relative paths in YAML files and `--set` overrides are resolved
from the project root (the nearest ancestor containing `pyproject.toml`). The
loader searches upward from the config file first, then from the working
directory. If neither is inside a project, paths use the working directory.
Absolute paths remain unchanged. The path given to `--config` itself is a
normal shell path; this rule applies to paths *inside* the configuration.
CLI `--set` values override the YAML values. The “Example value” column below
shows common values; the “Parser default” column is the value used when that
setting is omitted.

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

At least one file must be selected. Duplicate paths are removed.

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

Loader arguments managed by Vision Lens cannot be repeated in `model.options`:
`img_size` and `pretrained` for timm, and `weights` for torchvision.
`preprocessing.image_size` is the authoritative input size.

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

`mean` and `std` apply only when `normalize: true`; omit them when normalization
is disabled. All numeric preprocessing values must be finite.

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
| `heads` | `null` | Head indices; `null` selects all. For rollout, applies only to image comparison grids. | `[0, 1]` |
| `head_fusion` | `mean` | `mean`, `max`, or `none`. For rollout, applies only to image comparison grids. | `none` |

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
  foreground_separation: true
  foreground_threshold: 0.5
  foreground_side: low
  rgb_fit_scope: foreground
  projection: fit
  save_projection: null
```

| Setting | Parser default | Description | Example |
|---|---|---|---|
| `method` | required | Must be `patch_pca`. | `patch_pca` |
| `foreground_separation` | `true` | For images, use PC1 to mask one side of the patch distribution. Set to `false` to render every patch and fit RGB PCA from the full image set. Video PCA always uses full-frame mode. | `false` |
| `foreground_threshold` | `0.5` | Normalized first-component cutoff, or `auto` to choose an Otsu split from the fit data. | `auto` |
| `foreground_side` | `high` | Keep the `high` or `low` side. | `low` |
| `rgb_fit_scope` | `foreground` | With foreground separation enabled, fit the RGB PCA basis and bounds from foreground patches or from `all` patches. Full-frame mode always uses `all`. | `all` |
| `projection` | `fit` | Fit a shared projection or `load` one. | `load` |
| `projection_path` | `null` | Saved `.npz` loaded when `projection: load`. | `pca.npz` |
| `save_projection` | `null` | Save the fitted basis and normalization ranges. | `pca.npz` |

The four foreground/RGB-scope settings are image-only and are omitted from
video configurations and resolved video manifests. They are not valid video
PCA settings.

`projection: fit` accepts the image foreground settings and optional
`save_projection`; `projection_path` is invalid in this mode. `projection:
load` accepts only `projection_path` and reuses all fitted settings, so
foreground settings and `save_projection` are invalid. Saving a fitted
projection counts as an output, allowing a projection-only run with every
media and raw-array output disabled.

A loaded projection reuses its fitted foreground mode, rule, RGB fit scope, and
color ranges, so new images remain in the same PCA color space. For images,
`foreground_separation: false` skips the PC1 split, renders every patch, and
uses the full patch set for the RGB fit. Omit `foreground_threshold`,
`foreground_side`, and `rgb_fit_scope` in this mode because they are invalid.
With `foreground_threshold: auto`, image PCA fits one threshold from the
normalized first component and stores the resulting number in the projection.
A flat component uses `0.5`. `foreground_side` remains explicit because
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
| `workers` | `0` | Threads used to read images; image workflows only. `0` reads sequentially. | `4` |
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
  output_size: match
  interpolation: bilinear_mask
  anyup_query_chunk_size: null
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
| `columns` | `null` | Grid columns; `null` chooses a balanced, near-square layout. | `2` |
| `items_per_grid` | `null` | Maximum images/layers per grid file across the complete comparison; extra pages receive `_part-001` names. Inference batch boundaries never create parts. | `6` |
| `spacing` | `null` | Pixels between tiles; `null` retains workflow spacing. | `8` |
| `padding` | `null` | Outer padding in pixels. | `12` |
| `labels` | `null` | Show labels on non-PCA image grids; patch PCA grids deliberately contain no text. | `false` |
| `background` | `null` | Pillow/Matplotlib color. | `"#101010"` |
| `dpi` | `null` | Output DPI; `null` retains workflow behavior. | `150` |
| `output_size` | `null` | Standalone visualization size: `null` keeps the model-processed size, `match` uses each source's dimensions, a positive integer produces a square output, and `[width, height]` sets an exact size. Video dimensions must be even. | `[1280, 720]` |
| `interpolation` | `bilinear` | Visualization upscaling mode: `nearest`, `bilinear`, `bilinear_mask`, `anyup`, `anyup_mask`, `anyup_soft`, or `anyup_soft_mask`. | `anyup_soft` |
| `anyup_query_chunk_size` | `null` | Positive number of output queries processed per AnyUp attention chunk. Smaller values lower peak memory but increase runtime; `null` disables chunking. Requires an AnyUp mode and is mandatory for both soft AnyUp modes. | `4096` |
| `overlay_alpha` | `0.45` | Uniform heatmap opacity from 0 to 1. When an alpha curve is enabled, this scales the curve's per-pixel opacity. | `0.8` |
| `overlay_alpha_curve` | `null` | Optional sigmoid-like, value-dependent overlay opacity applied before `overlay_alpha`. `steepness` must be positive; `midpoint` defaults to `0.5` and moves the transition within the normalized 0–1 range. | `{steepness: 10, midpoint: 0.25}` |
| `cmap` | `viridis` | Matplotlib colormap. | `magma` |
| `cmap_black` | `null` | Optional black start using 0–255 palette positions. `threshold` stays black through that position; `blend_width` controls the linear transition; `transparent` reveals the source beneath pure black in overlays. | `{threshold: 20, blend_width: 35, transparent: true}` |
| `grid_format` | `png` | `png`, `pdf`, or `svg`. | `pdf` |
| `normalization` | `per_map` | `per_map`, `shared`, or `fixed`. | `shared` |
| `normalization_range` | `null` | Required `[min, max]` for `fixed`; otherwise must be `null`. | `[0, 1]` |

Grid layout settings are valid only for image runs with `output.grids: true`;
video configurations reject them. Patch PCA grids always omit labels and reject
map colormap, overlay, and normalization settings. Overlay styling requires an
overlay or comparison-grid output. Colormap settings require rendered map
output.

`spacing` controls the gaps between tiles, while `padding` controls the outer
border around the complete grid. Set either to `0` to remove it. Part-numbered
files are only produced when the number of comparison items exceeds
`items_per_grid`; leave `items_per_grid: null` to write one grid file.

`per_map` is the historical attention and Grad-CAM behavior. `shared` computes
one range across the run. `fixed` clips to an explicit range.
These normalization modes affect rendered heatmaps, overlays, and grids only.
Raw arrays always contain the extracted, interpolated analysis values before
visualization normalization.
`nearest` preserves one constant-color block per attention, Grad-CAM activation,
or PCA patch. `bilinear` smoothly interpolates all maps. For attention, rollout,
and Grad-CAM, `bilinear_mask` is identical to `bilinear`. For image patch PCA
with foreground separation enabled, `bilinear_mask` fits the RGB PCA basis and
scaling bounds from the configured scope, projects every patch through that
basis, bilinearly interpolates those colors, and then applies the foreground
mask with nearest-neighbor upscaling. This avoids blending foreground colors
with black while keeping a sharp foreground boundary. Without foreground
separation—including every video run—it behaves like full-frame `bilinear`.

The four AnyUp modes replace bilinear feature-map upscaling with the
[official AnyUp model](https://github.com/wimmerth/anyup). For attention,
rollout, and Grad-CAM, mask-suffixed and non-mask modes are identical. For patch
PCA, the PCA directions, foreground threshold, and projection bounds are fitted
from the original patch embeddings. `anyup` and `anyup_soft` apply the fitted PC1
projection and threshold to every dense AnyUp output vector, producing a binary
foreground mask at the final pixel resolution. `anyup_mask` and
`anyup_soft_mask` instead resize the original coarse binary foreground mask with
nearest-neighbor interpolation. The soft variants use a Vision Lens-specific
cosine taper at the local attention-window boundary instead of AnyUp's hard
boolean cutoff. For all four modes, exported foreground-mask arrays have the
final visualization resolution. The pretrained AnyUp weights are unchanged.

The first AnyUp run downloads the official multi-backbone implementation and
checkpoint from its pinned `checkpoint_v2` release through PyTorch Hub; later
runs use PyTorch's local cache. Vision Lens uses the original attention-based
AnyUp implementation, so NATTEN is not required. The reusable adapter is
available as `vision_lens.anyup`, including the model loader, ImageNet
guidance-image preparation, and generic `upsample_features` function.

Set `anyup_query_chunk_size` when a full AnyUp attention operation does not fit
in VRAM. Chunking preserves the requested output resolution and attention
calculation while processing fewer output queries at once. The included video
configurations use conservative chunk sizes; lower them further if memory is
still exhausted, or raise them for better throughput when more VRAM is
available. For patch PCA, the configured chunking path projects the
low-resolution values to RGB before AnyUp,
generates query features and locality masks one chunk at a time, and transfers
finished RGB chunks away from the execution device. This avoids materializing
both the global attention mask and a full-resolution embedding tensor. The
original embeddings are still used to produce AnyUp's keys, so this reorders a
linear projection without changing the intended attention calculation.

`output_size: match` makes image heatmaps, overlays, and patch PCA images use
the dimensions of each source image. For video it uses codec-compatible even
source dimensions. This mode intentionally performs analysis interpolation,
including AnyUp, at the model-processed resolution and then resizes the finished
visualization to the source dimensions (nearest for `nearest` heatmaps and
bilinear otherwise). To run AnyUp at a particular final resolution, specify an
explicit integer or `[width, height]`; choosing a smaller size also reduces its
memory use. `null` retains the model-processed dimensions. Composite grid
dimensions remain controlled by the grid layout and `tile_size`.
Set `overlay_alpha_curve` to `null` to use the constant `overlay_alpha` alone.
When the curve is enabled, the normalized map value first controls per-pixel
opacity through an endpoint-normalized sigmoid. `overlay_alpha` then uniformly
scales that opacity, so the lowest map value remains exactly transparent and
the highest has opacity `overlay_alpha`. The curve is centered at `midpoint`;
lowering it makes smaller values opaque sooner, while increasing `steepness`
makes the transition sharper. The curve is applied after the colormap and
optional `cmap_black` processing. If
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
| `overlays` | `true` | Export attention, rollout, or Grad-CAM overlays. Omit for patch PCA. | `false` |
| `grids` | `true` | Export image comparison grids. Image configurations only; omit this setting from video configurations. A single-image PCA run skips its redundant one-tile comparison. | `false` |
| `raw_arrays` | `false` | Export analysis arrays without rendering. Video PCA exports the normalized full-frame RGB patch projection, not a foreground mask. | `true` |
| `image_format` | `png` | `png`, `jpeg`, `tiff`, or `webp`; standalone image outputs only. | `webp` |
| `raw_format` | `npy` | `npy` or compressed `npz`; valid only when `raw_arrays: true`. | `npz` |
| `overwrite` | `error` | `replace`, `error`, or `skip` existing files. | `error` |

At least one output type must be enabled. Video configurations do not support
comparison grids, so `output.grids` is rejected when a `video` section is
present.

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
dimensions directly. Video frames are not cropped or padded, so `resize`,
`crop`, and `pad` are image-only settings and must be omitted from
video configurations. Image preprocessing keeps its configured resize
behavior. Spatial maps are rendered at
`visualization.output_size`; the manifest records both the rectangular model
input and resolved output resolution.

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
| `pca_fit_frames` | `32` | Maximum evenly distributed representative frames used to fit video PCA. | `64` |
| `temporal_smoothing` | `0.0` | Previous-frame blend strength from `0` (off) to `1` (strongest). | `0.35` |
| `codec` | `libx264` | PyAV/FFmpeg encoder name for MP4 outputs. | `libx264` |

`pca_fit_frames` applies only when fitting a video patch-PCA projection.
`codec` applies only when a rendered video output is enabled. For patch PCA,
`temporal_smoothing` applies only to the rendered PCA video. Video decoding is
sequential, so the image-only `runtime.workers` setting must be omitted.

Sampling follows decoded presentation timestamps rather than assuming the
source has a constant frame rate. Output frames receive consecutive timestamps
spaced at exactly `1 / sampling_rate`, making the playback duration explicitly
`sampled_frames / sampling_rate`. The run manifest records both values.
If `sampling_rate` is `auto` and the source has no valid reported FPS, the run
stops with an error; set a numeric FPS for that video. For variable-frame-rate
sources, `auto` uses the reported average FPS and still exports constant-FPS
video.

For attention and rollout, Vision Lens writes one heatmap stream and one overlay
stream per selected layer/head map when `output.heatmaps` and `output.overlays`
are enabled. Grad-CAM uses the same controls; set `analysis.target_class` to an
integer to freeze its target across the clip. Patch PCA uses `output.heatmaps`
for its RGB PCA video.

Video PCA is deliberately a full-frame feature visualization, not a foreground
segmenter. It fits one clip-global RGB PCA basis from every patch in evenly
distributed representative frames, fixes its color normalization to the 1st and
99th percentiles of those projected patches, and reuses that projection for the
complete clip. This keeps colors comparable across frames while avoiding the
unreliable assumption that PC1 separates subject from background. The colors
show dominant DINO patch-feature variation; they are not object classes,
attention, or a semantic mask. A loaded RGB projection remains frozen, and
video rendering still covers the full frame. Temporal smoothing is applied only
when its strength is greater than zero and operates sequentially across batch
boundaries.

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

Override leaf values within the configuration's existing workflow mode using
YAML values:

```bash
uv run vision-lens run --config configs/gradcam.yaml \
  --set input.limit=4 \
  --set runtime.batch_size=2 \
  --set analysis.target_class=207 \
  --set visualization.items_per_grid=2 \
  --set output.raw_arrays=true
```

Conditional mode settings (`analysis.method`, patch-PCA `projection` and
`foreground_separation`, and the presence of the `video` section) cannot be
switched with `--set`. Use a separate configuration file so mode-specific keys
remain explicit and auditable.
