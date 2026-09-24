# Configuration reference

Vision Lens accepts YAML files, CLI flags, or both. The sections are `input`,
`model`, `preprocessing`, `analysis`, `runtime`, `visualization`, `output`, and
the optional `video` section. Media types are detected from the selected files.
Unknown, missing, conflicting, and inapplicable settings are rejected before
model weights are loaded.

```bash
vision-lens validate --config workflow.yaml
vision-lens resolve --config workflow.yaml
vision-lens run --config workflow.yaml
```

The YAML file is optional. Every setting has a `--section-key` flag, and every
flag value is parsed as YAML:

```bash
vision-lens validate \
  --input-files '[examples/1.jpg]' \
  --model-architecture vit \
  --model-backend timm \
  --model-name vit_small_patch8_224.dino \
  --preprocessing-image-size 224 \
  --analysis-method attention \
  --analysis-layers '[11]' \
  --output-directory outputs/attention
```

Named flags override values from the YAML file. Both sources go through the same
parser and validator.

Video inputs are detected automatically and processed with the default video
settings. Individual settings can be customized in YAML or with the
corresponding `--video-*` named flags. `validate` reports detected media counts.
`resolve` prints the merged, expanded, defaulted, and path-resolved
configuration. Run `vision-lens --help` for every generated flag.

Relative paths in a YAML file resolve from its nearest ancestor containing
`pyproject.toml`, or from the YAML file's directory when no such ancestor
exists. Without `--config`, relative paths resolve from the nearest ancestor of
the working directory containing `pyproject.toml`, or from the working directory
when none exists. The `--config` path itself follows normal shell rules.

## Workflow compatibility

| Method | Required model | Images | Video |
|---|---|---:|---:|
| `attention` | `architecture: vit`, `backend: timm` | yes | yes |
| `rollout` | `architecture: vit`, `backend: timm` | yes | yes |
| `gradcam` | `architecture: cnn`, `backend: torchvision` | yes | yes |
| `patch_pca` | `architecture: vit`, `backend: timm` | yes | yes |

Image and video settings differ where noted below. Mixed inputs use the union of
the applicable settings, while each branch still receives its own canonical
configuration.

## Input

At least one existing supported media file must be selected. Explicit files
retain their order, duplicate paths are removed, and unsupported explicit file
types are rejected. Folder patterns are relative globs. Unsupported matches are
ignored. `limit` applies after expansion and media filtering.

| Setting | Default | Meaning |
|---|---|---|
| `input.files` | `[]` | Image or video file paths. |
| `input.folders` | `[]` | Folders searched for inputs. |
| `input.patterns` | `*` | Globs applied within each folder before supported-media filtering. |
| `input.recursive` | `false` | Search nested folders. |
| `input.limit` | `null` | Maximum number of selected files. |

`patterns` and `recursive` require at least one folder. Patterns are applied to
each folder independently.

Absolute glob patterns and patterns containing `..` are rejected. Supported
image extensions are `.jpg`, `.jpeg`, `.png`, and `.webp`. Supported video
extensions are `.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, and `.webm`. Matching is
case-insensitive. Pillow and PyAV perform the actual decoding, so corrupt or
mislabelled files still fail.

## Model

Attention, rollout, and patch PCA require `architecture: vit` with
`backend: timm`. Grad-CAM requires `architecture: cnn` with
`backend: torchvision`.

| Setting | Default | Meaning |
|---|---|---|
| `model.architecture` | required | `vit` or `cnn`. |
| `model.backend` | required | `timm` or `torchvision`. |
| `model.name` | required | Model identifier understood by the backend. |
| `model.pretrained` | `true` | Load pretrained weights. |
| `model.options` | `null` | Additional backend loader arguments. |

Vision Lens manages `img_size` and `pretrained` for timm,
`dynamic_img_size` for timm video models, and `weights` for torchvision. Do not
repeat those keys in `model.options`. Custom models must expose the internals
required by the selected method. Backend compatibility alone is insufficient.

## Preprocessing

| Setting | Default | Meaning |
|---|---|---|
| `preprocessing.image_size` | `672` | Square input size for images and longest inference side for video. |
| `preprocessing.resize` | `stretch` | Image resize: `stretch`, `shortest`, `longest`, or `none`. |
| `preprocessing.crop` | `none` | `none` or `center` crop to the input size. |
| `preprocessing.pad` | `none` | `none` or `center` padding to the input size. |
| `preprocessing.interpolation` | model-derived | `nearest`, `bilinear`, `bicubic`, or `lanczos`. |
| `preprocessing.normalize` | `true` | Apply RGB channel normalization. |
| `preprocessing.mean` | model-derived | Three RGB means. Valid when normalization is enabled. |
| `preprocessing.std` | model-derived | Three positive RGB standard deviations. Valid when normalization is enabled. |

For images, `longest` with center padding or `shortest` with center cropping
preserves aspect ratio. Crop and pad cannot both be enabled. Video always
preserves frame aspect ratio. `resize`, `crop`, and `pad` are image-only and
must be omitted from video-only configurations. ViT video dimensions are
aligned to patch multiples. The known fixed-size DINO ViT-S/8 model requires
`image_size: 224`.

## Analysis

### Attention and rollout

```yaml
analysis:
  method: attention
  layers: [2, 5, 8, 11]
```

| Setting | Default | Meaning |
|---|---|---|
| `analysis.method` | `attention` | `attention` or `rollout`. |
| `analysis.layers` | required | Layer indices or `all`. |
| `analysis.heads` | all heads | Nonempty list of head indices. For rollout, affects image comparison grids only. |
| `analysis.head_fusion` | `mean` | `mean`, `max`, or `none`. For rollout, affects image comparison grids only. |

`mean` and `max` fuse the selected heads into one map. `none` emits one map per
selected head. When `heads` is omitted, all heads participate.

For rollout image grids, choose whether to show rollout maps alone or compare
them with direct layer attention:

```yaml
visualization:
  rollout_grid: rollout
```

`visualization.rollout_grid` accepts `comparison` (the default) or `rollout`.
The `analysis.heads` and `analysis.head_fusion` settings apply to rollout only
when `rollout_grid: comparison`, because they control the direct-attention row.

### Grad-CAM

```yaml
analysis:
  method: gradcam
  target_layer: layer4
```

| Setting | Default | Meaning |
|---|---|---|
| `analysis.method` | required | `gradcam`. |
| `analysis.target_layer` | last convolution | Module path used for activation capture. |
| `analysis.target_class` | predicted class | Fixed nonnegative class index. Set this for a consistent target across video frames. |

### Patch PCA

```yaml
analysis:
  method: patch_pca
  foreground_side: low
```

| Setting | Default | Meaning |
|---|---|---|
| `analysis.method` | required | `patch_pca`. |
| `analysis.foreground_separation` | `true` for images | Use the first component to select image patches. Video always uses the full frame. |
| `analysis.foreground_threshold` | `0.5` | Normalized PC1 cutoff or `auto` for an Otsu split. Image foreground mode only. |
| `analysis.foreground_side` | `high` | Keep values above or below the cutoff. Image foreground mode only. |
| `analysis.rgb_fit_scope` | `foreground` | Fit RGB PCA from `foreground` or `all` image patches. Image foreground mode only. |
| `analysis.projection` | `fit` | Fit a projection or `load` a saved one. |
| `analysis.projection_path` | `null` | `.npz` file required with `projection: load`. |
| `analysis.save_projection` | `null` | Save a fitted projection to `.npz`. |

One image run fits a shared projection, threshold, and RGB range across all its
images, making colors comparable. With `foreground_separation: false`, omit the
other foreground settings. All patches contribute. With `projection: load`,
supply only `projection_path`. The saved foreground rule and color range are
reused. Video PCA fits a full-frame projection from representative frames
controlled by `video.pca_fit_frames`, so foreground settings are invalid for
video. Saving a projection may be the sole output of a PCA run.

Loaded projections must match the model's patch-embedding feature count. PCA
uses an approximate low-rank fit over selected embeddings. Large image groups or
long video fit windows may require substantial memory. The subject may appear on
either side of the first PCA component. Use `foreground_side: high` to keep
values above the threshold or `foreground_side: low` to keep values below it.

## Runtime

| Setting | Default | Meaning |
|---|---|---|
| `runtime.batch_size` | `8` | Maximum images or decoded frames processed together. |
| `runtime.device` | `auto` | `auto`, `cpu`, `cuda`, or `mps`. |
| `runtime.workers` | `0` | Image-loading threads. Omit for video. |
| `runtime.precision` | `float32` | `float32`, `float16`, or `bfloat16`. |
| `runtime.seed` | `null` | Seed Python, NumPy, and PyTorch. |

`auto` selects CUDA, then MPS, then CPU. CPU does not support `float16`. MPS
does not support `bfloat16`. Image PCA and `shared` normalization can require an
additional fitting pass. Multi-batch Python results report processed inputs
without retaining all analysis tensors.

If a run exhausts device memory, reduce `runtime.batch_size` from its default of
`8`. AnyUp can require substantially more memory. For an AnyUp out-of-memory
error, try `visualization.anyup_query_chunk_size: 4096` and lower it further if
needed.

## Visualization

### Layout and size

The grid settings below apply only to image runs with `output.grids: true`. Patch PCA grids omit labels.

| Setting | Default | Meaning |
|---|---|---|
| `visualization.tile_size` | workflow default | Grid tile `[width, height]`. |
| `visualization.columns` | automatic | Number of grid columns. |
| `visualization.items_per_grid` | all items | Maximum tiles per grid file. Further pages use `_part-001` suffixes. |
| `visualization.spacing` | workflow default | Pixels between tiles. |
| `visualization.padding` | workflow default | Outer pixels around the grid. |
| `visualization.labels` | workflow default | Show labels on non-PCA grids. |
| `visualization.background` | workflow default | Grid background color. |
| `visualization.dpi` | workflow default | Grid DPI. |
| `visualization.grid_format` | `png` | `png`, `pdf`, or `svg`. |
| `visualization.rollout_grid` | `comparison` | For rollout image grids, `comparison` includes direct attention or `rollout` shows rollout maps only. |
| `visualization.output_size` | model-processed size | `match` for source dimensions, a positive square size, or `[width, height]`. |

Video output dimensions must be even. For video, `output_size: match` analyzes
at model resolution and resizes the completed visualization to the source
dimensions. Set an explicit size to interpolate directly at that resolution.

### Maps and overlays

| Setting | Default | Meaning |
|---|---|---|
| `visualization.interpolation` | `bilinear` | `nearest`, `bilinear`, `bilinear_mask`, `anyup`, `anyup_mask`, `anyup_soft`, or `anyup_soft_mask`. |
| `visualization.anyup_query_chunk_size` | `null` | Positive query count per AnyUp chunk. Required for soft modes. |
| `visualization.overlay_alpha` | `0.45` | Overlay opacity from 0 to 1. |
| `visualization.overlay_alpha_curve` | `null` | Value-dependent opacity, e.g. `{steepness: 10, midpoint: 0.25}`. |
| `visualization.cmap` | `viridis` | Matplotlib colormap for rendered maps. |
| `visualization.cmap_black` | `null` | Black low end, e.g. `{threshold: 20, blend_width: 35, transparent: true}`. |
| `visualization.normalization` | `per_map` | `per_map`, `shared` across the run, or `fixed`. |
| `visualization.normalization_range` | `null` | Required `[min, max]` when normalization is `fixed`. |

`nearest` preserves patch or activation blocks. `bilinear` blends between them.
For foreground-separated image PCA, `bilinear_mask` blends projected colors but
keeps a sharp foreground boundary. Without foreground separation it behaves
like bilinear. The AnyUp modes load model code from a pinned revision of the
[official repository](https://github.com/wimmerth/anyup) and download its
pretrained checkpoint. Mask variants preserve a coarse PCA boundary. Soft
variants taper the local attention window and require query chunking. Smaller
chunks trade speed for lower peak memory.

`overlay_alpha_curve` uses `steepness` and an optional `midpoint` (default
`0.5`) to vary opacity with normalized map values. `overlay_alpha` scales the
result. It applies when overlays, transparent overlays, or grids are enabled.
`cmap_black` uses 0–255 palette positions: `threshold` remains black,
`blend_width` transitions into the selected colormap, and `transparent: true`
reveals the source where the resulting color is pure black. It applies when
heatmaps, overlays, transparent overlays, or grids are enabled. Patch PCA does
not accept map colormaps or overlay settings.

Normalization affects rendered maps, overlays, and grids. Raw arrays retain
analysis values before visualization normalization.

## Output

| Setting | Default | Meaning |
|---|---|---|
| `output.directory` | required | Output directory. |
| `output.heatmaps` | `true` | Save heatmaps or PCA color maps. |
| `output.overlays` | `true` except PCA | Save attention, rollout, or Grad-CAM overlays. |
| `output.transparent_overlays` | `false` | Save standalone attention, rollout, or Grad-CAM layers with transparency. Images are RGBA PNGs. Videos use `video.alpha_format`. |
| `output.grids` | `true` for images | Save image comparison grids. Invalid for video. |
| `output.raw_arrays` | `false` | Save analysis arrays. |
| `output.image_format` | `png` | Image heatmap and flattened-overlay format: `png`, `jpeg`, `tiff`, or `webp`. |
| `output.raw_format` | `npy` | `npy` or compressed `npz`. Requires raw arrays. |
| `output.overwrite` | `error` | `error`, `replace`, or `skip` existing outputs. |

At least one output type must be enabled. `overlays` writes a flattened
source-plus-map image. `transparent_overlays` writes the map layer without
source pixels. Both are invalid for PCA. Transparent image overlays are always
PNG, and grids use `visualization.grid_format`. `output.grids` and
`output.image_format` are image-only.

With `overwrite: error`, the run fails during preflight if any planned artifact
already exists. `replace` removes the previous managed `images/` and `videos/`
trees before writing a pristine set of outputs. It refuses to remove
an unrecognized root manifest or unrecognized contents from those reserved
paths. Unrelated files directly under `output.directory` are preserved. `skip`
preserves existing artifacts and writes missing ones. Input files, loaded PCA
projections, saved projections, and planned outputs are checked for path
collisions before execution.

Each completed run writes an aggregate `run-manifest.json` directly under
`output.directory`. Image and per-video directories also contain manifests with
branch-specific processing details. Manifests record the resolved configuration,
model identity, package versions, input metadata, output paths, and UTC run
times.

## Video

Install video support with `uv sync --locked --extra video`. Videos are sampled
by timestamp and exported as silent MP4 streams, plus optional alpha-capable MOV
or WebM streams. Source audio is not copied. Video settings override the defaults
below and are valid only when at least one video is selected.

| Setting | Default | Meaning |
|---|---|---|
| `video.start_time` | `0.0` | First source timestamp in seconds. |
| `video.end_time` | `null` | Exclusive end timestamp. `null` reads to the end. |
| `video.sampling_rate` | `5.0` | Sampled frames per second and output playback FPS. `auto` uses reported average source FPS. |
| `video.frame_limit` | `null` | Maximum sampled frames in the selected time range. |
| `video.pca_fit_frames` | `32` | Representative frames used to fit video PCA. |
| `video.temporal_smoothing` | `0.0` | Previous-frame blend strength from 0 to 1 for consecutive rendered maps. |
| `video.codec` | `libx264` | PyAV/FFmpeg encoder for MP4 outputs. |
| `video.alpha_format` | `prores_4444` | Transparent-overlay encoding. `prores_4444` produces a `.mov`, while `vp9` produces a `.webm`. Only applicable when `output.transparent_overlays: true`. |

Each video runs independently in a source-named subdirectory.
`sampling_rate: auto` requires a valid reported average source FPS. Use a number
otherwise. Output streams have a constant playback rate even for variable-rate
sources. Raw arrays are exported in bounded batches. NPZ batches include sample
timestamps.

ProRes 4444 is the default transparent-video format and targets editing
workflows. VP9 alpha produces smaller WebM files, but alpha playback support
varies by application. `video.codec` controls only ordinary heatmap and
flattened-overlay MP4 files. Video dependencies and requested encoders are
checked before a mixed run writes image outputs.
