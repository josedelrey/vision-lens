# Configuration reference

Vision Lens uses YAML sections for `input`, `model`, `preprocessing`, `analysis`, `runtime`, `visualization`, and `output`. Adding `video` selects frame-based processing. Omitted settings use the defaults below. Unknown or inapplicable settings are rejected during validation, before model weights are loaded.

```bash
uv run vision-lens validate --config workflow.yaml
uv run vision-lens resolve --config workflow.yaml
uv run vision-lens run --config workflow.yaml
```

`resolve` prints the configuration after defaults, input expansion, and path resolution. `--set section.key=value` can be repeated on any command; values are parsed as YAML. Overrides cannot switch `analysis.method`, PCA projection or foreground modes, or add a `video` section. Put mode changes in a separate YAML file.

Paths inside a configuration, including overridden paths, resolve from the nearest ancestor of the configuration file containing `pyproject.toml`. If none exists, the loader searches upward from the working directory. Otherwise it uses the working directory. The `--config` argument itself follows normal shell path rules.

## Input

At least one existing file must be selected. Explicit files retain their listed order; duplicate paths are removed. Folder patterns are relative globs, and `limit` applies after expansion.

| Setting | Default | Meaning |
|---|---|---|
| `input.files` | `[]` | Image or video file paths. |
| `input.folders` | `[]` | Folders searched for inputs. |
| `input.patterns` | `*.jpg`, `*.jpeg`, `*.png`, `*.webp` | Globs applied within each folder. Set video patterns when selecting videos from folders. |
| `input.recursive` | `false` | Search nested folders. |
| `input.limit` | `null` | Maximum number of selected files. |

Absolute glob patterns and patterns containing `..` are rejected.

## Model

Attention, rollout, and patch PCA require `architecture: vit` with `backend: timm`. Grad-CAM requires `architecture: cnn` with `backend: torchvision`.

| Setting | Default | Meaning |
|---|---|---|
| `model.architecture` | required | `vit` or `cnn`. |
| `model.backend` | required | `timm` or `torchvision`. |
| `model.name` | required | Model identifier understood by the backend. |
| `model.pretrained` | `true` | Load pretrained weights. |
| `model.options` | `null` | Additional backend loader arguments. |

Vision Lens manages `img_size` and `pretrained` for timm, `dynamic_img_size` for video timm models, and `weights` for torchvision. These keys cannot be repeated in `model.options`. A custom model must expose the internals required by the selected method; backend compatibility alone is insufficient.

## Preprocessing

| Setting | Default | Meaning |
|---|---|---|
| `preprocessing.image_size` | `672` | Square input size for images; longest inference side for video. |
| `preprocessing.resize` | `stretch` | Image resize: `stretch`, `shortest`, `longest`, or `none`. |
| `preprocessing.crop` | `none` | `none` or `center` crop to the input size. |
| `preprocessing.pad` | `none` | `none` or `center` padding to the input size. |
| `preprocessing.interpolation` | model-derived | `nearest`, `bilinear`, `bicubic`, or `lanczos`. |
| `preprocessing.normalize` | `true` | Apply RGB channel normalization. |
| `preprocessing.mean` | model-derived | Three RGB means; valid when normalization is enabled. |
| `preprocessing.std` | model-derived | Three positive RGB standard deviations; valid when normalization is enabled. |

For images, `longest` with center padding or `shortest` with center cropping preserves the aspect ratio. Crop and pad cannot both be enabled. Video always preserves the frame aspect ratio; `resize`, `crop`, and `pad` are image-only settings and must be omitted from video configurations. ViT video dimensions are aligned to patch multiples. The known fixed-size DINO ViT-S/8 model requires `image_size: 224`.

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

One image run fits a shared projection, threshold, and RGB range across its selected images, making their colors comparable. With `foreground_separation: false`, omit the other foreground settings; all patches contribute to the RGB fit. With `projection: load`, supply only `projection_path`: the saved foreground rule and color range are reused. Video PCA fits a full-frame projection from representative frames, controlled by `video.pca_fit_frames`; foreground settings are invalid. A saved projection can be the sole output of a PCA run.

PCA uses an approximate low-rank fit over selected patch embeddings. Large image groups or long video fit windows may require substantial memory. PCA component signs do not identify the subject; choose `foreground_side` from the intended selection.

## Runtime

| Setting | Default | Meaning |
|---|---|---|
| `runtime.batch_size` | `8` | Maximum images or decoded frames processed together. |
| `runtime.device` | `auto` | `auto`, `cpu`, `cuda`, or `mps`. |
| `runtime.workers` | `0` | Image-loading threads; omit for video. |
| `runtime.precision` | `float32` | `float32`, `float16`, or `bfloat16`. |
| `runtime.seed` | `null` | Seed Python, NumPy, and PyTorch. |

CPU does not support `float16`; MPS does not support `bfloat16`. Image PCA and `shared` map normalization can require an additional fitting pass. Multi-batch Python results report processed inputs without retaining all analysis tensors.

## Visualization

### Layout and size

The grid settings below apply only to image runs with `output.grids: true`. Patch PCA grids omit labels.

| Setting | Default | Meaning |
|---|---|---|
| `visualization.tile_size` | workflow default | Grid tile `[width, height]`. |
| `visualization.columns` | automatic | Number of grid columns. |
| `visualization.items_per_grid` | all items | Maximum tiles per grid file; further pages use `_part-001` suffixes. |
| `visualization.spacing` | workflow default | Pixels between tiles. |
| `visualization.padding` | workflow default | Outer pixels around the grid. |
| `visualization.labels` | workflow default | Show labels on non-PCA grids. |
| `visualization.background` | workflow default | Grid background color. |
| `visualization.dpi` | workflow default | Grid DPI. |
| `visualization.grid_format` | `png` | `png`, `pdf`, or `svg`. |
| `visualization.output_size` | model-processed size | `match` for source dimensions, a positive square size, or `[width, height]`. |

Video output dimensions must be even. `output_size: match` renders at model resolution, then resizes the completed visualization to source dimensions. Set an explicit size to perform AnyUp interpolation at that resolution.

### Maps and overlays

| Setting | Default | Meaning |
|---|---|---|
| `visualization.interpolation` | `bilinear` | `nearest`, `bilinear`, `bilinear_mask`, `anyup`, `anyup_mask`, `anyup_soft`, or `anyup_soft_mask`. |
| `visualization.anyup_query_chunk_size` | `null` | Positive query count per AnyUp chunk; required for soft modes. |
| `visualization.overlay_alpha` | `0.45` | Overlay opacity from 0 to 1. |
| `visualization.overlay_alpha_curve` | `null` | Value-dependent opacity, e.g. `{steepness: 10, midpoint: 0.25}`. |
| `visualization.cmap` | `viridis` | Matplotlib colormap for rendered maps. |
| `visualization.cmap_black` | `null` | Black low end, e.g. `{threshold: 20, blend_width: 35, transparent: true}`. |
| `visualization.normalization` | `per_map` | `per_map`, `shared` across the run, or `fixed`. |
| `visualization.normalization_range` | `null` | Required `[min, max]` when normalization is `fixed`. |

`nearest` preserves patch or activation blocks; `bilinear` blends between them. For foreground-separated image PCA, `bilinear_mask` blends projected colors but keeps a sharp foreground boundary. Without foreground separation, it behaves like bilinear. The four AnyUp modes use the [official AnyUp model](https://github.com/wimmerth/anyup) to upsample features. Mask variants preserve a coarse PCA foreground boundary; soft variants taper the local attention window and require query chunking. Smaller chunks reduce peak memory at the cost of runtime.

`overlay_alpha_curve` uses `steepness` and an optional `midpoint` (default `0.5`) to vary opacity with normalized map values; `overlay_alpha` scales the result. `cmap_black` uses 0–255 palette positions: `threshold` remains black, `blend_width` transitions into the selected colormap, and `transparent: true` reveals the source where the resulting color is pure black. These controls require corresponding rendered map or overlay outputs. Patch PCA does not accept map colormaps or overlay settings.

Normalization affects rendered maps, overlays, and grids. Raw arrays retain analysis values before visualization normalization.

## Output

| Setting | Default | Meaning |
|---|---|---|
| `output.directory` | required | Output directory. |
| `output.heatmaps` | `true` | Save heatmaps or PCA color maps. |
| `output.overlays` | `true` except PCA | Save attention, rollout, or Grad-CAM overlays. |
| `output.grids` | `true` for images | Save image comparison grids; invalid for video. |
| `output.raw_arrays` | `false` | Save analysis arrays. |
| `output.image_format` | `png` | Standalone image format: `png`, `jpeg`, `tiff`, or `webp`. |
| `output.raw_format` | `npy` | `npy` or compressed `npz`; requires raw arrays. |
| `output.overwrite` | `error` | `error`, `replace`, or `skip` existing outputs. |

At least one output type must be enabled. `output.overlays` is invalid for PCA; `output.grids` and `output.image_format` are image-only settings. `output.overwrite` also applies to the run manifest. The manifest records the resolved configuration, model identity, package versions, input metadata, output paths, and UTC run times. A loaded or saved PCA projection cannot collide with inputs or other planned outputs.

## Video

Install video support with `uv sync --locked --extra video`. A `video` section processes timestamp-sampled frames and exports silent MP4 streams. Source audio is not copied.

| Setting | Default | Meaning |
|---|---|---|
| `video.start_time` | `0.0` | First source timestamp in seconds. |
| `video.end_time` | `null` | Exclusive end timestamp; `null` reads to the end. |
| `video.sampling_rate` | `5.0` | Sampled frames per second and output playback FPS; `auto` uses reported average source FPS. |
| `video.frame_limit` | `null` | Maximum sampled frames in the selected time range. |
| `video.pca_fit_frames` | `32` | Representative frames used to fit video PCA. |
| `video.temporal_smoothing` | `0.0` | Previous-frame blend strength from 0 to 1 for rendered PCA video. |
| `video.codec` | `libx264` | PyAV/FFmpeg encoder for MP4 outputs. |

Each selected video runs independently. With multiple inputs, outputs and `run-manifest.json` are written in separate source-named subdirectories. `sampling_rate: auto` requires a valid reported source FPS; use a number otherwise. Output frames have constant playback FPS even for variable-rate sources. Raw arrays are exported in bounded batches; NPZ batches include sample timestamps.
