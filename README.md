# Vision Lens

[![CI](https://github.com/josedelrey/vision-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/josedelrey/vision-lens/actions/workflows/ci.yml)

Vision Lens produces visualizations of pretrained vision models for images and sampled video frames. It supports transformer attention, attention rollout, Grad-CAM, and principal component analysis (PCA) of patch features. Runs are defined in YAML and record their resolved settings in a manifest.

## Results

<table border="0" cellspacing="0" style="border: 0; border-collapse: collapse;">
  <thead>
    <tr style="background: transparent; border: 0;">
      <th align="center" style="border: 0;">Interpolation</th>
      <th style="border: 0;"><div align="center">Horse 5</div></th>
      <th style="border: 0;"><div align="center">Horse 6</div></th>
    </tr>
  </thead>
  <tbody>
    <tr style="background: transparent; border: 0;">
      <th align="center" style="border: 0;">Original</th>
      <td align="center" style="border: 0;"><img src="examples/5.jpg" alt="Original horse 5" width="320"></td>
      <td align="center" style="border: 0;"><img src="examples/6.jpg" alt="Original horse 6" width="320"></td>
    </tr>
    <tr style="background: transparent; border: 0;">
      <th align="center" style="border: 0;">Bilinear</th>
      <td align="center" style="border: 0;"><img src="assets/pca/bilinear-5.png" alt="Horse 5 PCA with bilinear interpolation" width="320"></td>
      <td align="center" style="border: 0;"><img src="assets/pca/bilinear-6.png" alt="Horse 6 PCA with bilinear interpolation" width="320"></td>
    </tr>
    <tr style="background: transparent; border: 0;">
      <th align="center" style="border: 0;">Bilinear mask</th>
      <td align="center" style="border: 0;"><img src="assets/pca/bilinear-mask-5.png" alt="Horse 5 PCA with bilinear mask interpolation" width="320"></td>
      <td align="center" style="border: 0;"><img src="assets/pca/bilinear-mask-6.png" alt="Horse 6 PCA with bilinear mask interpolation" width="320"></td>
    </tr>
    <tr style="background: transparent; border: 0;">
      <th align="center" style="border: 0;">Nearest</th>
      <td align="center" style="border: 0;"><img src="assets/pca/nearest-5.png" alt="Horse 5 PCA with nearest-neighbor interpolation" width="320"></td>
      <td align="center" style="border: 0;"><img src="assets/pca/nearest-6.png" alt="Horse 6 PCA with nearest-neighbor interpolation" width="320"></td>
    </tr>
    <tr style="background: transparent; border: 0;">
      <th align="center" style="border: 0;">AnyUp soft</th>
      <td align="center" style="border: 0;"><img src="assets/pca/anyup-soft-5.png" alt="Horse 5 PCA with AnyUp soft interpolation" width="320"></td>
      <td align="center" style="border: 0;"><img src="assets/pca/anyup-soft-6.png" alt="Horse 6 PCA with AnyUp soft interpolation" width="320"></td>
    </tr>
    <tr style="background: transparent; border: 0;">
      <th align="center" style="border: 0;">AnyUp soft mask</th>
      <td align="center" style="border: 0;"><img src="assets/pca/anyup-soft-mask-5.png" alt="Horse 5 PCA with AnyUp soft mask interpolation" width="320"></td>
      <td align="center" style="border: 0;"><img src="assets/pca/anyup-soft-mask-6.png" alt="Horse 6 PCA with AnyUp soft mask interpolation" width="320"></td>
    </tr>
  </tbody>
</table>

**Figure 1.** Patch PCA of two horse images across the selected interpolation modes.

<table border="0" cellspacing="0" style="border: 0; border-collapse: collapse;">
  <thead>
    <tr style="background: transparent; border: 0;">
      <th style="border: 0;"><div align="center">Example</div></th>
      <th style="border: 0;"><div align="center">Original</div></th>
      <th style="border: 0;"><div align="center">Layer 2</div></th>
      <th style="border: 0;"><div align="center">Layer 5</div></th>
      <th style="border: 0;"><div align="center">Layer 8</div></th>
      <th style="border: 0;"><div align="center">Layer 11</div></th>
    </tr>
  </thead>
  <tbody>
    <tr style="background: transparent; border: 0;">
      <th style="border: 0;"><div align="center">1</div></th>
      <td align="center" style="border: 0;"><img src="examples/1.jpg" alt="Original example 1" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-1-layer-2.png" alt="Example 1 attention heatmap at layer 2" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-1-layer-5.png" alt="Example 1 attention heatmap at layer 5" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-1-layer-8.png" alt="Example 1 attention heatmap at layer 8" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-1-layer-11.png" alt="Example 1 attention heatmap at layer 11" width="160"></td>
    </tr>
    <tr style="background: transparent; border: 0;">
      <th style="border: 0;"><div align="center">2</div></th>
      <td align="center" style="border: 0;"><img src="examples/2.jpg" alt="Original example 2" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-2-layer-2.png" alt="Example 2 attention heatmap at layer 2" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-2-layer-5.png" alt="Example 2 attention heatmap at layer 5" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-2-layer-8.png" alt="Example 2 attention heatmap at layer 8" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-2-layer-11.png" alt="Example 2 attention heatmap at layer 11" width="160"></td>
    </tr>
    <tr style="background: transparent; border: 0;">
      <th style="border: 0;"><div align="center">3</div></th>
      <td align="center" style="border: 0;"><img src="examples/3.jpg" alt="Original example 3" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-3-layer-2.png" alt="Example 3 attention heatmap at layer 2" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-3-layer-5.png" alt="Example 3 attention heatmap at layer 5" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-3-layer-8.png" alt="Example 3 attention heatmap at layer 8" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-3-layer-11.png" alt="Example 3 attention heatmap at layer 11" width="160"></td>
    </tr>
    <tr style="background: transparent; border: 0;">
      <th style="border: 0;"><div align="center">4</div></th>
      <td align="center" style="border: 0;"><img src="examples/4.jpg" alt="Original example 4" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-4-layer-2.png" alt="Example 4 attention heatmap at layer 2" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-4-layer-5.png" alt="Example 4 attention heatmap at layer 5" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-4-layer-8.png" alt="Example 4 attention heatmap at layer 8" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-4-layer-11.png" alt="Example 4 attention heatmap at layer 11" width="160"></td>
    </tr>
    <tr style="background: transparent; border: 0;">
      <th style="border: 0;"><div align="center">5</div></th>
      <td align="center" style="border: 0;"><img src="examples/5.jpg" alt="Original example 5" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-5-layer-2.png" alt="Example 5 attention heatmap at layer 2" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-5-layer-5.png" alt="Example 5 attention heatmap at layer 5" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-5-layer-8.png" alt="Example 5 attention heatmap at layer 8" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-5-layer-11.png" alt="Example 5 attention heatmap at layer 11" width="160"></td>
    </tr>
    <tr style="background: transparent; border: 0;">
      <th style="border: 0;"><div align="center">6</div></th>
      <td align="center" style="border: 0;"><img src="examples/6.jpg" alt="Original example 6" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-6-layer-2.png" alt="Example 6 attention heatmap at layer 2" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-6-layer-5.png" alt="Example 6 attention heatmap at layer 5" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-6-layer-8.png" alt="Example 6 attention heatmap at layer 8" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-6-layer-11.png" alt="Example 6 attention heatmap at layer 11" width="160"></td>
    </tr>
    <tr style="background: transparent; border: 0;">
      <th style="border: 0;"><div align="center">7</div></th>
      <td align="center" style="border: 0;"><img src="examples/7.jpg" alt="Original example 7" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-7-layer-2.png" alt="Example 7 attention heatmap at layer 2" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-7-layer-5.png" alt="Example 7 attention heatmap at layer 5" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-7-layer-8.png" alt="Example 7 attention heatmap at layer 8" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-7-layer-11.png" alt="Example 7 attention heatmap at layer 11" width="160"></td>
    </tr>
    <tr style="background: transparent; border: 0;">
      <th style="border: 0;"><div align="center">8</div></th>
      <td align="center" style="border: 0;"><img src="examples/8.jpg" alt="Original example 8" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-8-layer-2.png" alt="Example 8 attention heatmap at layer 2" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-8-layer-5.png" alt="Example 8 attention heatmap at layer 5" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-8-layer-8.png" alt="Example 8 attention heatmap at layer 8" width="160"></td>
      <td align="center" style="border: 0;"><img src="assets/attention/example-8-layer-11.png" alt="Example 8 attention heatmap at layer 11" width="160"></td>
    </tr>
  </tbody>
</table>

**Figure 2.** Mean-head attention heatmaps from layers 2, 5, 8, and 11 of DINOv2 ViT-S/14 with registers, rendered with AnyUp.

## Methods

| Method | Output | Model family |
|---|---|---|
| Attention | Spatial maps from selected transformer layers and heads | Vision Transformer via `timm` |
| Rollout | Attention propagated through successive transformer layers | Vision Transformer via `timm` |
| Grad-CAM | Class-specific activation maps | CNN via `torchvision` |
| Patch PCA | RGB projection of patch embeddings, with optional image foreground selection | Vision Transformer via `timm` |

These are diagnostic representations of model behavior. Attention and Grad-CAM maps do not establish causal explanations; PCA colors represent feature variation rather than semantic classes.

## Install

Vision Lens supports Linux and Python 3.12 or newer. With [uv](https://docs.astral.sh/uv/) installed, create the locked environment from the repository root:

```bash
uv sync --locked
```

For video decoding and export, include the optional dependency:

```bash
uv sync --locked --extra video
```

Pretrained weights are downloaded on first use. AnyUp interpolation also downloads its model and checkpoint on first use.

## Run

Run an image workflow:

```bash
uv run vision-lens run --config configs/attention.dino_vits8.yaml --set input.limit=1
```

To process your own video, copy an image configuration, replace its `input` section, and add a `video` section:

```yaml
input:
  files: [/path/to/your/video.mp4]

video:
  sampling_rate: auto
```

Then run it with `uv run vision-lens run --config path/to/your/config.yaml`. Standard video outputs are silent MP4 files; transparent overlays can additionally be exported as ProRes 4444 MOV or VP9 WebM files. See the [configuration reference](docs/configuration.md#video) for sampling and encoding options.

Validate settings without loading a model, or inspect all resolved values before a run:

```bash
uv run vision-lens validate --config configs/attention.dino_vits8.yaml
uv run vision-lens resolve --config configs/attention.dino_vits8.yaml
```

`--set section.key=value` overrides a setting using YAML value syntax and can be repeated. For example, `--set runtime.batch_size=2` changes the inference batch size. See the [configuration reference](docs/configuration.md) for available settings and workflow constraints.

### Configuration

A minimal image workflow selects inputs, a model, an analysis, and an output directory:

```yaml
input:
  files: [examples/1.jpg]

model:
  architecture: vit
  backend: timm
  name: vit_small_patch8_224.dino

preprocessing:
  image_size: 224

analysis:
  method: attention
  layers: [11]

output:
  directory: outputs/attention
```

The configurations in [`configs/`](configs/) cover all four methods. Image inputs can be listed explicitly or selected from folders. A `video` section enables timestamp-based frame sampling. Relative paths in a configuration resolve from the nearest project root containing `pyproject.toml`, or from the working directory when no project root is found.

### Outputs

Depending on the workflow, Vision Lens writes heatmaps or PCA color maps, flattened overlays, transparent RGBA overlays, comparison grids, MP4/MOV/WebM streams, and optional raw arrays. Each completed image or single-video run writes `run-manifest.json` with resolved settings, model and runtime details, input metadata, and output paths. Multiple videos produce separate subdirectories and manifests.

Grid layout, output resolution, interpolation, colormap, normalization, and overwrite behavior are configurable. The [configuration reference](docs/configuration.md) documents their defaults and compatibility rules.

## Python API

```python
from vision_lens import load_config, run_pipeline_from_config

config = load_config("configs/attention.dino_vits8.yaml")
result = run_pipeline_from_config(config, show_progress=False)

for path in result.output_paths:
    print(path)
```

The public API also provides `run_pipeline(path)`, `parse_config`, `validate_config`, and `resolved_config_yaml`. Results expose the resolved `config` and generated `output_paths`. Configuration failures raise `ConfigurationError`; execution failures raise `PipelineError`. Both inherit from `VisionLensError`.

## License and attribution

Vision Lens is available under the [MIT License](LICENSE). Sources for bundled [images](examples/ATTRIBUTION.md) are documented separately.
