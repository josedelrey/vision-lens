# Video example scripts

The five video scripts mirror the image examples while processing one MP4 per
run. Install the optional decoder and encoder dependencies from the repository
root:

```bash
uv sync --extra video
```

By default, the scripts expect `videos/lego.mp4` and `videos/fern.mp4`. Encode
these from the NeRF PNG sequences at 10 and 15 source FPS respectively. Every
script uses `video.sampling_rate: auto`, so it samples and exports at the source
FPS: the Lego orbit produces about 40 output frames over four seconds and
Fern's two-orbit path about 120 frames over eight seconds. Both `videos/` and
`outputs/` are ignored by Git.

Check the input MP4 timing before running the examples. The NeRF renderer's
automatic MP4s are 30 FPS: using them directly would make Lego 1.33 seconds
and Fern four seconds, and changing Vision Lens's sampling rate would **not**
fix that playback speed. Re-encode the saved PNG sequences with FFmpeg's input
`-framerate` option if the files in `videos/` still have those timings.

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=avg_frame_rate,nb_frames,duration \
  -of default=noprint_wrappers=1 videos/lego.mp4
```

Expect 10 FPS, 40 frames, and four seconds for Lego; for Fern expect 15 FPS,
120 frames, and eight seconds.

| Analysis | Default scene | Command |
| --- | --- | --- |
| DINO ViT-S/8 attention | Lego | `uv run --extra video python scripts/run_dino_vits8_attention_video.py` |
| DINOv2 registers attention | Fern | `uv run --extra video python scripts/run_dinov2_reg4_attention_video.py` |
| DINOv2 attention rollout | Fern | `uv run --extra video python scripts/run_dinov2_reg4_rollout_video.py` |
| ResNet-50 Grad-CAM | Lego | `uv run --extra video python scripts/run_resnet50_gradcam_video.py` |
| DINOv2 patch PCA | Fern | `uv run --extra video python scripts/run_dinov2_pca_video.py` |

Attention examples select the last transformer layer to keep the number of
output videos manageable. Attention, rollout, and Grad-CAM write overlay and
side-by-side comparison MP4s. Patch PCA writes its color map and comparison
MP4, using one projection fitted across representative video frames. Outputs
are named after the input file and saved in method- and scene-specific folders
under `outputs/video/`.

To use Fern with the DINO ViT-S/8 attention script, override both the input
and output destination:

```bash
uv run --extra video python scripts/run_dino_vits8_attention_video.py \
  --set 'input.files=["videos/fern.mp4"]' \
  --set 'output.directory=outputs/video/dino_vits8_attention/fern'
```

Paths passed via `--set` resolve relative to the project root, just like paths
in the YAML config. Other useful overrides include
`--set video.temporal_smoothing=0.2` to reduce flicker and
`--set video.frame_limit=20` for a short preview. The Grad-CAM example uses the
model's top predicted class for each frame by default; set
`analysis.target_class` to a chosen ImageNet class index if a fixed target is
more appropriate for the demonstration.

The `.mp4` files are inputs and outputs of Vision Lens. Convert only the
reviewed result MP4s to GIFs for the README.
