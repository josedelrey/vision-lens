from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from vision_lens.config import OutputConfig, VisionLensConfig, load_config  # noqa: E402
from vision_lens.pipeline import (  # noqa: E402
    GradCamPipelineResult,
    run_gradcam_from_config,
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    config = load_config(args.config)
    result = export_gradcam(config, output_dir)

    print(f"saved CNN Grad-CAM figures to {output_dir}")
    print(f"grad-cam: {len(result.output_paths)} files")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CNN Grad-CAM figures for the course exercise.",
    )
    parser.add_argument(
        "--config",
        default="configs/gradcam.example.yaml",
        help="Config used for CNN Grad-CAM figures.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/cnn_gradcam",
        help="Directory where generated CNN Grad-CAM figures are written.",
    )
    return parser.parse_args()


def export_gradcam(
    config: VisionLensConfig,
    output_dir: Path,
) -> GradCamPipelineResult:
    config = replace(
        config,
        output=OutputConfig(output_dir),
    )
    return run_gradcam_from_config(config)


if __name__ == "__main__":
    main()
