from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from vision_lens.config import AttentionConfig, OutputConfig, load_config
from vision_lens.pipeline import (
    GradCamPipelineResult,
    PipelineResult,
    run_gradcam_from_config,
    run_vit_attention_from_config,
    run_vit_rollout_comparison_from_config,
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    vit_config = load_config(args.vit_config)
    cnn_config = load_config(args.cnn_config)

    results = {
        "vit fused heads": export_vit_fused(vit_config, output_dir),
        "vit individual heads": export_vit_heads(
            vit_config,
            output_dir,
            heads=tuple(args.heads),
        ),
        "vit rollout": export_vit_rollout(vit_config, output_dir),
        "cnn grad-cam": export_cnn_gradcam(cnn_config, output_dir),
    }

    print(f"saved visualization gallery to {output_dir}")
    for label, result in results.items():
        print(f"{label}: {len(result.output_paths)} files")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export example figures for every Vision Lens visualization path.",
    )
    parser.add_argument(
        "--vit-config",
        default="configs/vit_attention.example.yaml",
        help="Config used for ViT attention and rollout figures.",
    )
    parser.add_argument(
        "--cnn-config",
        default="configs/gradcam.example.yaml",
        help="Config used for CNN Grad-CAM figures.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/visualization_gallery",
        help="Directory where all generated figure folders are written.",
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="ViT heads exported individually for the head-comparison example.",
    )
    return parser.parse_args()


def export_vit_fused(config, output_dir: Path) -> PipelineResult:
    config = replace(
        config,
        output=OutputConfig(output_dir / "vit_fused_heads"),
    )
    return run_vit_attention_from_config(config)


def export_vit_heads(
    config,
    output_dir: Path,
    heads: tuple[int, ...],
) -> PipelineResult:
    if config.attention is None:
        raise ValueError("ViT head export requires an attention config.")

    config = replace(
        config,
        attention=AttentionConfig(
            layers=config.attention.layers,
            heads=heads,
            head_fusion="none",
        ),
        output=OutputConfig(output_dir / "vit_individual_heads"),
    )
    return run_vit_attention_from_config(config)


def export_vit_rollout(config, output_dir: Path) -> PipelineResult:
    return run_vit_rollout_comparison_from_config(
        config,
        output_dir=output_dir / "vit_rollout",
    )


def export_cnn_gradcam(config, output_dir: Path) -> GradCamPipelineResult:
    config = replace(
        config,
        output=OutputConfig(output_dir / "cnn_gradcam"),
    )
    return run_gradcam_from_config(config)


if __name__ == "__main__":
    main()
