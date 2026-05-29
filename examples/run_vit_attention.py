from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from vision_lens.config import (  # noqa: E402
    AttentionConfig,
    OutputConfig,
    VisionLensConfig,
    load_config,
)
from vision_lens.pipeline import (  # noqa: E402
    PipelineResult,
    run_vit_attention_from_config,
    run_vit_rollout_comparison_from_config,
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    config = load_config(args.config)

    results = {
        "fused heads": export_fused_heads(config, output_dir),
        "individual heads": export_individual_heads(
            config,
            output_dir,
            heads=tuple(args.heads),
        ),
        "attention rollout": export_rollout(config, output_dir),
    }

    print(f"saved ViT attention figures to {output_dir}")
    for label, result in results.items():
        print(f"{label}: {len(result.output_paths)} files")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ViT attention figures for the course exercise.",
    )
    parser.add_argument(
        "--config",
        default="configs/vit_attention.example.yaml",
        help="Config used for ViT attention and rollout figures.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/vit_attention",
        help="Directory where generated ViT figure folders are written.",
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Heads exported individually for the head-comparison example.",
    )
    return parser.parse_args()


def export_fused_heads(
    config: VisionLensConfig,
    output_dir: Path,
) -> PipelineResult:
    config = replace(
        config,
        output=OutputConfig(output_dir / "fused_heads"),
    )
    return run_vit_attention_from_config(config)


def export_individual_heads(
    config: VisionLensConfig,
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
        output=OutputConfig(output_dir / "individual_heads"),
    )
    return run_vit_attention_from_config(config)


def export_rollout(
    config: VisionLensConfig,
    output_dir: Path,
) -> PipelineResult:
    return run_vit_rollout_comparison_from_config(
        config,
        output_dir=output_dir / "rollout",
    )


if __name__ == "__main__":
    main()
