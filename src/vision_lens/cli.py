from __future__ import annotations

import argparse
from pathlib import Path

from vision_lens.pipeline import run_vit_attention


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vision-lens",
        description="Run Vision Lens attention visualization pipelines.",
    )
    parser.add_argument(
        "--config",
        default="configs/vit_attention.example.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args(argv)

    result = run_vit_attention(Path(args.config))
    print(f"saved {len(result.output_paths)} files to {result.config.output.directory}")
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
