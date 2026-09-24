from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from vision_lens import (
    VisionLensError,
    load_config,
    parse_config,
    resolved_config_yaml,
)
from vision_lens.config.media import media_count_summary
from vision_lens.config.schema import ANALYSIS_KEYS, SECTION_KEYS

CONFIG_OPTION_PATHS = tuple(
    (section, key)
    for section, keys in {
        **SECTION_KEYS,
        "analysis": set().union(*ANALYSIS_KEYS.values()),
    }.items()
    for key in sorted(keys)
)
CONFIG_OPTION_DESTINATIONS = {
    f"config_{section}_{key}": f"{section}.{key}"
    for section, key in CONFIG_OPTION_PATHS
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        overrides = _cli_overrides(args)
        config = (
            load_config(Path(args.config), overrides=overrides)
            if args.config is not None
            else parse_config(overrides)
        )
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))

    if args.command == "validate":
        print(f"configuration is valid ({media_count_summary(config.input.paths)})")
        return 0
    if args.command == "resolve":
        print(resolved_config_yaml(config), end="")
        return 0

    from vision_lens import run_pipeline_from_config
    from vision_lens.pipeline import MixedPipelineResult
    from vision_lens.pipeline.video import VideoBatchPipelineResult

    try:
        result = run_pipeline_from_config(config)
    except VisionLensError as error:
        parser.error(str(error))
    if isinstance(result, MixedPipelineResult):
        video_count = (
            len(result.video.videos)
            if isinstance(result.video, VideoBatchPipelineResult)
            else 1
        )
        print(
            f"saved {len(result.output_paths)} outputs plus "
            f"{video_count + 1} run manifests under "
            f"{result.config.output.directory}"
        )
    elif isinstance(result, VideoBatchPipelineResult):
        print(
            f"saved {len(result.output_paths)} outputs plus "
            f"{len(result.videos)} run manifests under "
            f"{result.config.output.directory}"
        )
    else:
        print(
            f"saved {len(result.output_paths)} outputs plus run manifest to "
            f"{result.config.output.directory}"
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vision-lens",
        description="Run Vision Lens model visualization pipelines.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "validate", "resolve"),
        default="run",
        help="Run the workflow, validate configuration, or print resolved YAML.",
    )
    parser.add_argument(
        "--config",
        help="Optional YAML config file; command-line values override it.",
    )
    config_group = parser.add_argument_group(
        "configuration fields",
        "Named flags accept the same YAML values as their config fields. "
        "They override --config values.",
    )
    for destination, path in CONFIG_OPTION_DESTINATIONS.items():
        config_group.add_argument(
            f"--{path.replace('.', '-').replace('_', '-')}",
            dest=destination,
            default=argparse.SUPPRESS,
            metavar="YAML",
            help=f"Set {path}.",
        )
    return parser


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for destination, path in CONFIG_OPTION_DESTINATIONS.items():
        if not hasattr(args, destination):
            continue
        section, key = path.split(".", maxsplit=1)
        overrides.setdefault(section, {})[key] = yaml.safe_load(
            getattr(args, destination)
        )
    return overrides


if __name__ == "__main__":
    raise SystemExit(main())
