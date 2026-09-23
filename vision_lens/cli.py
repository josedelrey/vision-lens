from __future__ import annotations

import argparse
from collections.abc import Mapping
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
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override one setting; may be repeated and accepts YAML values.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help=(
            "Deprecated compatibility flag; video inputs are detected automatically."
        ),
    )
    config_group = parser.add_argument_group(
        "configuration fields",
        "Named flags accept the same YAML values as their config fields. "
        "They override both --config and --set values.",
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
    set_overrides = _parse_overrides(args.set)
    named_assignments = [
        f"{path}={getattr(args, destination)}"
        for destination, path in CONFIG_OPTION_DESTINATIONS.items()
        if hasattr(args, destination)
    ]
    named_overrides = _parse_overrides(named_assignments)
    if args.video:
        named_overrides = _deep_merge({"video": {}}, named_overrides)
    return _deep_merge(set_overrides, named_overrides)


def _parse_overrides(assignments: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for assignment in assignments:
        path, separator, raw_value = assignment.partition("=")
        keys = path.split(".")
        if not separator or not all(keys):
            raise ValueError(
                f"Invalid override {assignment!r}; expected SECTION.KEY=VALUE."
            )

        target = overrides
        for key in keys[:-1]:
            existing = target.setdefault(key, {})
            if not isinstance(existing, dict):
                raise ValueError(
                    f"Override path {path!r} conflicts with another value."
                )
            target = existing
        target[keys[-1]] = yaml.safe_load(raw_value)
    return overrides


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
