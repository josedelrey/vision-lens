from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vision_lens.config import load_config, resolved_config_yaml
from vision_lens.pipeline import run_pipeline_from_config
from vision_lens.video_pipeline import VideoBatchPipelineResult


def main(argv: list[str] | None = None) -> int:
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
        required=True,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override one setting; may be repeated and accepts YAML values.",
    )
    args = parser.parse_args(argv)

    try:
        overrides = _parse_overrides(args.set)
        config = load_config(Path(args.config), overrides=overrides)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))

    if args.command == "validate":
        print("configuration is valid")
        return 0
    if args.command == "resolve":
        print(resolved_config_yaml(config), end="")
        return 0

    try:
        result = run_pipeline_from_config(config)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    if isinstance(result, VideoBatchPipelineResult):
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


if __name__ == "__main__":
    raise SystemExit(main())
