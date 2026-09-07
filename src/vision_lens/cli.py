from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vision_lens.config import load_config, load_preset
from vision_lens.pipeline import run_pipeline_from_config
from vision_lens.presets import available_presets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vision-lens",
        description="Run Vision Lens model visualization pipelines.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--preset",
        choices=available_presets(),
        help="Named workflow whose settings are applied before config values.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override one setting; may be repeated and accepts YAML values.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List the built-in workflow presets and exit.",
    )
    args = parser.parse_args(argv)

    if args.list_presets:
        for preset_name in available_presets():
            print(preset_name)
        return 0

    overrides = _parse_overrides(args.set)
    if args.config is None and args.preset is None:
        config = load_config(
            Path("configs/vit_attention.example.yaml"),
            overrides=overrides,
        )
    elif args.config is None:
        config = load_preset(args.preset, overrides=overrides)
    else:
        config = load_config(
            Path(args.config),
            preset=args.preset,
            overrides=overrides,
        )

    result = run_pipeline_from_config(config)
    print(f"saved {len(result.output_paths)} files to {result.config.output.directory}")
    for output_path in result.output_paths:
        print(output_path)
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
