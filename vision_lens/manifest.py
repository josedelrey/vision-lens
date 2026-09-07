from __future__ import annotations

import json
import platform
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from vision_lens.config import VisionLensConfig, config_to_dict
from vision_lens.models import LoadedModel

MANIFEST_NAME = "run-manifest.json"


def manifest_path(config: VisionLensConfig) -> Path:
    return config.output.directory / MANIFEST_NAME


def check_manifest_overwrite(config: VisionLensConfig) -> None:
    path = manifest_path(config)
    if path.exists() and config.output.overwrite == "error":
        raise FileExistsError(
            f"Run manifest already exists: {path}. Choose a new output directory or "
            "set output.overwrite to 'replace' or 'skip'."
        )


def write_run_manifest(
    config: VisionLensConfig,
    loaded_model: LoadedModel,
    input_labels: Sequence[str],
    output_paths: Sequence[Path],
    *,
    started_at: datetime,
) -> Path | None:
    path = manifest_path(config)
    if path.exists() and config.output.overwrite == "skip":
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = loaded_model.metadata
    payload = {
        "schema_version": 1,
        "status": "completed",
        "started_at": started_at.astimezone(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "configuration": config_to_dict(config),
        "model": {
            "architecture": metadata.architecture,
            "backend": metadata.backend,
            "name": metadata.name,
            "pretrained": metadata.pretrained,
            "device": metadata.device,
            "precision": config.runtime.precision,
            "input_size": list(metadata.input_size),
            "patch_size": (
                None if metadata.patch_size is None else list(metadata.patch_size)
            ),
            "num_classes": metadata.num_classes,
        },
        "versions": _versions(),
        "inputs": [
            {
                "index": index,
                "id": label,
                "path": str(input_path),
                "size_bytes": input_path.stat().st_size,
                "modified_ns": input_path.stat().st_mtime_ns,
            }
            for index, (input_path, label) in enumerate(
                zip(config.input.paths, input_labels, strict=True)
            )
        ],
        "outputs": [str(output_path.resolve()) for output_path in output_paths],
    }
    temporary_path = path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary_path.replace(path)
    return path


def _versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "vision_lens": _package_version("vision-lens"),
        "torch": _package_version("torch"),
        "torchvision": _package_version("torchvision"),
        "timm": _package_version("timm"),
        "numpy": _package_version("numpy"),
        "pillow": _package_version("pillow"),
        "matplotlib": _package_version("matplotlib"),
        "pyyaml": _package_version("pyyaml"),
        "executable": sys.executable,
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
