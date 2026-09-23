"""Media classification and per-workflow configuration derivation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

from vision_lens.config.schema import (
    GRID_VISUALIZATION_KEYS,
    SECTION_DEFAULTS,
    InputConfig,
    PatchPCAAnalysisConfig,
    RolloutAnalysisConfig,
    VideoConfig,
    VisionLensConfig,
)

MediaKind = Literal["image", "video"]

IMAGE_INPUT_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png", ".webp"})
VIDEO_INPUT_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
SUPPORTED_INPUT_EXTENSIONS = IMAGE_INPUT_EXTENSIONS | VIDEO_INPUT_EXTENSIONS


def media_kind(path: Path) -> MediaKind | None:
    """Classify a supported input path by its case-insensitive extension."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_INPUT_EXTENSIONS:
        return "image"
    if suffix in VIDEO_INPUT_EXTENSIONS:
        return "video"
    return None


def partition_media_paths(
    paths: tuple[Path, ...],
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return image and video paths while retaining their relative order."""
    images = tuple(path for path in paths if media_kind(path) == "image")
    videos = tuple(path for path in paths if media_kind(path) == "video")
    return images, videos


def media_count_summary(paths: tuple[Path, ...]) -> str:
    """Format detected image and video counts for user-facing status output."""
    images, videos = partition_media_paths(paths)
    image_label = "image" if len(images) == 1 else "images"
    video_label = "video" if len(videos) == 1 else "videos"
    return f"{len(images)} {image_label}, {len(videos)} {video_label}"


def split_media_configs(
    config: VisionLensConfig,
) -> tuple[VisionLensConfig | None, VisionLensConfig | None]:
    """Derive canonical image and video configurations from one user config."""
    image_paths, video_paths = partition_media_paths(config.input.paths)
    mixed = bool(image_paths and video_paths)
    if not mixed:
        return (config, None) if image_paths else (None, config)

    image_config = None
    if image_paths:
        image_analysis = config.analysis
        if (
            mixed
            and isinstance(image_analysis, PatchPCAAnalysisConfig)
            and image_analysis.save_projection is not None
        ):
            image_analysis = replace(
                image_analysis,
                save_projection=_suffixed_path(
                    image_analysis.save_projection,
                    "images",
                ),
            )
        image_config = replace(
            config,
            input=InputConfig(paths=image_paths),
            analysis=image_analysis,
            output=replace(
                config.output,
                directory=(
                    config.output.directory / "images"
                    if mixed
                    else config.output.directory
                ),
            ),
            video=None,
        )

    video_config = None
    if video_paths:
        video_analysis = config.analysis
        if isinstance(video_analysis, PatchPCAAnalysisConfig):
            if video_analysis.projection == "load":
                video_analysis = PatchPCAAnalysisConfig(
                    projection="load",
                    projection_path=video_analysis.projection_path,
                )
            else:
                video_analysis = PatchPCAAnalysisConfig(
                    projection="fit",
                    save_projection=video_analysis.save_projection,
                )
        elif isinstance(video_analysis, RolloutAnalysisConfig):
            video_analysis = replace(video_analysis, heads=None, head_fusion="mean")

        video_output_directory = config.output.directory
        if mixed:
            video_output_directory = config.output.directory / "videos"
            if len(video_paths) == 1:
                video_output_directory /= video_paths[0].stem

        grid_defaults = {
            key: SECTION_DEFAULTS["visualization"][key]
            for key in GRID_VISUALIZATION_KEYS
        }
        video_config = replace(
            config,
            input=InputConfig(paths=video_paths),
            preprocessing=replace(
                config.preprocessing,
                resize=SECTION_DEFAULTS["preprocessing"]["resize"],
                crop=SECTION_DEFAULTS["preprocessing"]["crop"],
                pad=SECTION_DEFAULTS["preprocessing"]["pad"],
            ),
            analysis=video_analysis,
            runtime=replace(
                config.runtime,
                workers=SECTION_DEFAULTS["runtime"]["workers"],
            ),
            visualization=replace(
                config.visualization,
                **grid_defaults,
            ),
            output=replace(
                config.output,
                directory=video_output_directory,
                grids=False,
                image_format=SECTION_DEFAULTS["output"]["image_format"],
            ),
            video=config.video or VideoConfig(),
        )

    return image_config, video_config


def _suffixed_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")
