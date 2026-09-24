from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vision_lens.config.media import partition_media_paths, split_media_configs
from vision_lens.config.schema import (
    ALPHA_FORMAT_EXTENSIONS,
    KNOWN_VIT_DEPTHS,
    KNOWN_VIT_HEADS,
    AttentionAnalysisConfig,
    PatchPCAAnalysisConfig,
    RolloutAnalysisConfig,
    RolloutGrid,
    VisionLensConfig,
)

RUN_MANIFEST_NAME = "run-manifest.json"
PATCH_PCA_GRID_STEM = "patch_pca_comparison"


@dataclass(frozen=True)
class VideoRunLayout:
    source_path: Path
    output_directory: Path
    projection_path: Path | None


@dataclass(frozen=True)
class ArtifactPattern:
    directory: Path
    filename: re.Pattern[str]

    def matches(self, path: Path) -> bool:
        return (
            path.parent == self.directory
            and self.filename.fullmatch(path.name) is not None
        )


@dataclass(frozen=True)
class ArtifactPlan:
    output_directories: frozenset[Path]
    exact_paths: frozenset[Path]
    reserved_patterns: tuple[ArtifactPattern, ...]
    non_projection_paths: frozenset[Path]

    def matches(self, path: Path) -> bool:
        return path in self.exact_paths or any(
            pattern.matches(path) for pattern in self.reserved_patterns
        )

    def matches_non_projection(self, path: Path) -> bool:
        return path in self.non_projection_paths or any(
            pattern.matches(path) for pattern in self.reserved_patterns
        )

    def existing_paths(self) -> tuple[Path, ...]:
        existing = {path for path in self.exact_paths if path.exists()}
        for reserved in self.reserved_patterns:
            if not reserved.directory.is_dir():
                continue
            existing.update(
                path
                for path in reserved.directory.iterdir()
                if path.is_file() and reserved.matches(path)
            )
        return tuple(sorted(existing))


def unique_input_labels(paths: Sequence[Path]) -> tuple[str, ...]:
    stem_counts: dict[str, int] = {}
    for path in paths:
        key = path.stem.casefold()
        stem_counts[key] = stem_counts.get(key, 0) + 1

    labels = []
    used = set()
    for index, path in enumerate(paths):
        stem = path.stem
        label = stem
        if stem_counts[stem.casefold()] > 1 or label.casefold() in used:
            digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
            label = f"{stem}_{digest}"
        while label.casefold() in used:
            label = f"{stem}_{digest}_{index + 1}"
        labels.append(label)
        used.add(label.casefold())
    return tuple(labels)


def image_artifact_path(
    directory: Path,
    label: str,
    artifact: str,
    extension: str,
) -> Path:
    return directory / f"{label}_{artifact}.{extension}"


def named_artifact_path(directory: Path, stem: str, extension: str) -> Path:
    return directory / f"{stem}.{extension}"


def attention_image_stem(label: str, layer_index: int, head_name: str) -> str:
    return f"{label}_layer-{layer_index}_{head_name}"


def attention_layers_grid_stem(label: str, head_name: str) -> str:
    return f"{label}_layers_{head_name}"


def attention_images_grid_stem(layer_index: int, head_name: str) -> str:
    return f"layer-{layer_index}_images_{head_name}"


def rollout_image_stem(label: str, layer_index: int) -> str:
    return f"{label}_rollout-{layer_index}"


def rollout_grid_stem(label: str, mode: RolloutGrid = "comparison") -> str:
    suffix = "rollout_comparison" if mode == "comparison" else "rollout_layers"
    return f"{label}_{suffix}"


def gradcam_image_stem(label: str) -> str:
    return f"{label}_gradcam"


def gradcam_grid_stem() -> str:
    return "gradcam_images"


def map_stream_name(method: str, layer_index: int, head_name: str) -> str:
    return f"{method}_layer-{layer_index}_{head_name}"


def rendered_stream_name(
    stream_name: str,
    kind: Literal["heatmap", "overlay", "transparent_overlay"],
) -> str:
    return f"{stream_name}_{kind}"


def grid_page_path(
    directory: Path,
    stem: str,
    extension: str,
    page_index: int,
    page_count: int,
) -> Path:
    if page_count == 1:
        return directory / f"{stem}.{extension}"
    return directory / f"{stem}_part-{page_index + 1:03d}.{extension}"


def video_artifact_path(
    directory: Path,
    source_stem: str,
    name: str,
    extension: str = "mp4",
) -> Path:
    return directory / f"{source_stem}_{name}.{extension}"


def video_raw_batch_path(
    directory: Path,
    source_stem: str,
    name: str,
    batch_index: int,
    extension: str,
) -> Path:
    return directory / (f"{source_stem}_{name}_frames-{batch_index:06d}.{extension}")


def run_manifest_path(directory: Path) -> Path:
    return directory / RUN_MANIFEST_NAME


def video_run_layouts(config: VisionLensConfig) -> tuple[VideoRunLayout, ...]:
    if config.video is None:
        return ()
    labels = unique_input_labels(config.input.paths)
    multiple = len(config.input.paths) > 1
    projection = (
        config.analysis.save_projection
        if isinstance(config.analysis, PatchPCAAnalysisConfig)
        else None
    )
    layouts = []
    for source_path, label in zip(config.input.paths, labels, strict=True):
        output_directory = (
            config.output.directory / label if multiple else config.output.directory
        )
        projection_path = projection
        if multiple and projection_path is not None:
            projection_path = projection_path.with_name(
                f"{projection_path.stem}_{label}{projection_path.suffix}"
            )
        layouts.append(
            VideoRunLayout(
                source_path=source_path,
                output_directory=output_directory,
                projection_path=projection_path,
            )
        )
    return tuple(layouts)


def artifact_plan(config: VisionLensConfig) -> ArtifactPlan:
    image_paths, video_paths = partition_media_paths(config.input.paths)
    if image_paths and video_paths:
        image_config, video_config = split_media_configs(config)
        assert image_config is not None and video_config is not None
        plans = (artifact_plan(image_config), artifact_plan(video_config))
        return ArtifactPlan(
            output_directories=frozenset(
                directory for plan in plans for directory in plan.output_directories
            ),
            exact_paths=frozenset(path for plan in plans for path in plan.exact_paths),
            reserved_patterns=tuple(
                pattern for plan in plans for pattern in plan.reserved_patterns
            ),
            non_projection_paths=frozenset(
                path for plan in plans for path in plan.non_projection_paths
            ),
        )

    exact_paths: set[Path] = set()
    projection_paths: set[Path] = set()
    patterns: list[ArtifactPattern] = []
    if config.video is None:
        directories = frozenset({config.output.directory})
        exact_paths.add(run_manifest_path(config.output.directory))
        if isinstance(config.analysis, PatchPCAAnalysisConfig):
            exact_paths.update(_image_patch_pca_paths(config))
            if config.analysis.save_projection is not None:
                projection_paths.add(config.analysis.save_projection)
        else:
            patterns.extend(
                ArtifactPattern(config.output.directory, pattern)
                for pattern in _image_artifact_patterns(config)
            )
    else:
        layouts = video_run_layouts(config)
        directories = frozenset(layout.output_directory for layout in layouts)
        for layout in layouts:
            exact_paths.add(run_manifest_path(layout.output_directory))
            if layout.projection_path is not None:
                projection_paths.add(layout.projection_path)
            patterns.extend(
                ArtifactPattern(layout.output_directory, pattern)
                for pattern in _video_artifact_patterns(config, layout.source_path.stem)
            )
    non_projection_paths = frozenset(exact_paths)
    exact_paths.update(projection_paths)
    return ArtifactPlan(
        output_directories=directories,
        exact_paths=frozenset(exact_paths),
        reserved_patterns=tuple(patterns),
        non_projection_paths=non_projection_paths,
    )


def validate_artifact_paths(config: VisionLensConfig) -> None:
    image_paths, video_paths = partition_media_paths(config.input.paths)
    if image_paths and video_paths:
        image_config, video_config = split_media_configs(config)
        assert image_config is not None and video_config is not None
        validate_artifact_paths(image_config)
        validate_artifact_paths(video_config)
        image_plan = artifact_plan(image_config)
        video_plan = artifact_plan(video_config)
        collision = _cross_plan_collision(image_plan, video_plan)
        if collision is not None:
            raise ValueError(
                "A planned image output collides with a planned video output: "
                f"{collision}. Choose different output or projection paths."
            )
        _validate_artifact_plan(
            config,
            artifact_plan(config),
            save_paths=(
                *_projection_save_paths(image_config),
                *_projection_save_paths(video_config),
            ),
        )
        return

    plan = artifact_plan(config)
    _validate_artifact_plan(
        config,
        plan,
        save_paths=_projection_save_paths(config),
    )


def _validate_artifact_plan(
    config: VisionLensConfig,
    plan: ArtifactPlan,
    *,
    save_paths: tuple[Path, ...],
) -> None:
    for directory in plan.output_directories:
        _validate_output_directory(directory)
    for input_path in config.input.paths:
        if plan.matches_non_projection(input_path):
            raise ValueError(
                f"A generated output would overwrite input file {input_path}. "
                "Choose a different output.directory or rename the input."
            )

    if not isinstance(config.analysis, PatchPCAAnalysisConfig):
        return
    if config.analysis.projection_path is not None and plan.matches(
        config.analysis.projection_path
    ):
        raise ValueError(
            "analysis.projection_path is a protected input and must not collide "
            f"with a planned output: {config.analysis.projection_path}."
        )

    for save_path in save_paths:
        _validate_write_path(save_path, config, plan)


def _projection_save_paths(config: VisionLensConfig) -> tuple[Path, ...]:
    if not isinstance(config.analysis, PatchPCAAnalysisConfig):
        return ()
    if config.video is None:
        return (
            (config.analysis.save_projection,)
            if config.analysis.save_projection is not None
            else ()
        )
    return tuple(
        layout.projection_path
        for layout in video_run_layouts(config)
        if layout.projection_path is not None
    )


def _cross_plan_collision(
    left: ArtifactPlan,
    right: ArtifactPlan,
) -> Path | None:
    for path in sorted(left.exact_paths):
        if right.matches(path):
            return path
    for path in sorted(right.exact_paths):
        if left.matches(path):
            return path
    return None


def check_artifact_overwrite(config: VisionLensConfig) -> None:
    if config.output.overwrite != "error":
        return
    existing = artifact_plan(config).existing_paths()
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output artifact(s) already exist: {paths}. Choose a new output "
            "directory or set output.overwrite to 'replace' or 'skip'."
        )


def _image_patch_pca_paths(config: VisionLensConfig) -> set[Path]:
    paths: set[Path] = set()
    labels = unique_input_labels(config.input.paths)
    if config.output.heatmaps:
        paths.update(
            image_artifact_path(
                config.output.directory,
                label,
                "patch_pca",
                config.output.image_format,
            )
            for label in labels
        )
    if config.output.raw_arrays:
        for label in labels:
            paths.add(
                image_artifact_path(
                    config.output.directory,
                    label,
                    "patch_embeddings",
                    config.output.raw_format,
                )
            )
            paths.add(
                image_artifact_path(
                    config.output.directory,
                    label,
                    "foreground_mask",
                    config.output.raw_format,
                )
            )
    if config.output.grids:
        page_size = config.visualization.items_per_grid or len(config.input.paths)
        page_count = (len(config.input.paths) + page_size - 1) // page_size
        paths.update(
            grid_page_path(
                config.output.directory,
                PATCH_PCA_GRID_STEM,
                config.visualization.grid_format,
                page_index,
                page_count,
            )
            for page_index in range(page_count)
        )
    return paths


def _validate_write_path(
    path: Path,
    config: VisionLensConfig,
    plan: ArtifactPlan,
) -> None:
    if path.exists() and path.is_dir():
        raise ValueError(
            f"analysis.save_projection must be a file path, not a directory: {path}."
        )
    _validate_existing_parent(path, "analysis.save_projection")
    if path in config.input.paths:
        raise ValueError("analysis.save_projection must not overwrite an input file.")
    if plan.matches_non_projection(path):
        raise ValueError(
            f"analysis.save_projection must not use a planned output path: {path}."
        )


def _image_artifact_patterns(
    config: VisionLensConfig,
) -> tuple[re.Pattern[str], ...]:
    labels = "|".join(
        re.escape(label) for label in unique_input_labels(config.input.paths)
    )
    image_extension = re.escape(config.output.image_format)
    raw_extension = re.escape(config.output.raw_format)
    grid_extension = re.escape(config.visualization.grid_format)
    image_page = _page_pattern(
        len(config.input.paths), config.visualization.items_per_grid
    )
    patterns: list[str] = []

    if config.analysis.method == "gradcam":
        if config.output.heatmaps:
            patterns.append(rf"(?:{labels})_gradcam_heatmap\.{image_extension}")
        if config.output.overlays:
            patterns.append(rf"(?:{labels})_gradcam_overlay\.{image_extension}")
        if config.output.transparent_overlays:
            patterns.append(rf"(?:{labels})_gradcam_transparent_overlay\.png")
        if config.output.raw_arrays:
            patterns.append(rf"(?:{labels})_gradcam\.{raw_extension}")
        if config.output.grids:
            patterns.append(rf"gradcam_images{image_page}\.{grid_extension}")
        return tuple(re.compile(pattern) for pattern in patterns)

    assert isinstance(config.analysis, AttentionAnalysisConfig | RolloutAnalysisConfig)
    known_layer_count = KNOWN_VIT_DEPTHS.get(config.model.name)
    layers = _number_pattern(config.analysis.layers, known_layer_count)
    layer_page = _layer_page_pattern(
        config.analysis.layers,
        config.visualization.items_per_grid,
        known_layer_count,
    )
    if config.analysis.method == "rollout":
        stem = rf"(?:{labels})_rollout-(?:{layers})"
        _append_map_output_patterns(
            patterns,
            stem,
            config,
            image_extension,
            raw_extension,
        )
        if config.output.grids:
            grid_stem = (
                "rollout_comparison"
                if config.visualization.rollout_grid == "comparison"
                else "rollout_layers"
            )
            patterns.append(rf"(?:{labels})_{grid_stem}{layer_page}\.{grid_extension}")
        return tuple(re.compile(pattern) for pattern in patterns)

    heads = _attention_head_pattern(
        config.analysis,
        KNOWN_VIT_HEADS.get(config.model.name),
    )
    stem = rf"(?:{labels})_layer-(?:{layers})_(?:{heads})"
    _append_map_output_patterns(
        patterns,
        stem,
        config,
        image_extension,
        raw_extension,
    )
    if config.output.grids:
        patterns.extend(
            (
                rf"(?:{labels})_layers_(?:{heads}){layer_page}\.{grid_extension}",
                rf"layer-(?:{layers})_images_(?:{heads}){image_page}\.{grid_extension}",
            )
        )
    return tuple(re.compile(pattern) for pattern in patterns)


def _video_artifact_patterns(
    config: VisionLensConfig,
    source_stem: str,
) -> tuple[re.Pattern[str], ...]:
    assert config.video is not None
    prefix = re.escape(source_stem)
    patterns: list[str] = []
    stream: str
    if config.analysis.method == "patch_pca":
        stream = "patch_pca"
    elif config.analysis.method == "gradcam":
        stream = "gradcam"
    else:
        assert isinstance(
            config.analysis, AttentionAnalysisConfig | RolloutAnalysisConfig
        )
        layers = _number_pattern(
            config.analysis.layers,
            KNOWN_VIT_DEPTHS.get(config.model.name),
        )
        heads = (
            _attention_head_pattern(
                config.analysis,
                KNOWN_VIT_HEADS.get(config.model.name),
            )
            if config.analysis.method == "attention"
            else "heads-mean"
        )
        stream = rf"{config.analysis.method}_layer-(?:{layers})_(?:{heads})"

    if config.output.heatmaps:
        heatmap_stream = stream if stream == "patch_pca" else f"{stream}_heatmap"
        patterns.append(rf"{prefix}_{heatmap_stream}\.mp4")
    if config.output.overlays:
        patterns.append(rf"{prefix}_{stream}_overlay\.mp4")
    if config.output.transparent_overlays:
        alpha_extension = ALPHA_FORMAT_EXTENSIONS[config.video.alpha_format]
        patterns.append(rf"{prefix}_{stream}_transparent_overlay\.{alpha_extension}")
    if config.output.raw_arrays:
        raw_stream = "patch_pca_rgb" if stream == "patch_pca" else stream
        extension = re.escape(config.output.raw_format)
        patterns.append(rf"{prefix}_{raw_stream}_frames-\d{{6,}}\.{extension}")
    return tuple(re.compile(pattern) for pattern in patterns)


def _append_map_output_patterns(
    patterns: list[str],
    stem: str,
    config: VisionLensConfig,
    image_extension: str,
    raw_extension: str,
) -> None:
    if config.output.heatmaps:
        patterns.append(rf"{stem}_heatmap\.{image_extension}")
    if config.output.overlays:
        patterns.append(rf"{stem}_overlay\.{image_extension}")
    if config.output.transparent_overlays:
        patterns.append(rf"{stem}_transparent_overlay\.png")
    if config.output.raw_arrays:
        patterns.append(rf"{stem}\.{raw_extension}")


def _number_pattern(
    values: str | tuple[int, ...],
    known_count: int | None,
) -> str:
    if values == "all":
        if known_count is None:
            return r"\d+"
        return "|".join(str(value) for value in range(known_count))
    return "|".join(str(value) for value in values)


def _page_pattern(item_count: int, items_per_page: int | None) -> str:
    page_size = items_per_page or item_count
    page_count = (item_count + page_size - 1) // page_size
    if page_count == 1:
        return ""
    pages = "|".join(f"{page:03d}" for page in range(1, page_count + 1))
    return rf"_part-(?:{pages})"


def _layer_page_pattern(
    layers: str | tuple[int, ...],
    items_per_page: int | None,
    known_layer_count: int | None,
) -> str:
    if layers == "all":
        if known_layer_count is None:
            return r"(?:_part-\d{3,})?"
        return _page_pattern(known_layer_count, items_per_page)
    return _page_pattern(len(layers), items_per_page)


def _attention_head_pattern(
    analysis: AttentionAnalysisConfig | RolloutAnalysisConfig,
    known_head_count: int | None,
) -> str:
    if analysis.head_fusion != "none":
        return f"heads-{re.escape(analysis.head_fusion)}"
    if analysis.heads is None:
        if known_head_count is None:
            return r"head-\d+"
        return "|".join(f"head-{head}" for head in range(known_head_count))
    return "|".join(f"head-{head}" for head in analysis.heads)


def _validate_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(
                f"output.directory must be a directory, not a file: {path}."
            )
        return
    _validate_existing_parent(path, "output.directory")


def _validate_existing_parent(path: Path, field_name: str) -> None:
    parent = path.parent
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        raise ValueError(
            f"{field_name} cannot be created because a parent path is not a "
            f"directory: {parent}."
        )
