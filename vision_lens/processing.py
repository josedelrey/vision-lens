from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_lens.config import PreprocessingConfig
from vision_lens.images import (
    build_preprocess,
    load_images,
    preprocess_images,
    tensors_to_display_images,
)
from vision_lens.models import LoadedModel


@dataclass(frozen=True)
class InputBatch:
    index: int
    count: int
    paths: tuple[Path, ...]
    labels: tuple[str, ...]
    images: tuple[Any, ...]


@dataclass(frozen=True)
class PreprocessedBatch:
    source: InputBatch
    inputs: Any
    display_images: tuple[Any, ...]


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


def iter_input_batches(
    paths: Sequence[Path],
    labels: Sequence[str],
    *,
    batch_size: int,
    workers: int = 0,
) -> Iterator[InputBatch]:
    if len(paths) != len(labels):
        raise ValueError("Input paths and labels must have the same length.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    for index, start in enumerate(range(0, len(paths), batch_size)):
        batch_paths = tuple(paths[start : start + batch_size])
        batch_labels = tuple(labels[start : start + batch_size])
        images = tuple(load_images(batch_paths, workers=workers))
        yield InputBatch(
            index=index,
            count=len(batch_paths),
            paths=batch_paths,
            labels=batch_labels,
            images=images,
        )


def build_batch_preprocessor(
    loaded_model: LoadedModel,
    config: PreprocessingConfig,
) -> Any:
    return build_preprocess(
        loaded_model.model,
        backend=loaded_model.metadata.backend,
        image_size=loaded_model.metadata.image_size,
        config=config,
        data_config=loaded_model.metadata.data_config,
    )


def preprocess_batch(
    batch: InputBatch,
    transform: Any,
    loaded_model: LoadedModel,
    *,
    include_display_images: bool = True,
) -> PreprocessedBatch:
    inputs = preprocess_images(batch.images, transform)
    actual = (int(inputs.shape[-2]), int(inputs.shape[-1]))
    expected = loaded_model.metadata.image_size
    if actual != expected:
        raise ValueError(
            f"Preprocessing produced image size {actual}, but the model requires "
            f"{expected}. Adjust preprocessing.resize, crop, or pad."
        )
    display_images = (
        tuple(tensors_to_display_images(inputs, loaded_model.metadata.data_config))
        if include_display_images
        else ()
    )
    return PreprocessedBatch(
        source=batch,
        inputs=inputs,
        display_images=display_images,
    )
