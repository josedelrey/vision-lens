from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Iterator
from typing import Any, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def status(message: str) -> None:
    """Write a short status line without disrupting an active progress bar."""
    tqdm.write(f"vision-lens: {message}", file=sys.stderr)


def track_image_batches(
    batches: Iterable[Any],
    *,
    total: int,
    description: str,
) -> Iterator[Any]:
    yield from track_units(
        batches,
        total=total,
        description=description,
        unit="image",
        size=lambda batch: batch.count,
    )


def track_video_batches(
    batches: Iterable[Any],
    *,
    total: int | None,
    description: str,
) -> Iterator[Any]:
    yield from track_units(
        batches,
        total=total,
        description=description,
        unit="frame",
        size=lambda batch: len(batch.frames),
    )


def track_units(
    items: Iterable[T],
    *,
    total: int | None,
    description: str,
    unit: str,
    size: Callable[[T], int],
) -> Iterator[T]:
    with _bar(total=total, description=description, unit=unit) as bar:
        for item in items:
            yield item
            bar.update(size(item))


def _bar(*, total: int | None, description: str, unit: str) -> Any:
    return tqdm(
        total=total,
        desc=description,
        unit=unit,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.5,
        disable=not sys.stderr.isatty(),
    )
