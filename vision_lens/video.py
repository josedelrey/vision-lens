from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vision_lens.config import VideoConfig


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    width: int
    height: int
    duration: float | None
    source_frame_rate: float | None


@dataclass(frozen=True)
class SampledVideoFrame:
    index: int
    timestamp: float
    source_timestamp: float
    image: Image.Image


@dataclass(frozen=True)
class VideoFrameBatch:
    index: int
    frames: tuple[SampledVideoFrame, ...]


def require_video_dependencies() -> Any:
    try:
        import av
    except ImportError as error:
        raise RuntimeError(
            "Video support requires the optional dependencies. Install them with "
            "`pip install 'vision-lens[video]'` or use the repository's Conda "
            "environment."
        ) from error
    return av


def probe_video(path: str | Path) -> VideoMetadata:
    av = require_video_dependencies()
    source = Path(path)
    with av.open(str(source)) as container:
        stream = _video_stream(container)
        duration = _stream_duration(stream)
        if duration is None and container.duration is not None:
            duration = float(container.duration / av.time_base)
        source_rate = (
            None if stream.average_rate is None else float(stream.average_rate)
        )
        return VideoMetadata(
            path=source,
            width=int(stream.codec_context.width),
            height=int(stream.codec_context.height),
            duration=duration,
            source_frame_rate=source_rate,
        )


def iter_sampled_frames(
    path: str | Path,
    config: VideoConfig,
) -> Iterator[SampledVideoFrame]:
    av = require_video_dependencies()
    source = Path(path)
    with av.open(str(source)) as container:
        stream = _video_stream(container)
        stream_time_base = float(stream.time_base)
        origin = (
            float(stream.start_time * stream.time_base)
            if stream.start_time is not None
            else 0.0
        )
        if config.start_time > 0:
            absolute_start = origin + config.start_time
            container.seek(
                int(absolute_start / stream_time_base),
                backward=True,
                any_frame=False,
                stream=stream,
            )

        output_index = 0
        next_timestamp = config.start_time
        for decoded_index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                source_rate = (
                    None if stream.average_rate is None else float(stream.average_rate)
                )
                if source_rate is None or source_rate <= 0:
                    continue
                source_timestamp = decoded_index / source_rate
            else:
                source_timestamp = float(frame.pts * frame.time_base) - origin
            if source_timestamp + 1e-9 < next_timestamp:
                continue

            image = frame.to_image().convert("RGB")
            while source_timestamp + 1e-9 >= next_timestamp:
                if config.end_time is not None and next_timestamp >= config.end_time:
                    return
                if (
                    config.frame_limit is not None
                    and output_index >= config.frame_limit
                ):
                    return
                yield SampledVideoFrame(
                    index=output_index,
                    timestamp=next_timestamp,
                    source_timestamp=source_timestamp,
                    image=image.copy(),
                )
                output_index += 1
                next_timestamp = config.start_time + (
                    output_index / config.sampling_rate
                )


def iter_video_batches(
    path: str | Path,
    config: VideoConfig,
    batch_size: int,
) -> Iterator[VideoFrameBatch]:
    frames = []
    batch_index = 0
    for frame in iter_sampled_frames(path, config):
        frames.append(frame)
        if len(frames) == batch_size:
            yield VideoFrameBatch(batch_index, tuple(frames))
            frames.clear()
            batch_index += 1
    if frames:
        yield VideoFrameBatch(batch_index, tuple(frames))


def estimated_sample_count(
    metadata: VideoMetadata,
    config: VideoConfig,
) -> int | None:
    end_time = config.end_time
    if metadata.duration is not None:
        end_time = (
            metadata.duration if end_time is None else min(end_time, metadata.duration)
        )
    if end_time is None:
        return config.frame_limit
    duration = max(0.0, end_time - config.start_time)
    count = int(np.ceil(duration * config.sampling_rate - 1e-9))
    if config.frame_limit is not None:
        count = min(count, config.frame_limit)
    return count


def representative_frame_indices(
    sample_count: int | None,
    maximum: int,
) -> tuple[int, ...] | None:
    if sample_count is None:
        return None
    selected_count = min(sample_count, maximum)
    if selected_count <= 0:
        return ()
    return tuple(
        sorted(
            set(
                int(round(value))
                for value in np.linspace(0, sample_count - 1, selected_count)
            )
        )
    )


def resolved_output_resolution(
    metadata: VideoMetadata,
    configured: tuple[int, int] | None,
) -> tuple[int, int]:
    if configured is not None:
        return configured
    return (_even(metadata.width), _even(metadata.height))


class VideoWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        frame_rate: float,
        resolution: tuple[int, int],
        codec: str,
    ) -> None:
        av = require_video_dependencies()
        self._av = av
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(self.path), mode="w")
        self._rate = Fraction(str(frame_rate)).limit_denominator(100_000)
        self._time_base = 1 / self._rate
        self._stream = self._container.add_stream(codec, rate=self._rate)
        self._stream.width, self._stream.height = resolution
        self._stream.pix_fmt = "yuv420p"
        self._stream.codec_context.time_base = self._time_base
        self._resolution = resolution
        self._frame_index = 0
        self._closed = False

    def write(self, image: Image.Image) -> None:
        resized = image.convert("RGB").resize(
            self._resolution,
            resample=Image.Resampling.BILINEAR,
        )
        frame = self._av.VideoFrame.from_ndarray(
            np.asarray(resized),
            format="rgb24",
        )
        frame.pts = self._frame_index
        frame.time_base = self._time_base
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self._frame_index += 1

    def close(self) -> None:
        if self._closed:
            return
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._closed = True

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def close_video_writers(writers: Sequence[VideoWriter]) -> None:
    first_error = None
    for writer in writers:
        try:
            writer.close()
        except Exception as error:  # pragma: no cover - defensive close path
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _video_stream(container: Any) -> Any:
    if not container.streams.video:
        raise ValueError("Input file does not contain a video stream.")
    return container.streams.video[0]


def _stream_duration(stream: Any) -> float | None:
    if stream.duration is None:
        return None
    return float(stream.duration * stream.time_base)


def _even(value: int) -> int:
    if value < 2:
        return 2
    return value if value % 2 == 0 else value - 1
