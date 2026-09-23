from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vision_lens.config import AlphaFormat, VideoConfig
from vision_lens.config.schema import ALPHA_FORMAT_EXTENSIONS


@dataclass(frozen=True)
class AlphaVideoEncoding:
    extension: str
    codec: str
    pixel_format: str
    codec_options: Mapping[str, str]


ALPHA_VIDEO_ENCODINGS: dict[AlphaFormat, AlphaVideoEncoding] = {
    "prores_4444": AlphaVideoEncoding(
        extension=ALPHA_FORMAT_EXTENSIONS["prores_4444"],
        codec="prores_ks",
        pixel_format="yuva444p10le",
        codec_options={"profile": "4444"},
    ),
    "vp9": AlphaVideoEncoding(
        extension=ALPHA_FORMAT_EXTENSIONS["vp9"],
        codec="libvpx-vp9",
        pixel_format="yuva420p",
        codec_options={"crf": "18", "b": "0", "auto-alt-ref": "0"},
    ),
}


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
            "Video support requires the optional 'av' dependency. "
            'Install it with `pip install "vision-lens[video]"`.'
        ) from error
    return av


def require_video_encoder(
    codec: str,
    *,
    field_name: str,
    pixel_format: str | None = None,
) -> None:
    av = require_video_dependencies()
    try:
        encoder = av.Codec(codec, "w")
    except (ValueError, LookupError) as error:
        raise RuntimeError(
            f"{field_name} requires unavailable video encoder {codec!r}."
        ) from error
    if pixel_format is None:
        return
    supported_formats = {
        video_format.name for video_format in (encoder.video_formats or ())
    }
    if pixel_format not in supported_formats:
        raise RuntimeError(
            f"{field_name} requires {codec!r} to support pixel format {pixel_format!r}."
        )


def probe_video(path: str | Path) -> VideoMetadata:
    av = require_video_dependencies()
    source = Path(path)
    with av.open(str(source)) as container:
        stream = _video_stream(container)
        duration = _stream_duration(stream)
        if duration is None and container.duration is not None:
            duration = float(container.duration / av.time_base)
        source_rate = _source_frame_rate(stream)
        return VideoMetadata(
            path=source,
            width=int(stream.codec_context.width),
            height=int(stream.codec_context.height),
            duration=duration,
            source_frame_rate=source_rate,
        )


def resolve_sampling_rate(
    configured: float | str,
    source_frame_rate: float | None,
) -> float:
    if configured != "auto":
        return float(configured)
    if (
        source_frame_rate is None
        or not isfinite(source_frame_rate)
        or source_frame_rate <= 0
    ):
        raise ValueError(
            "video.sampling_rate=auto requires a valid source video FPS; "
            "set video.sampling_rate to a positive number instead."
        )
    return source_frame_rate


def iter_sampled_frames(
    path: str | Path,
    config: VideoConfig,
) -> Iterator[SampledVideoFrame]:
    av = require_video_dependencies()
    source = Path(path)
    with av.open(str(source)) as container:
        stream = _video_stream(container)
        source_rate = _source_frame_rate(stream)
        sampling_rate = resolve_sampling_rate(config.sampling_rate, source_rate)
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
                next_timestamp = config.start_time + (output_index / sampling_rate)


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
    sampling_rate = resolve_sampling_rate(
        config.sampling_rate, metadata.source_frame_rate
    )
    end_time = config.end_time
    if metadata.duration is not None:
        end_time = (
            metadata.duration if end_time is None else min(end_time, metadata.duration)
        )
    if end_time is None:
        return config.frame_limit
    duration = max(0.0, end_time - config.start_time)
    count = int(np.ceil(duration * sampling_rate - 1e-9))
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
    default: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if configured is not None:
        return configured
    if default is not None:
        return (_even(default[0]), _even(default[1]))
    return (_even(metadata.width), _even(metadata.height))


class VideoWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        frame_rate: float,
        resolution: tuple[int, int],
        codec: str,
        pixel_format: str = "yuv420p",
        preserve_alpha: bool = False,
        codec_options: Mapping[str, str] | None = None,
    ) -> None:
        av = require_video_dependencies()
        self._av = av
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(self.path), mode="w")
        self._rate = Fraction(str(frame_rate)).limit_denominator(100_000)
        self._time_base = 1 / self._rate
        self._stream = self._container.add_stream(
            codec,
            rate=self._rate,
            options=dict(codec_options or {}),
        )
        self._stream.width, self._stream.height = resolution
        self._stream.pix_fmt = pixel_format
        self._stream.codec_context.time_base = self._time_base
        self._resolution = resolution
        self._preserve_alpha = preserve_alpha
        self._frame_index = 0
        self._closed = False

    def write(self, image: Image.Image) -> None:
        mode = "RGBA" if self._preserve_alpha else "RGB"
        frame_format = "rgba" if self._preserve_alpha else "rgb24"
        resized = image if image.mode == mode else image.convert(mode)
        if resized.size != self._resolution:
            resized = resized.resize(
                self._resolution,
                resample=Image.Resampling.BILINEAR,
            )
        frame = self._av.VideoFrame.from_ndarray(
            np.asarray(resized),
            format=frame_format,
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


def _source_frame_rate(stream: Any) -> float | None:
    if stream.average_rate is None:
        return None
    rate = float(stream.average_rate)
    return rate if isfinite(rate) and rate > 0 else None


def _stream_duration(stream: Any) -> float | None:
    if stream.duration is None:
        return None
    return float(stream.duration * stream.time_base)


def _even(value: int) -> int:
    if value < 2:
        return 2
    return value if value % 2 == 0 else value - 1
