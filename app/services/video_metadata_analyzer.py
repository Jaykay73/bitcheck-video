from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings, settings
from app.utils.video_utils import (
    AI_VIDEO_TOOL_KEYWORDS,
    first_present,
    lower_joined_values,
    parse_fraction,
    safe_float,
    safe_int,
)


class VideoDurationError(ValueError):
    code = "video_too_long"

    def __init__(self, duration_seconds: float, max_duration_seconds: int) -> None:
        self.duration_seconds = duration_seconds
        self.max_duration_seconds = max_duration_seconds
        super().__init__(
            f"Demo currently supports videos up to {max_duration_seconds} seconds."
        )


@dataclass
class VideoMetadataResult:
    checked: bool
    duration_seconds: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    codec: str | None = None
    container_format: str | None = None
    bitrate: int | None = None
    audio_present: bool = False
    audio_codec: str | None = None
    creation_time: str | None = None
    encoder: str | None = None
    metadata_risk_score: float = 0.0
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    analyze_first_seconds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "codec": self.codec,
            "container_format": self.container_format,
            "bitrate": self.bitrate,
            "audio_present": self.audio_present,
            "audio_codec": self.audio_codec,
            "creation_time": self.creation_time,
            "encoder": self.encoder,
            "metadata_risk_score": round(self.metadata_risk_score, 4),
            "flags": self.flags,
            "warnings": self.warnings,
        }


def analyze_video_metadata(
    video_path: Path | str,
    allow_trim_to_5_seconds: bool | None = None,
    app_settings: Settings = settings,
) -> VideoMetadataResult:
    path = Path(video_path)
    allow_trim = (
        app_settings.allow_trim_to_5_seconds
        if allow_trim_to_5_seconds is None
        else allow_trim_to_5_seconds
    )

    result = _probe_with_ffmpeg(path)
    if result is None:
        result = _probe_with_opencv(path)

    if result is None:
        return VideoMetadataResult(
            checked=False,
            metadata_risk_score=0.02,
            warnings=["Video metadata could not be read; treating missing metadata as low risk."],
        )

    _score_metadata_risk(result)
    _enforce_duration_limit(result, allow_trim, app_settings.max_video_duration_seconds)
    return result


def _probe_with_ffmpeg(path: Path) -> VideoMetadataResult | None:
    try:
        import ffmpeg  # type: ignore
    except ImportError:
        return None

    try:
        probe = ffmpeg.probe(str(path))
    except Exception:
        return None

    return _metadata_from_ffprobe(probe)


def _metadata_from_ffprobe(probe: dict[str, Any]) -> VideoMetadataResult:
    format_info = probe.get("format") or {}
    streams = probe.get("streams") or []
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})

    format_tags = format_info.get("tags") or {}
    video_tags = video_stream.get("tags") or {}
    encoder = first_present(
        format_tags.get("encoder"),
        format_tags.get("software"),
        video_tags.get("encoder"),
        video_tags.get("software"),
    )

    duration = first_present(video_stream.get("duration"), format_info.get("duration"))
    bitrate = first_present(format_info.get("bit_rate"), video_stream.get("bit_rate"))

    return VideoMetadataResult(
        checked=True,
        duration_seconds=safe_float(duration),
        fps=parse_fraction(first_present(video_stream.get("avg_frame_rate"), video_stream.get("r_frame_rate"))),
        width=safe_int(video_stream.get("width")),
        height=safe_int(video_stream.get("height")),
        frame_count=safe_int(video_stream.get("nb_frames")),
        codec=video_stream.get("codec_name"),
        container_format=format_info.get("format_name"),
        bitrate=safe_int(bitrate),
        audio_present=bool(audio_stream),
        audio_codec=audio_stream.get("codec_name"),
        creation_time=first_present(format_tags.get("creation_time"), video_tags.get("creation_time")),
        encoder=encoder,
        warnings=[] if duration is not None else ["Duration metadata is missing."],
    )


def _probe_with_opencv(path: Path) -> VideoMetadataResult | None:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None

        fps = capture.get(cv2.CAP_PROP_FPS) or None
        frame_count = safe_int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = safe_int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = safe_int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = None
        if fps and frame_count:
            duration = frame_count / fps

        warnings = ["Metadata read with OpenCV fallback; container/audio fields may be unavailable."]
        if duration is None:
            warnings.append("Duration metadata is missing.")

        return VideoMetadataResult(
            checked=True,
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            audio_present=False,
            metadata_risk_score=0.02,
            warnings=warnings,
        )
    finally:
        capture.release()


def _enforce_duration_limit(
    result: VideoMetadataResult,
    allow_trim: bool,
    max_duration_seconds: int,
) -> None:
    if result.duration_seconds is None or result.duration_seconds <= max_duration_seconds:
        return

    if allow_trim:
        result.analyze_first_seconds = max_duration_seconds
        result.warnings.append(
            f"Video is longer than {max_duration_seconds} seconds; only the first {max_duration_seconds} seconds will be analyzed."
        )
        result.flags.append("trim_to_demo_limit")
        return

    raise VideoDurationError(result.duration_seconds, max_duration_seconds)


def _score_metadata_risk(result: VideoMetadataResult) -> None:
    score = 0.0
    flags: list[str] = []

    combined = lower_joined_values(result.encoder, result.codec, result.container_format)
    if any(keyword in combined for keyword in AI_VIDEO_TOOL_KEYWORDS):
        score += 0.35
        flags.append("ai_video_tool_metadata_keyword")

    if result.encoder and any(token in result.encoder.lower() for token in ("lavf", "ffmpeg", "handbrake")):
        score += 0.04
        flags.append("possible_reencoded_video")

    if result.fps is not None and (result.fps < 10 or result.fps > 120):
        score += 0.06
        flags.append("unusual_fps")

    if result.width is not None and result.height is not None:
        if result.width < 160 or result.height < 120:
            score += 0.04
            flags.append("very_low_resolution")
        if result.width > 4096 or result.height > 4096:
            score += 0.05
            flags.append("unusually_high_resolution")
    else:
        score += 0.02
        result.warnings.append("Resolution metadata is missing.")

    if result.codec and result.codec.lower() not in {"h264", "hevc", "h265", "vp8", "vp9", "av1", "mpeg4"}:
        score += 0.04
        flags.append("uncommon_video_codec")

    if result.duration_seconds is None:
        score += 0.02

    result.metadata_risk_score = min(score, 1.0)
    result.flags.extend(flag for flag in flags if flag not in result.flags)
