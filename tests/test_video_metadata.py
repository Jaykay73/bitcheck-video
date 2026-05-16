from pathlib import Path

import pytest

from app.config import Settings
from app.services import video_metadata_analyzer
from app.services.video_metadata_analyzer import (
    VideoDurationError,
    VideoMetadataResult,
    analyze_video_metadata,
)


class MetadataSettings(Settings):
    def __init__(self, max_duration_seconds: int = 5) -> None:
        self.max_video_duration_seconds = max_duration_seconds
        self.allow_trim_to_5_seconds = False


def test_metadata_analyzer_handles_invalid_file_gracefully(tmp_path: Path) -> None:
    invalid_video = tmp_path / "invalid.mp4"
    invalid_video.write_bytes(b"not a real video")

    result = analyze_video_metadata(invalid_video, app_settings=MetadataSettings())

    assert result.checked is False
    assert result.metadata_risk_score < 0.1
    assert result.warnings


def test_duration_limit_returns_video_too_long(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "long.mp4"
    video.write_bytes(b"placeholder")

    monkeypatch.setattr(
        video_metadata_analyzer,
        "_probe_with_ffmpeg",
        lambda path: VideoMetadataResult(checked=True, duration_seconds=5.2, fps=30.0),
    )
    monkeypatch.setattr(video_metadata_analyzer, "_probe_with_opencv", lambda path: None)

    with pytest.raises(VideoDurationError) as exc:
        analyze_video_metadata(video, allow_trim_to_5_seconds=False, app_settings=MetadataSettings())

    assert exc.value.code == "video_too_long"
    assert exc.value.duration_seconds == 5.2


def test_duration_limit_can_mark_trim_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "long.mp4"
    video.write_bytes(b"placeholder")

    monkeypatch.setattr(
        video_metadata_analyzer,
        "_probe_with_ffmpeg",
        lambda path: VideoMetadataResult(checked=True, duration_seconds=8.0, fps=30.0),
    )
    monkeypatch.setattr(video_metadata_analyzer, "_probe_with_opencv", lambda path: None)

    result = analyze_video_metadata(video, allow_trim_to_5_seconds=True, app_settings=MetadataSettings())

    assert result.analyze_first_seconds == 5
    assert "trim_to_demo_limit" in result.flags
    assert result.warnings


def test_missing_metadata_does_not_crash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "missing.mp4"
    video.write_bytes(b"placeholder")

    monkeypatch.setattr(
        video_metadata_analyzer,
        "_probe_with_ffmpeg",
        lambda path: VideoMetadataResult(checked=True),
    )
    monkeypatch.setattr(video_metadata_analyzer, "_probe_with_opencv", lambda path: None)

    result = analyze_video_metadata(video, app_settings=MetadataSettings())

    assert result.checked is True
    assert result.duration_seconds is None
    assert result.metadata_risk_score < 0.1
