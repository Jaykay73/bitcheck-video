from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import wave

from app.config import Settings, settings
from app.services.video_metadata_analyzer import VideoMetadataResult


DEFAULT_AUDIO_SAMPLE_RATE = 22050


@dataclass
class AudioExtractionResult:
    checked: bool
    status: str
    audio_present: bool
    audio_path: str | None = None
    duration_seconds: float | None = None
    sample_rate: int | None = None
    risk_score: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "checked": self.checked,
            "status": self.status,
            "audio_present": self.audio_present,
            "warnings": self.warnings,
        }
        if self.status == "no_audio":
            payload["risk_score"] = self.risk_score
            return payload

        payload.update(
            {
                "audio_path": self.audio_path,
                "duration_seconds": self.duration_seconds,
                "sample_rate": self.sample_rate,
            }
        )
        return payload


def extract_video_audio(
    video_path: Path | str,
    verification_id: str,
    metadata: VideoMetadataResult | dict[str, Any],
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    max_seconds: int | None = None,
    app_settings: Settings = settings,
) -> AudioExtractionResult:
    audio_present = _metadata_audio_present(metadata)
    if not audio_present:
        return AudioExtractionResult(
            checked=True,
            status="no_audio",
            audio_present=False,
            risk_score=None,
            warnings=["No audio stream was found in the video."],
        )

    output_path = app_settings.outputs_dir / f"{verification_id}_audio.wav"
    app_settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    limit_seconds = max_seconds or app_settings.max_video_duration_seconds

    warnings: list[str] = []
    extracted = _extract_with_ffmpeg_python(
        Path(video_path),
        output_path,
        sample_rate=sample_rate,
        max_seconds=limit_seconds,
        warnings=warnings,
    )
    if not extracted:
        extracted = _extract_with_moviepy(
            Path(video_path),
            output_path,
            sample_rate=sample_rate,
            max_seconds=limit_seconds,
            warnings=warnings,
        )

    if not extracted or not output_path.exists():
        if output_path.exists():
            output_path.unlink()
        if not warnings:
            warnings.append("Audio extraction failed for an unknown reason.")
        return AudioExtractionResult(
            checked=True,
            status="audio_extraction_failed",
            audio_present=True,
            audio_path=None,
            duration_seconds=None,
            sample_rate=sample_rate,
            warnings=warnings,
        )

    duration_seconds, actual_sample_rate = _read_wav_metadata(output_path)
    relative_path = f"outputs/{output_path.name}"
    return AudioExtractionResult(
        checked=True,
        status="audio_extracted",
        audio_present=True,
        audio_path=relative_path,
        duration_seconds=duration_seconds,
        sample_rate=actual_sample_rate or sample_rate,
        warnings=warnings,
    )


def _metadata_audio_present(metadata: VideoMetadataResult | dict[str, Any]) -> bool:
    if isinstance(metadata, VideoMetadataResult):
        return bool(metadata.audio_present)
    return bool(metadata.get("audio_present"))


def _extract_with_ffmpeg_python(
    video_path: Path,
    output_path: Path,
    sample_rate: int,
    max_seconds: int,
    warnings: list[str],
) -> bool:
    try:
        import ffmpeg  # type: ignore
    except ImportError:
        warnings.append("ffmpeg-python is not installed; trying moviepy fallback.")
        return False

    try:
        stream = ffmpeg.input(str(video_path), t=max_seconds)
        stream = ffmpeg.output(
            stream.audio,
            str(output_path),
            format="wav",
            acodec="pcm_s16le",
            ac=1,
            ar=sample_rate,
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as exc:
        warnings.append(f"ffmpeg audio extraction failed: {exc}")
        return False


def _extract_with_moviepy(
    video_path: Path,
    output_path: Path,
    sample_rate: int,
    max_seconds: int,
    warnings: list[str],
) -> bool:
    try:
        from moviepy import VideoFileClip  # type: ignore
    except ImportError:
        try:
            from moviepy.editor import VideoFileClip  # type: ignore
        except ImportError:
            warnings.append("moviepy is not installed; audio extraction could not run.")
            return False

    clip = None
    audio_clip = None
    try:
        clip = VideoFileClip(str(video_path))
        if clip.audio is None:
            warnings.append("Metadata indicated audio, but no readable audio track was found.")
            return False
        audio_clip = clip.audio.subclip(0, min(max_seconds, clip.duration or max_seconds))
        audio_clip.write_audiofile(
            str(output_path),
            fps=sample_rate,
            nbytes=2,
            codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"],
            logger=None,
        )
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as exc:
        warnings.append(f"moviepy audio extraction failed: {exc}")
        return False
    finally:
        if audio_clip is not None:
            audio_clip.close()
        if clip is not None:
            clip.close()


def _read_wav_metadata(path: Path) -> tuple[float | None, int | None]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frame_count / sample_rate if sample_rate else None
            return duration, sample_rate
    except Exception:
        return None, None
