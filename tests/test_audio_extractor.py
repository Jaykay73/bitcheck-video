from pathlib import Path
import wave

from app.config import Settings
from app.services import audio_extractor
from app.services.audio_extractor import extract_video_audio
from app.services.video_metadata_analyzer import VideoMetadataResult


class AudioSettings(Settings):
    def __init__(self, output_dir: Path) -> None:
        self.outputs_dir = output_dir
        self.max_video_duration_seconds = 5


def write_tiny_wav(path: Path, sample_rate: int = 22050) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * sample_rate)


def test_no_audio_video_returns_no_audio(tmp_path: Path) -> None:
    result = extract_video_audio(
        tmp_path / "video.mp4",
        verification_id="noaudio",
        metadata=VideoMetadataResult(checked=True, audio_present=False),
        app_settings=AudioSettings(tmp_path / "outputs"),
    )

    assert result.checked is True
    assert result.status == "no_audio"
    assert result.audio_present is False
    assert result.risk_score is None
    assert result.warnings == ["No audio stream was found in the video."]


def test_bad_video_does_not_crash(tmp_path: Path) -> None:
    bad_video = tmp_path / "bad.mp4"
    bad_video.write_bytes(b"not a valid video")

    result = extract_video_audio(
        bad_video,
        verification_id="bad",
        metadata=VideoMetadataResult(checked=True, audio_present=True),
        app_settings=AudioSettings(tmp_path / "outputs"),
    )

    assert result.checked is True
    assert result.status == "audio_extraction_failed"
    assert result.audio_present is True
    assert result.audio_path is None
    assert result.warnings


def test_extraction_function_returns_structured_output(monkeypatch, tmp_path: Path) -> None:
    def fake_ffmpeg_extract(video_path, output_path, sample_rate, max_seconds, warnings):
        write_tiny_wav(output_path, sample_rate=sample_rate)
        return True

    monkeypatch.setattr(audio_extractor, "_extract_with_ffmpeg_python", fake_ffmpeg_extract)
    monkeypatch.setattr(audio_extractor, "_extract_with_moviepy", lambda *args, **kwargs: False)

    result = extract_video_audio(
        tmp_path / "video.mp4",
        verification_id="audio",
        metadata=VideoMetadataResult(checked=True, audio_present=True),
        app_settings=AudioSettings(tmp_path / "outputs"),
    )

    assert result.checked is True
    assert result.status == "audio_extracted"
    assert result.audio_present is True
    assert result.audio_path == "outputs/audio_audio.wav"
    assert result.duration_seconds == 1.0
    assert result.sample_rate == 22050
