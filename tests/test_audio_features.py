from pathlib import Path
import math
import wave

import numpy as np

from app.services.audio_feature_extractor import FEATURE_NAMES, extract_audio_features


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = 22050) -> None:
    samples = np.asarray(samples, dtype=np.float32)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def test_feature_extractor_returns_26_features(tmp_path: Path) -> None:
    sample_rate = 22050
    seconds = 1.0
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    samples = 0.2 * np.sin(2 * np.pi * 440 * t)
    audio_path = tmp_path / "tone.wav"
    write_wav(audio_path, samples, sample_rate=sample_rate)

    result = extract_audio_features(audio_path)

    assert result.checked is True
    assert result.features_found is True
    assert result.feature_count == 26
    assert len(result.features) == 26


def test_columns_match_expected_order(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    write_wav(audio_path, np.ones(22050, dtype=np.float32) * 0.05)

    result = extract_audio_features(audio_path)

    assert result.feature_names == FEATURE_NAMES
    assert result.dataframe is not None
    assert list(result.dataframe.columns) == FEATURE_NAMES


def test_silent_audio_does_not_crash(tmp_path: Path) -> None:
    audio_path = tmp_path / "silent.wav"
    write_wav(audio_path, np.zeros(22050, dtype=np.float32))

    result = extract_audio_features(audio_path)

    assert result.checked is True
    assert result.feature_count == 26
    assert result.features_found is True
    assert result.warnings


def test_nan_or_inf_not_returned(tmp_path: Path) -> None:
    audio_path = tmp_path / "silent.wav"
    write_wav(audio_path, np.zeros(22050, dtype=np.float32))

    result = extract_audio_features(audio_path)

    assert all(math.isfinite(value) for value in result.features.values())
