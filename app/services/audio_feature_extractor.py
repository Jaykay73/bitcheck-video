from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "chroma_stft",
    "rms",
    "spectral_centroid",
    "spectral_bandwidth",
    "rolloff",
    "zero_crossing_rate",
    "mfcc1",
    "mfcc2",
    "mfcc3",
    "mfcc4",
    "mfcc5",
    "mfcc6",
    "mfcc7",
    "mfcc8",
    "mfcc9",
    "mfcc10",
    "mfcc11",
    "mfcc12",
    "mfcc13",
    "mfcc14",
    "mfcc15",
    "mfcc16",
    "mfcc17",
    "mfcc18",
    "mfcc19",
    "mfcc20",
]


@dataclass
class AudioFeatureResult:
    checked: bool
    features_found: bool
    feature_count: int
    feature_names: list[str]
    features: dict[str, float]
    dataframe: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "features_found": self.features_found,
            "feature_count": self.feature_count,
            "feature_names": self.feature_names,
            "features": self.features,
            "warnings": self.warnings,
        }


def extract_audio_features(
    audio_path: Path | str,
    sample_rate: int = 22050,
    max_duration_seconds: int = 5,
) -> AudioFeatureResult:
    warnings: list[str] = []

    try:
        import librosa  # type: ignore
    except ImportError:
        return AudioFeatureResult(
            checked=True,
            features_found=False,
            feature_count=0,
            feature_names=FEATURE_NAMES.copy(),
            features={},
            warnings=["librosa is not installed; audio feature extraction was skipped."],
        )

    try:
        samples, loaded_sr = librosa.load(
            str(audio_path),
            sr=sample_rate,
            mono=True,
            duration=max_duration_seconds,
        )
    except Exception as exc:
        return AudioFeatureResult(
            checked=True,
            features_found=False,
            feature_count=0,
            feature_names=FEATURE_NAMES.copy(),
            features={},
            warnings=[f"Audio feature extraction failed while loading audio: {exc}"],
        )

    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        features = _zero_features()
        dataframe = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
        return AudioFeatureResult(
            checked=True,
            features_found=False,
            feature_count=len(FEATURE_NAMES),
            feature_names=FEATURE_NAMES.copy(),
            features=features,
            dataframe=dataframe,
            warnings=["Audio file is empty; returning safe zero-valued features."],
        )

    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    rms_energy = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    if rms_energy == 0:
        warnings.append("Audio appears silent; features are safe but low-information.")
        features = _zero_features()
        dataframe = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
        return AudioFeatureResult(
            checked=True,
            features_found=True,
            feature_count=len(FEATURE_NAMES),
            feature_names=FEATURE_NAMES.copy(),
            features=features,
            dataframe=dataframe,
            warnings=warnings,
        )
    elif rms_energy < 1e-4:
        warnings.append("Audio has very low energy; model confidence may be limited.")

    try:
        features = _compute_features(librosa, samples, loaded_sr)
    except Exception as exc:
        features = _zero_features()
        warnings.append(f"Audio feature computation failed; returning safe zero-valued features: {exc}")

    features = _sanitize_features(features)
    dataframe = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)

    return AudioFeatureResult(
        checked=True,
        features_found=True,
        feature_count=len(FEATURE_NAMES),
        feature_names=FEATURE_NAMES.copy(),
        features=features,
        dataframe=dataframe,
        warnings=warnings,
    )


def _compute_features(librosa: Any, samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    stft_magnitude = np.abs(librosa.stft(samples))
    mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=20)

    features: dict[str, float] = {
        "chroma_stft": _mean(librosa.feature.chroma_stft(S=stft_magnitude, sr=sample_rate)),
        "rms": _mean(librosa.feature.rms(y=samples)),
        "spectral_centroid": _mean(librosa.feature.spectral_centroid(y=samples, sr=sample_rate)),
        "spectral_bandwidth": _mean(librosa.feature.spectral_bandwidth(y=samples, sr=sample_rate)),
        "rolloff": _mean(librosa.feature.spectral_rolloff(y=samples, sr=sample_rate)),
        "zero_crossing_rate": _mean(librosa.feature.zero_crossing_rate(y=samples)),
    }

    for index in range(20):
        features[f"mfcc{index + 1}"] = _mean(mfcc[index])

    return features


def _mean(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.nanmean(array))


def _zero_features() -> dict[str, float]:
    return {name: 0.0 for name in FEATURE_NAMES}


def _sanitize_features(features: dict[str, float]) -> dict[str, float]:
    sanitized: dict[str, float] = {}
    for name in FEATURE_NAMES:
        value = float(features.get(name, 0.0))
        if not math.isfinite(value):
            value = 0.0
        sanitized[name] = value
    return sanitized
