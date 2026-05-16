from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import Settings, settings


PREFERRED_AUDIO_MODEL_PATH = Path("models/BitcheckDeepfake.joblib")
FALLBACK_AUDIO_MODEL_PATHS = [
    Path("models/BitcheckDeepFake"),
    Path("models/BitcheckDeepFake.joblib"),
    Path("models/BitcheckDeepfake"),
]


@dataclass
class AudioModelLoadResult:
    checked: bool
    model_found: bool
    model: Any | None = None
    model_path: Path | None = None
    model_type: str | None = None
    warnings: list[str] = field(default_factory=list)


def resolve_audio_model_path(app_settings: Settings = settings) -> Path | None:
    candidates = [app_settings.expected_audio_model_path]
    candidates.extend(app_settings.base_dir / path for path in FALLBACK_AUDIO_MODEL_PATHS)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=1)
def load_audio_model_cached() -> AudioModelLoadResult:
    return load_audio_model()


def load_audio_model(
    model_path: Path | str | None = None,
    app_settings: Settings = settings,
) -> AudioModelLoadResult:
    resolved_path = Path(model_path) if model_path is not None else resolve_audio_model_path(app_settings)
    preferred_display_path = str(PREFERRED_AUDIO_MODEL_PATH)

    if resolved_path is None or not resolved_path.exists():
        return AudioModelLoadResult(
            checked=False,
            model_found=False,
            warnings=[f"Audio model file was not found at {preferred_display_path}."],
        )

    try:
        import joblib
    except ImportError:
        return AudioModelLoadResult(
            checked=False,
            model_found=True,
            model_path=resolved_path,
            warnings=["joblib is not installed; audio model could not be loaded."],
        )

    try:
        model = joblib.load(resolved_path)
    except Exception as exc:
        return AudioModelLoadResult(
            checked=False,
            model_found=True,
            model_path=resolved_path,
            warnings=[f"Audio model could not be loaded: {exc}"],
        )

    if not hasattr(model, "predict"):
        return AudioModelLoadResult(
            checked=False,
            model_found=True,
            model=model,
            model_path=resolved_path,
            model_type=type(model).__name__,
            warnings=["Loaded audio model does not expose a predict method."],
        )

    warnings: list[str] = []
    if resolved_path.name != PREFERRED_AUDIO_MODEL_PATH.name:
        warnings.append(
            f"Using fallback audio model path {resolved_path}; preferred path is {preferred_display_path}."
        )

    return AudioModelLoadResult(
        checked=True,
        model_found=True,
        model=model,
        model_path=resolved_path,
        model_type=type(model).__name__,
        warnings=warnings,
    )
