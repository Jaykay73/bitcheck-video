from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.services.audio_feature_extractor import FEATURE_NAMES, AudioFeatureResult
from app.services.audio_model_loader import AudioModelLoadResult, load_audio_model_cached


FAKE_STRING_LABELS = {"fake", "ai", "ai_generated", "ai-generated", "deepfake", "synthetic"}
REAL_STRING_LABELS = {"real", "human", "authentic", "genuine"}


@dataclass
class AudioAnalysisResult:
    checked: bool
    model_found: bool
    model_type: str | None = None
    feature_count: int | None = None
    predicted_label: str | None = None
    fake_probability: float | None = None
    real_probability: float | None = None
    risk_score: float | None = None
    class_mapping: dict[str, str] = field(default_factory=dict)
    feature_importance_top: list[dict[str, float | str]] = field(default_factory=list)
    raw_probabilities: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "checked": self.checked,
            "model_found": self.model_found,
            "risk_score": self.risk_score,
            "warnings": self.warnings,
        }
        if not self.model_found:
            return payload

        payload.update(
            {
                "model_type": self.model_type,
                "feature_count": self.feature_count,
                "predicted_label": self.predicted_label,
                "fake_probability": self.fake_probability,
                "real_probability": self.real_probability,
                "class_mapping": self.class_mapping,
                "feature_importance_top": self.feature_importance_top,
            }
        )
        if self.raw_probabilities:
            payload["raw_probabilities"] = self.raw_probabilities
        return payload


def analyze_audio_deepfake(
    features: pd.DataFrame | AudioFeatureResult,
    model_result: AudioModelLoadResult | None = None,
) -> AudioAnalysisResult:
    load_result = model_result or load_audio_model_cached()
    if not load_result.model_found:
        return AudioAnalysisResult(
            checked=False,
            model_found=False,
            risk_score=None,
            warnings=load_result.warnings.copy(),
        )
    if not load_result.checked or load_result.model is None:
        return AudioAnalysisResult(
            checked=False,
            model_found=True,
            model_type=load_result.model_type,
            risk_score=None,
            warnings=load_result.warnings.copy(),
        )

    dataframe = _coerce_features_to_dataframe(features)
    warnings = load_result.warnings.copy()
    model = load_result.model

    try:
        prediction = model.predict(dataframe)[0]
    except Exception as exc:
        return AudioAnalysisResult(
            checked=False,
            model_found=True,
            model_type=load_result.model_type,
            feature_count=len(dataframe.columns),
            risk_score=None,
            warnings=[*warnings, f"Audio model prediction failed: {exc}"],
        )

    class_mapping = _infer_class_mapping(getattr(model, "classes_", None))
    raw_probabilities: dict[str, float] = {}
    fake_probability: float | None = None
    real_probability: float | None = None

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(dataframe)[0]
            classes = list(getattr(model, "classes_", range(len(probabilities))))
            raw_probabilities = {
                str(class_label): _safe_probability(probability)
                for class_label, probability in zip(classes, probabilities, strict=False)
            }
            fake_probability, real_probability = _mapped_probabilities(
                classes,
                probabilities,
                class_mapping,
                warnings,
            )
        except Exception as exc:
            warnings.append(f"Audio model predict_proba failed: {exc}")
    else:
        warnings.append("Audio model does not expose predict_proba; probability outputs are unavailable.")

    predicted_label = _normalize_prediction_label(prediction, class_mapping)
    risk_score = fake_probability
    if risk_score is None and predicted_label == "fake":
        risk_score = 1.0
    elif risk_score is None and predicted_label == "real":
        risk_score = 0.0

    return AudioAnalysisResult(
        checked=True,
        model_found=True,
        model_type=load_result.model_type,
        feature_count=len(dataframe.columns),
        predicted_label=predicted_label,
        fake_probability=fake_probability,
        real_probability=real_probability,
        risk_score=risk_score,
        class_mapping=class_mapping,
        feature_importance_top=_top_feature_importances(model, list(dataframe.columns)),
        raw_probabilities=raw_probabilities,
        warnings=warnings,
    )


def _coerce_features_to_dataframe(features: pd.DataFrame | AudioFeatureResult) -> pd.DataFrame:
    if isinstance(features, AudioFeatureResult):
        if features.dataframe is not None:
            return features.dataframe[FEATURE_NAMES]
        return pd.DataFrame([[features.features.get(name, 0.0) for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    return features[FEATURE_NAMES] if all(name in features.columns for name in FEATURE_NAMES) else features


def _infer_class_mapping(classes: Any) -> dict[str, str]:
    if classes is None:
        return {}

    class_list = list(classes)
    if len(class_list) == 2 and class_list == [0, 1]:
        return {"0": "fake", "1": "real"}

    mapping: dict[str, str] = {}
    for label in class_list:
        normalized = str(label).strip().lower()
        if normalized in FAKE_STRING_LABELS:
            mapping[str(label)] = "fake"
        elif normalized in REAL_STRING_LABELS:
            mapping[str(label)] = "real"
    return mapping


def _mapped_probabilities(
    classes: list[Any],
    probabilities: Any,
    class_mapping: dict[str, str],
    warnings: list[str],
) -> tuple[float | None, float | None]:
    fake_probability = None
    real_probability = None

    for class_label, probability in zip(classes, probabilities, strict=False):
        mapped_label = class_mapping.get(str(class_label))
        if mapped_label == "fake":
            fake_probability = _safe_probability(probability)
        elif mapped_label == "real":
            real_probability = _safe_probability(probability)

    if fake_probability is None or real_probability is None:
        warnings.append("Audio model class mapping is unclear; raw probabilities are exposed safely.")

    return fake_probability, real_probability


def _normalize_prediction_label(prediction: Any, class_mapping: dict[str, str]) -> str | None:
    mapped = class_mapping.get(str(prediction))
    if mapped:
        return mapped
    normalized = str(prediction).strip().lower()
    if normalized in FAKE_STRING_LABELS:
        return "fake"
    if normalized in REAL_STRING_LABELS:
        return "real"
    return str(prediction)


def _top_feature_importances(model: Any, feature_names: list[str], limit: int = 5) -> list[dict[str, float | str]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    pairs = [
        (feature, float(importance))
        for feature, importance in zip(feature_names, importances, strict=False)
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [
        {"feature": feature, "importance": round(importance, 6)}
        for feature, importance in pairs[:limit]
    ]


def _safe_probability(probability: Any) -> float:
    value = float(probability)
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
