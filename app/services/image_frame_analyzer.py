from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.services.frame_sampler import SampledFrame
from app.services.image_model_loader import ImageModelLoadResult, load_image_model_cached, preprocess_frame_for_image_model
from app.services.visual_forensics import analyze_visual_forensics
from app.services.watermark_analyzer import analyze_frame_watermark


FRAME_RISK_WEIGHTS = {
    "visible_watermark_risk": 0.30,
    "metadata_or_provenance_risk": 0.20,
    "forensic_risk": 0.20,
    "classifier_risk": 0.15,
    "gradcam_confidence_support": 0.05,
    "filename_context_risk": 0.05,
    "image_quality_artifact_risk": 0.05,
}


@dataclass
class FrameAnalysisResult:
    frame_id: int
    timestamp: float
    frame_url: str
    risk_score: float
    classifier: dict[str, Any]
    forensics: dict[str, Any]
    watermark: dict[str, Any]
    flags: list[str] = field(default_factory=list)
    contributions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "frame_url": self.frame_url,
            "risk_score": round(self.risk_score, 4),
            "classifier": self.classifier,
            "forensics": self.forensics,
            "watermark": self.watermark,
            "flags": self.flags,
            "contributions": {key: round(value, 4) for key, value in self.contributions.items()},
        }


def analyze_sampled_frames(
    frames: list[dict[str, Any] | SampledFrame],
    verification_id: str = "",
    image_model: ImageModelLoadResult | None = None,
    filename_context: str = "",
) -> dict[str, Any]:
    model_result = image_model or load_image_model_cached()
    results: list[FrameAnalysisResult] = []
    warnings: list[str] = []
    if model_result.warnings:
        warnings.extend(model_result.warnings)

    for frame in frames:
        frame_data = _frame_to_dict(frame)
        result = analyze_single_frame(frame_data, model_result, filename_context)
        results.append(result)

    risks = [item.risk_score for item in results]
    return {
        "checked": True,
        "frames_analyzed": len(results),
        "high_risk_frames": sum(1 for risk in risks if risk >= 0.6),
        "mean_frame_risk": float(np.mean(risks)) if risks else 0.0,
        "max_frame_risk": max(risks) if risks else 0.0,
        "top_suspicious_frames": [item.to_dict() for item in sorted(results, key=lambda x: x.risk_score, reverse=True)[:5]],
        "frames": [item.to_dict() for item in results],
        "warnings": _dedupe(warnings),
    }


def analyze_single_frame(
    frame: dict[str, Any],
    image_model: ImageModelLoadResult | None = None,
    filename_context: str = "",
) -> FrameAnalysisResult:
    model_result = image_model or load_image_model_cached()
    path = _resolve_frame_path(frame)
    classifier = _classify_frame(path, model_result)
    forensics = analyze_visual_forensics(path).to_dict()
    watermark = analyze_frame_watermark(path, filename_context=filename_context).to_dict()
    filename_risk = _filename_context_risk(filename_context)
    metadata_risk = 0.0

    signals = {
        "visible_watermark_risk": watermark.get("risk_score"),
        "metadata_or_provenance_risk": metadata_risk,
        "forensic_risk": forensics.get("risk_score"),
        "classifier_risk": classifier.get("classifier_risk"),
        "filename_context_risk": filename_risk,
        "image_quality_artifact_risk": forensics.get("quality_artifact_risk", forensics.get("risk_score")),
    }
    risk_score, contributions = _dynamic_weighted_score(signals)
    classifier["weighted_contribution"] = round(contributions.get("classifier_risk", 0.0), 4)

    flags = []
    flags.extend(forensics.get("flags", []))
    flags.extend(watermark.get("flags", []))
    if filename_risk > 0:
        flags.append("filename_context_risk")

    return FrameAnalysisResult(
        frame_id=int(frame.get("frame_id", 0)),
        timestamp=float(frame.get("timestamp", 0.0)),
        frame_url=str(frame.get("url") or frame.get("frame_url") or ""),
        risk_score=risk_score,
        classifier=classifier,
        forensics=forensics,
        watermark=watermark,
        flags=_dedupe(flags),
        contributions=contributions,
    )


def _classify_frame(path: Path, model_result: ImageModelLoadResult) -> dict[str, Any]:
    base = {
        "checked": False,
        "ai_probability": None,
        "classifier_risk": None,
        "weighted_contribution": 0.0,
        "warning": "Classifier is treated as a weak/moderate signal.",
    }
    if not model_result.checked or model_result.model is None:
        base["warnings"] = model_result.warnings
        return base

    try:
        import torch  # type: ignore
        tensor = preprocess_frame_for_image_model(
            path,
            image_size=model_result.image_size,
            mean=model_result.normalization_mean,
            std=model_result.normalization_std,
        )
        with torch.no_grad():
            output = model_result.model(torch.from_numpy(tensor).float())
            output_array = output.detach().cpu().numpy().reshape(-1)
        probability = _output_to_ai_probability(output_array)
        return {
            **base,
            "checked": True,
            "ai_probability": probability,
            "classifier_risk": probability,
            "raw_output": output_array.tolist(),
        }
    except Exception as exc:
        base["warnings"] = [f"Frame classifier inference failed: {exc}"]
        return base


def _output_to_ai_probability(output: np.ndarray) -> float:
    if output.size == 1:
        return float(1.0 / (1.0 + np.exp(-output[0])))
    exps = np.exp(output - np.max(output))
    probs = exps / np.sum(exps)
    return float(probs[-1])


def _dynamic_weighted_score(signals: dict[str, float | None]) -> tuple[float, dict[str, float]]:
    available = {key: float(value) for key, value in signals.items() if value is not None}
    total_weight = sum(FRAME_RISK_WEIGHTS[key] for key in available)
    if total_weight <= 0:
        return 0.0, {}
    contributions = {
        key: max(0.0, min(1.0, value)) * (FRAME_RISK_WEIGHTS[key] / total_weight)
        for key, value in available.items()
    }
    classifier_cap = FRAME_RISK_WEIGHTS["classifier_risk"]
    if contributions.get("classifier_risk", 0.0) > classifier_cap:
        contributions["classifier_risk"] = classifier_cap
    return min(1.0, sum(contributions.values())), contributions


def _filename_context_risk(filename: str) -> float:
    text = filename.lower()
    terms = ["ai", "generated", "deepfake", "runway", "pika", "sora", "synthesia", "heygen"]
    return 0.35 if any(term in text for term in terms) else 0.0


def _resolve_frame_path(frame: dict[str, Any]) -> Path:
    path = frame.get("path") or frame.get("frame_path")
    if path is None:
        return Path("")
    return Path(path)


def _frame_to_dict(frame: dict[str, Any] | SampledFrame) -> dict[str, Any]:
    return frame.to_dict() if hasattr(frame, "to_dict") else dict(frame)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
