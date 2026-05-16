from __future__ import annotations

from typing import Any

import numpy as np

from app.services.trust_scorer import cap_maximum_risk_level, enforce_minimum_risk_level, score_trust


VIDEO_RISK_WEIGHTS = {
    "audio_risk": 0.30,
    "visual_multisignal_risk": 0.25,
    "watermark_provenance_risk": 0.18,
    "temporal_consistency_risk": 0.12,
    "video_metadata_risk": 0.10,
    "filename_risk": 0.05,
}


def aggregate_video_risk(
    audio_analysis: dict[str, Any] | None = None,
    visual_analysis: dict[str, Any] | None = None,
    temporal_analysis: dict[str, Any] | None = None,
    video_metadata: dict[str, Any] | None = None,
    filename_risk: float | None = None,
) -> dict[str, Any]:
    audio_analysis = audio_analysis or {}
    visual_analysis = visual_analysis or {}
    temporal_analysis = temporal_analysis or {}
    video_metadata = video_metadata or {}

    frames = list(visual_analysis.get("frames") or visual_analysis.get("top_suspicious_frames") or [])
    visual_multisignal_risk = _visual_multisignal_risk(visual_analysis, temporal_analysis)
    watermark_risk = _watermark_provenance_risk(frames, video_metadata)

    signals = {
        "audio_risk": _numeric_or_none(audio_analysis.get("risk_score")),
        "visual_multisignal_risk": visual_multisignal_risk,
        "watermark_provenance_risk": watermark_risk,
        "temporal_consistency_risk": _numeric_or_none(temporal_analysis.get("temporal_consistency_risk")),
        "video_metadata_risk": _numeric_or_none(video_metadata.get("metadata_risk_score")),
        "filename_risk": _numeric_or_none(filename_risk),
    }
    risk_score, contributions = _weighted_score(signals)
    no_evidence = not any((value or 0.0) > 0 for value in signals.values() if value is not None)
    if no_evidence:
        risk_score = 0.50

    score = score_trust(risk_score)
    override_flags: list[str] = []

    if _multiple_visible_watermarks(frames):
        score = enforce_minimum_risk_level(score, "High Risk")
        override_flags.append("Visible AI watermark-like signals appeared in multiple sampled frames.")

    if _metadata_indicates_ai(video_metadata):
        score = enforce_minimum_risk_level(score, "High Risk")
        override_flags.append("Metadata or provenance fields contain AI-generation indicators.")

    fake_probability = _numeric_or_none(audio_analysis.get("fake_probability"))
    if fake_probability is not None and fake_probability >= 0.85 and visual_multisignal_risk is not None and visual_multisignal_risk >= 0.55:
        score = enforce_minimum_risk_level(score, "High Risk")
        override_flags.append("Audio and visual signals agree on elevated synthetic-media risk.")

    if _classifier_only_high(frames, signals):
        score = enforce_minimum_risk_level(score, "Suspicious")
        score = cap_maximum_risk_level(score, "Suspicious")
        override_flags.append("Image classifier risk was high without enough supporting signals, so the decision was capped at Suspicious.")

    if _only_filename_signal(signals):
        score = cap_maximum_risk_level(score, "Suspicious")
        override_flags.append("Filename context was the only suspicious signal.")

    if signals["audio_risk"] is None and (visual_multisignal_risk or 0.0) < 0.40:
        score = enforce_minimum_risk_level(score, "Suspicious")
        override_flags.append("No audio signal was available and visual evidence was weak, so human review is recommended.")

    summary = _summary(score, audio_analysis, visual_multisignal_risk, temporal_analysis, override_flags)
    return {
        **score,
        "summary": summary,
        "signal_scores": {key: value for key, value in signals.items() if value is not None},
        "weighted_contributions": contributions,
        "visual_multisignal_risk": visual_multisignal_risk,
        "flags": override_flags,
    }


def _visual_multisignal_risk(visual_analysis: dict[str, Any], temporal_analysis: dict[str, Any]) -> float | None:
    if not visual_analysis and not temporal_analysis:
        return None
    top_mean = _numeric_or_zero(temporal_analysis.get("top_20_percent_frame_risk_mean", visual_analysis.get("max_frame_risk")))
    mean = _numeric_or_zero(temporal_analysis.get("mean_frame_risk", visual_analysis.get("mean_frame_risk")))
    suspicious_ratio = _numeric_or_zero(temporal_analysis.get("suspicious_frame_ratio"))
    max_risk = _numeric_or_zero(temporal_analysis.get("max_frame_risk", visual_analysis.get("max_frame_risk")))
    repeated = _numeric_or_zero(temporal_analysis.get("repeated_signal_consistency"))
    return _clamp(
        0.35 * top_mean
        + 0.25 * mean
        + 0.20 * suspicious_ratio
        + 0.10 * max_risk
        + 0.10 * repeated
    )


def _weighted_score(signals: dict[str, float | None]) -> tuple[float, dict[str, float]]:
    available = {key: value for key, value in signals.items() if value is not None}
    if not available:
        return 0.50, {}
    total_weight = sum(VIDEO_RISK_WEIGHTS[key] for key in available)
    if total_weight <= 0:
        return 0.50, {}
    contributions = {
        key: _clamp(value) * (VIDEO_RISK_WEIGHTS[key] / total_weight)
        for key, value in available.items()
    }
    return _clamp(sum(contributions.values())), {key: round(value, 4) for key, value in contributions.items()}


def _watermark_provenance_risk(frames: list[dict[str, Any]], metadata: dict[str, Any]) -> float | None:
    frame_scores = [
        _numeric_or_zero((frame.get("watermark") or {}).get("risk_score"))
        for frame in frames
        if frame.get("watermark") is not None
    ]
    metadata_risk = 0.85 if _metadata_indicates_ai(metadata) else 0.0
    if not frame_scores and metadata_risk == 0.0:
        return None
    return max(frame_scores + [metadata_risk])


def _multiple_visible_watermarks(frames: list[dict[str, Any]]) -> bool:
    count = 0
    for frame in frames:
        watermark = frame.get("watermark") or {}
        flags = [str(flag).lower() for flag in watermark.get("flags", [])]
        if watermark.get("possible_watermark_found") or any("watermark" in flag or "generator" in flag for flag in flags):
            count += 1
    return count >= 2


def _metadata_indicates_ai(metadata: dict[str, Any]) -> bool:
    text = " ".join(str(metadata.get(key, "")) for key in ("encoder", "software", "creation_time", "container_format")).lower()
    return any(term in text for term in ["runway", "pika", "sora", "kling", "luma", "veo", "synthesia", "heygen", "ai video", "diffusion"])


def _classifier_only_high(frames: list[dict[str, Any]], signals: dict[str, float | None]) -> bool:
    classifier_scores = [
        _numeric_or_zero((frame.get("classifier") or {}).get("classifier_risk"))
        for frame in frames
        if frame.get("classifier") is not None
    ]
    if not classifier_scores or max(classifier_scores) < 0.80:
        return False
    supporting = [
        signals.get("audio_risk"),
        signals.get("watermark_provenance_risk"),
        signals.get("temporal_consistency_risk"),
        signals.get("video_metadata_risk"),
        signals.get("filename_risk"),
    ]
    frame_support = []
    for frame in frames:
        frame_support.append(_numeric_or_zero((frame.get("forensics") or {}).get("risk_score")))
        frame_support.append(_numeric_or_zero((frame.get("watermark") or {}).get("risk_score")))
    return max([_numeric_or_zero(value) for value in supporting] + frame_support + [0.0]) < 0.35


def _only_filename_signal(signals: dict[str, float | None]) -> bool:
    filename = signals.get("filename_risk") or 0.0
    others = [value or 0.0 for key, value in signals.items() if key != "filename_risk"]
    return filename > 0 and max(others or [0.0]) < 0.10


def _summary(
    score: dict[str, Any],
    audio_analysis: dict[str, Any],
    visual_risk: float | None,
    temporal_analysis: dict[str, Any],
    flags: list[str],
) -> str:
    parts = [f"The video is assessed as {score['risk_level']} with a trust score of {score['trust_score']}."]
    fake_probability = audio_analysis.get("fake_probability")
    if fake_probability is not None:
        parts.append(f"The audio model estimated synthetic-speech risk at {float(fake_probability):.2f}.")
    if visual_risk is not None:
        parts.append(f"Visual multi-signal risk was {visual_risk:.2f}, with temporal risk {float(temporal_analysis.get('temporal_consistency_risk', 0.0)):.2f}.")
    if flags:
        parts.append(flags[0])
    parts.append("The image classifier contributes as a weak/moderate signal and is not the sole basis for the decision.")
    return " ".join(parts)


def _numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return _clamp(number)


def _numeric_or_zero(value: Any) -> float:
    number = _numeric_or_none(value)
    return 0.0 if number is None else number


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
