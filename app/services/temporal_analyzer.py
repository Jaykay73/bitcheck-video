from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


HIGH_RISK_THRESHOLD = 0.60
SUSPICIOUS_THRESHOLD = 0.45


def analyze_temporal_consistency(frame_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not frame_results:
        return {
            "checked": True,
            "frames_analyzed": 0,
            "mean_frame_risk": 0.0,
            "median_frame_risk": 0.0,
            "max_frame_risk": 0.0,
            "top_20_percent_frame_risk_mean": 0.0,
            "suspicious_frame_ratio": 0.0,
            "high_risk_frame_count": 0,
            "risk_volatility": 0.0,
            "repeated_signal_consistency": 0.0,
            "temporal_consistency_risk": 0.0,
            "flags": [],
            "warnings": ["No frame analysis results were available for temporal analysis."],
        }

    risks = np.asarray([_risk(frame) for frame in frame_results], dtype=np.float32)
    mean_risk = float(np.mean(risks))
    median_risk = float(np.median(risks))
    max_risk = float(np.max(risks))
    top_count = max(1, int(np.ceil(len(risks) * 0.20)))
    top_mean = float(np.mean(np.sort(risks)[-top_count:]))
    suspicious_ratio = float(np.mean(risks >= SUSPICIOUS_THRESHOLD))
    high_risk_count = int(np.sum(risks >= HIGH_RISK_THRESHOLD))
    volatility = float(np.std(risks))
    repeated_consistency = _repeated_signal_consistency(frame_results)

    temporal_risk = (
        0.30 * mean_risk
        + 0.25 * top_mean
        + 0.20 * suspicious_ratio
        + 0.15 * repeated_consistency
        + 0.10 * max_risk
    )
    if high_risk_count <= 1 and len(risks) >= 4:
        temporal_risk = min(temporal_risk, 0.55)
    if high_risk_count >= max(2, len(risks) // 2):
        temporal_risk = min(1.0, temporal_risk + 0.08)
    if repeated_consistency >= 0.60 and suspicious_ratio >= 0.50:
        temporal_risk = min(1.0, temporal_risk + 0.07)

    flags: list[str] = []
    if suspicious_ratio >= 0.50:
        flags.append("Suspicious visual signals appear repeatedly across sampled frames.")
    if repeated_consistency >= 0.60:
        flags.append("Similar suspicious frame-level signals repeat across the sampled video.")
    if high_risk_count == 1 and len(risks) > 1:
        flags.append("One sampled frame was high risk, but the signal was not consistent across the video.")

    return {
        "checked": True,
        "frames_analyzed": len(frame_results),
        "mean_frame_risk": _round(mean_risk),
        "median_frame_risk": _round(median_risk),
        "max_frame_risk": _round(max_risk),
        "top_20_percent_frame_risk_mean": _round(top_mean),
        "suspicious_frame_ratio": _round(suspicious_ratio),
        "high_risk_frame_count": high_risk_count,
        "risk_volatility": _round(volatility),
        "repeated_signal_consistency": _round(repeated_consistency),
        "temporal_consistency_risk": _round(temporal_risk),
        "flags": flags,
        "warnings": [],
    }


def _risk(frame: dict[str, Any]) -> float:
    return max(0.0, min(1.0, float(frame.get("risk_score", 0.0) or 0.0)))


def _repeated_signal_consistency(frame_results: list[dict[str, Any]]) -> float:
    counter: Counter[str] = Counter()
    frames_with_flags = 0
    for frame in frame_results:
        flags = set(str(flag) for flag in frame.get("flags", []) if flag)
        for section_name in ("forensics", "watermark"):
            section = frame.get(section_name) or {}
            flags.update(str(flag) for flag in section.get("flags", []) if flag)
        if flags:
            frames_with_flags += 1
            counter.update(flags)

    if not counter:
        return 0.0

    repeated_counts = [count for count in counter.values() if count > 1]
    if not repeated_counts:
        return 0.0
    strongest_repeat = max(repeated_counts) / max(1, len(frame_results))
    coverage = frames_with_flags / max(1, len(frame_results))
    return max(0.0, min(1.0, 0.7 * strongest_repeat + 0.3 * coverage))


def _round(value: float) -> float:
    return round(float(max(0.0, min(1.0, value))), 4)
