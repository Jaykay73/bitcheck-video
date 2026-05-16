from __future__ import annotations

from typing import Any


LIMITATIONS = [
    "BitCheck analyzes sampled frames, not every frame.",
    "Video analysis is probabilistic and should not be treated as absolute proof.",
    "The image classifier is treated as a weak/moderate signal because it may not generalize to all generators.",
    "Audio Random Forest inference depends on matching the exact feature extraction used during training.",
    "Compressed or noisy audio may affect audio model accuracy.",
    "Social media compression may affect visual forensic signals.",
    "Grad-CAM shows model attention, not proof of manipulation.",
    "Missing metadata does not prove a video is fake.",
    "High-stakes decisions should involve human review.",
]

RECOMMENDED_ACTIONS = [
    "Review the top suspicious frames.",
    "Check whether audio and visual signals agree.",
    "Request the original uncompressed video where possible.",
    "Verify the source of the video independently.",
    "Use manual review for high-risk outputs.",
]


def build_verification_report(
    trust: dict[str, Any],
    aggregation: dict[str, Any],
    risk_flags: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    deduped_flags = _dedupe(risk_flags or [])
    summary = build_summary(trust, aggregation, deduped_flags)
    return {
        "summary": summary,
        "risk_flags": deduped_flags,
        "recommended_actions": recommended_actions_for_decision(str(trust.get("decision", "review"))),
        "limitations": LIMITATIONS.copy(),
        "warnings": _dedupe(warnings or []),
    }


def build_summary(trust: dict[str, Any], aggregation: dict[str, Any], risk_flags: list[str] | None = None) -> str:
    risk_level = str(trust.get("risk_level", aggregation.get("risk_level", "Suspicious")))
    trust_score = trust.get("trust_score", aggregation.get("trust_score", 50))
    signal_scores = aggregation.get("signal_scores", {})
    flags = risk_flags or aggregation.get("flags", [])

    if risk_level in {"High Risk", "Very High Risk"}:
        opener = "The video shows elevated synthetic-media risk"
    elif risk_level == "Suspicious":
        opener = "The video has mixed or limited signals and should be reviewed"
    else:
        opener = "The video currently shows lower synthetic-media risk"

    reasons: list[str] = []
    if (signal_scores.get("audio_risk") or 0.0) >= 0.70:
        reasons.append("the audio model indicated elevated synthetic-speech risk")
    if (signal_scores.get("visual_multisignal_risk") or 0.0) >= 0.50:
        reasons.append("multiple sampled frames showed visual risk signals")
    if (signal_scores.get("temporal_consistency_risk") or 0.0) >= 0.50:
        reasons.append("suspicious visual signals repeated over time")
    if signal_scores.get("watermark_provenance_risk"):
        reasons.append("watermark or provenance-like indicators were present")
    if not reasons and flags:
        reasons.append(flags[0])

    reason_text = f" because {', and '.join(reasons)}" if reasons else " based on the available signals"
    return (
        f"{opener}{reason_text}. Trust score: {trust_score}. "
        "The result is risk-based and should not be treated as absolute proof."
    )


def recommended_actions_for_decision(decision: str) -> list[str]:
    if decision == "approve":
        return RECOMMENDED_ACTIONS[:4]
    return RECOMMENDED_ACTIONS.copy()


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))
