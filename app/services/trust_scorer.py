from __future__ import annotations

from typing import Any


def score_trust(risk_score: float) -> dict[str, Any]:
    risk = max(0.0, min(1.0, float(risk_score)))
    trust_score = round((1.0 - risk) * 100)
    risk_level, decision = _level_for_trust_score(trust_score)
    return {
        "trust_score": trust_score,
        "risk_score": round(risk, 4),
        "risk_level": risk_level,
        "decision": decision,
    }


def enforce_minimum_risk_level(score: dict[str, Any], minimum_level: str) -> dict[str, Any]:
    minimum_risk = {
        "Suspicious": 0.41,
        "High Risk": 0.61,
        "Very High Risk": 0.81,
    }.get(minimum_level)
    if minimum_risk is None:
        return score
    if float(score["risk_score"]) < minimum_risk:
        return score_trust(minimum_risk)
    return score


def cap_maximum_risk_level(score: dict[str, Any], maximum_level: str) -> dict[str, Any]:
    maximum_risk = {
        "Likely Authentic": 0.20,
        "Low Risk": 0.40,
        "Suspicious": 0.60,
        "High Risk": 0.80,
    }.get(maximum_level)
    if maximum_risk is None:
        return score
    if float(score["risk_score"]) > maximum_risk:
        return score_trust(maximum_risk)
    return score


def _level_for_trust_score(trust_score: int) -> tuple[str, str]:
    if trust_score >= 80:
        return "Likely Authentic", "approve"
    if trust_score >= 60:
        return "Low Risk", "approve"
    if trust_score >= 40:
        return "Suspicious", "review"
    if trust_score >= 20:
        return "High Risk", "manual_review"
    return "Very High Risk", "manual_review"
