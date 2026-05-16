from __future__ import annotations

from typing import Any


AI_VIDEO_TOOL_KEYWORDS = {
    "runway",
    "pika",
    "sora",
    "kling",
    "luma",
    "veo",
    "gen-2",
    "gen2",
    "stable video",
    "diffusion",
    "ai video",
    "synthesia",
    "heygen",
    "d-id",
    "deepbrain",
}


def safe_float(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def parse_fraction(value: Any) -> float | None:
    if not value or value == "0/0":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    if "/" not in text:
        return safe_float(text)
    numerator, denominator = text.split("/", 1)
    top = safe_float(numerator)
    bottom = safe_float(denominator)
    if top is None or bottom in (None, 0):
        return None
    return top / bottom


def first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", "N/A"):
            return value
    return None


def lower_joined_values(*values: Any) -> str:
    return " ".join(str(value).lower() for value in values if value)
