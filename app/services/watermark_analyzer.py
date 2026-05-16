from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


GENERATOR_TERMS = ["runway", "pika", "sora", "kling", "luma", "veo", "synthesia", "heygen", "d-id"]


@dataclass
class WatermarkResult:
    checked: bool
    possible_watermark_found: bool
    risk_score: float
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "possible_watermark_found": self.possible_watermark_found,
            "risk_score": round(self.risk_score, 4),
            "flags": self.flags,
            "warnings": self.warnings,
        }


def analyze_frame_watermark(image_path: str | Path, filename_context: str = "") -> WatermarkResult:
    flags: list[str] = []
    context = filename_context.lower()
    if any(term in context for term in GENERATOR_TERMS):
        flags.append("possible generator mark")

    try:
        import cv2  # type: ignore
    except ImportError:
        risk = 0.55 if flags else 0.0
        return WatermarkResult(False, bool(flags), risk, flags, ["OpenCV is not installed."])

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        risk = 0.55 if flags else 0.0
        return WatermarkResult(False, bool(flags), risk, flags, ["Frame image could not be read."])

    height, width = image.shape[:2]
    corner = image[int(height * 0.72):height, int(width * 0.62):width]
    if corner.size:
        edges = cv2.Canny(corner, 80, 180)
        edge_density = float(np.mean(edges > 0))
        contrast = float(np.std(corner))
        if edge_density > 0.08 and contrast > 20:
            flags.append("bottom-right watermark-like artifact")

    possible = bool(flags)
    risk_score = 0.65 if any("bottom-right" in flag for flag in flags) else (0.45 if possible else 0.0)
    if possible and risk_score < 0.5:
        flags.append("possible visible watermark")
    return WatermarkResult(True, possible, risk_score, flags)
