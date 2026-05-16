from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class VisualForensicsResult:
    checked: bool
    risk_score: float
    quality_artifact_risk: float
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "risk_score": round(self.risk_score, 4),
            "quality_artifact_risk": round(self.quality_artifact_risk, 4),
            "flags": self.flags,
            "warnings": self.warnings,
        }


def analyze_visual_forensics(image_path: str | Path) -> VisualForensicsResult:
    try:
        import cv2  # type: ignore
    except ImportError:
        return VisualForensicsResult(False, 0.0, 0.0, warnings=["OpenCV is not installed."])

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return VisualForensicsResult(False, 0.0, 0.0, warnings=["Frame image could not be read."])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags: list[str] = []
    risks: list[float] = []

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < 25:
        flags.append("low_sharpness")
        risks.append(0.45)
    elif sharpness > 1800:
        flags.append("edge_oversharpening")
        risks.append(0.35)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    if brightness < 25 or brightness > 235:
        flags.append("brightness_anomaly")
        risks.append(0.35)
    if contrast < 12 or contrast > 95:
        flags.append("contrast_anomaly")
        risks.append(0.30)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))
    if edge_density < 0.01 or edge_density > 0.35:
        flags.append("edge_density_anomaly")
        risks.append(0.30)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = float(np.std(gray.astype(np.float32) - blur.astype(np.float32)))
    if noise < 1.5 or noise > 35:
        flags.append("noise_inconsistency")
        risks.append(0.30)

    # JPEG proxy: block boundary differences that are much stronger than interior differences.
    horizontal = np.abs(np.diff(gray.astype(np.float32), axis=0))
    vertical = np.abs(np.diff(gray.astype(np.float32), axis=1))
    block_score = 0.0
    if horizontal.size and vertical.size:
        block_rows = horizontal[7::8, :]
        block_cols = vertical[:, 7::8]
        interior_rows = horizontal[np.arange(horizontal.shape[0]) % 8 != 7, :]
        interior_cols = vertical[:, np.arange(vertical.shape[1]) % 8 != 7]
        interior = float(np.mean(interior_rows) + np.mean(interior_cols) + 1e-6)
        boundary = float(np.mean(block_rows) + np.mean(block_cols))
        block_score = boundary / interior
    if block_score > 1.45:
        flags.append("compression_artifact_proxy")
        risks.append(min(0.45, (block_score - 1.0) / 2.0))

    risk_score = float(np.mean(risks)) if risks else 0.0
    quality_risk = min(1.0, risk_score + (0.1 if len(flags) >= 3 else 0.0))
    return VisualForensicsResult(True, min(1.0, risk_score), quality_risk, flags)
