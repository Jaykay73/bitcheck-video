from pathlib import Path

import cv2
import numpy as np

from app.services.image_frame_analyzer import _dynamic_weighted_score, analyze_single_frame
from app.services.image_model_loader import ImageModelLoadResult
from app.services.visual_forensics import analyze_visual_forensics


def write_frame(path: Path) -> None:
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (60, 60), (90, 90), (245, 245, 245), -1)
    cv2.imwrite(str(path), image)


def test_frame_analyzer_handles_missing_image_model(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.jpg"
    write_frame(frame_path)
    model = ImageModelLoadResult(False, False, warnings=["missing model"])

    result = analyze_single_frame(
        {"frame_id": 1, "timestamp": 0.4, "url": "/outputs/frame.jpg", "path": str(frame_path)},
        image_model=model,
    )

    assert result.classifier["checked"] is False
    assert result.risk_score >= 0.0


def test_classifier_only_high_risk_does_not_dominate() -> None:
    risk, contributions = _dynamic_weighted_score(
        {
            "classifier_risk": 1.0,
            "visible_watermark_risk": 0.0,
            "metadata_or_provenance_risk": 0.0,
            "forensic_risk": 0.0,
            "filename_context_risk": 0.0,
            "image_quality_artifact_risk": 0.0,
        }
    )

    assert contributions["classifier_risk"] <= 0.15
    assert risk <= 0.15


def test_dynamic_weighting_renormalizes_available_signals() -> None:
    risk, contributions = _dynamic_weighted_score({"forensic_risk": 0.5, "classifier_risk": None})

    assert risk == 0.5
    assert contributions["forensic_risk"] == 0.5


def test_visual_forensics_returns_structured_output(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.jpg"
    write_frame(frame_path)

    result = analyze_visual_forensics(frame_path).to_dict()

    assert result["checked"] is True
    assert "risk_score" in result
    assert isinstance(result["flags"], list)
