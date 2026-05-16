from pathlib import Path

import numpy as np

from app.services.image_gradcam import generate_gradcam_for_top_frames
from app.services.image_model_loader import ImageModelLoadResult


def test_gradcam_disabled_path_works():
    result = generate_gradcam_for_top_frames([], enabled=False)

    assert result["checked"] is False
    assert result["frames_explained"] == 0
    assert result["items"] == []
    assert result["warnings"]


def test_missing_model_returns_warning():
    model = ImageModelLoadResult(
        checked=False,
        model_found=False,
        warnings=["Image model file was not found at models/ai_vs_real_image_detector.pth."],
    )

    result = generate_gradcam_for_top_frames([{"frame_id": 1, "risk_score": 0.9}], image_model=model)

    assert result["checked"] is False
    assert result["frames_explained"] == 0
    assert "not found" in result["warnings"][0]


def test_output_path_is_relative(tmp_path):
    frame_path = tmp_path / "frame_001.jpg"
    _write_frame(frame_path)
    model = ImageModelLoadResult(
        checked=True,
        model_found=True,
        model=object(),
        warnings=[],
    )

    result = generate_gradcam_for_top_frames(
        [{"frame_id": 1, "timestamp": 0.2, "risk_score": 0.9, "path": str(frame_path)}],
        image_model=model,
    )

    assert result["checked"] is True
    assert result["frames_explained"] == 0
    assert result["warnings"]
    assert all(not item.get("gradcam_overlay_url", "").startswith(str(Path.cwd())) for item in result["items"])


def _write_frame(path: Path) -> None:
    import cv2  # type: ignore

    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(path), image)
