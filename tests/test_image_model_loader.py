from pathlib import Path

import numpy as np

from app.services.image_model_loader import (
    DEFAULT_IMAGE_SIZE,
    load_image_model,
    preprocess_frame_for_image_model,
)


def test_missing_image_model_returns_warning(tmp_path: Path) -> None:
    result = load_image_model(model_path=tmp_path / "missing.pth")

    assert result.checked is False
    assert result.model_found is False
    assert result.warnings


def test_image_model_loader_does_not_crash_app(tmp_path: Path) -> None:
    result = load_image_model(model_path=tmp_path / "missing.pth")

    assert result.model is None
    assert isinstance(result.warnings, list)


def test_preprocessing_produces_correct_tensor_shape() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)

    tensor = preprocess_frame_for_image_model(image)

    assert tensor.shape == (1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    assert tensor.dtype == np.float32
