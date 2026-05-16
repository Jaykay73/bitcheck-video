from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings, settings


DEFAULT_IMAGE_SIZE = 224
DEFAULT_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
DEFAULT_NORMALIZATION_STD = [0.229, 0.224, 0.225]


@dataclass
class ImageModelLoadResult:
    checked: bool
    model_found: bool
    model: Any | None = None
    model_path: Path | None = None
    model_type: str | None = None
    image_size: int = DEFAULT_IMAGE_SIZE
    normalization_mean: list[float] = field(default_factory=lambda: DEFAULT_NORMALIZATION_MEAN.copy())
    normalization_std: list[float] = field(default_factory=lambda: DEFAULT_NORMALIZATION_STD.copy())
    threshold: float = 0.5
    checkpoint_keys: list[str] = field(default_factory=list)
    output_mapping: str = "ai_probability"
    warnings: list[str] = field(default_factory=list)


def preprocess_frame_for_image_model(
    image: str | Path | np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> np.ndarray:
    frame = _load_image_as_rgb_array(image)
    resized = _resize_rgb(frame, image_size)
    normalized = resized.astype(np.float32) / 255.0
    mean_array = np.asarray(mean or DEFAULT_NORMALIZATION_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.asarray(std or DEFAULT_NORMALIZATION_STD, dtype=np.float32).reshape(1, 1, 3)
    normalized = (normalized - mean_array) / std_array
    return np.transpose(normalized, (2, 0, 1))[None, :, :, :].astype(np.float32)


def load_image_model(
    model_path: Path | str | None = None,
    app_settings: Settings = settings,
) -> ImageModelLoadResult:
    resolved_path = Path(model_path) if model_path is not None else app_settings.image_model_path

    if not resolved_path.exists():
        return ImageModelLoadResult(
            checked=False,
            model_found=False,
            model_path=resolved_path,
            warnings=[f"Image model file was not found at {app_settings.image_model_path.relative_to(app_settings.base_dir)}."],
        )

    lfs_warning = _git_lfs_pointer_warning(resolved_path)
    if lfs_warning:
        return ImageModelLoadResult(
            checked=False,
            model_found=False,
            model_path=resolved_path,
            warnings=[lfs_warning],
        )

    try:
        import torch  # type: ignore
    except ImportError:
        return ImageModelLoadResult(
            checked=False,
            model_found=True,
            model_path=resolved_path,
            warnings=[
                "PyTorch is not installed; image model loading was skipped.",
                "The frame image classifier is a weak/moderate signal and should not dominate video risk.",
            ],
        )

    try:
        checkpoint = torch.load(str(resolved_path), map_location="cpu", weights_only=False)
    except TypeError:
        try:
            checkpoint = torch.load(str(resolved_path), map_location="cpu")
        except Exception as exc:
            return _load_failure(resolved_path, f"Image model checkpoint could not be loaded: {exc}")
    except Exception as exc:
        return _load_failure(resolved_path, f"Image model checkpoint could not be loaded: {exc}")

    checkpoint_info = _inspect_checkpoint(checkpoint)
    warnings = ["The image classifier is treated as a weak/moderate signal."]
    model = None

    if _looks_like_torch_module(checkpoint):
        model = checkpoint
    else:
        state_dict = checkpoint_info["state_dict"]
        if state_dict is None:
            return ImageModelLoadResult(
                checked=False,
                model_found=True,
                model_path=resolved_path,
                image_size=checkpoint_info["image_size"],
                normalization_mean=checkpoint_info["normalization_mean"],
                normalization_std=checkpoint_info["normalization_std"],
                threshold=checkpoint_info["threshold"],
                checkpoint_keys=checkpoint_info["checkpoint_keys"],
                warnings=[
                    *warnings,
                    "Checkpoint did not contain model_state_dict or state_dict; raw probability mapping is unclear.",
                ],
            )
        model, build_warnings = _build_efficientnet_b0(state_dict)
        warnings.extend(build_warnings)

    if model is None:
        return ImageModelLoadResult(
            checked=False,
            model_found=True,
            model_path=resolved_path,
            image_size=checkpoint_info["image_size"],
            normalization_mean=checkpoint_info["normalization_mean"],
            normalization_std=checkpoint_info["normalization_std"],
            threshold=checkpoint_info["threshold"],
            checkpoint_keys=checkpoint_info["checkpoint_keys"],
            warnings=[*warnings, "EfficientNet-B0 model could not be constructed."],
        )

    try:
        model.eval()
        model.to("cpu")
    except Exception as exc:
        warnings.append(f"Image model loaded but could not be moved to CPU/eval mode: {exc}")

    return ImageModelLoadResult(
        checked=True,
        model_found=True,
        model=model,
        model_path=resolved_path,
        model_type=type(model).__name__,
        image_size=checkpoint_info["image_size"],
        normalization_mean=checkpoint_info["normalization_mean"],
        normalization_std=checkpoint_info["normalization_std"],
        threshold=checkpoint_info["threshold"],
        checkpoint_keys=checkpoint_info["checkpoint_keys"],
        warnings=warnings,
    )


@lru_cache(maxsize=1)
def load_image_model_cached() -> ImageModelLoadResult:
    return load_image_model()


def _inspect_checkpoint(checkpoint: Any) -> dict[str, Any]:
    checkpoint_keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    state_dict = None
    image_size = DEFAULT_IMAGE_SIZE
    threshold = 0.5
    mean = DEFAULT_NORMALIZATION_MEAN.copy()
    std = DEFAULT_NORMALIZATION_STD.copy()

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if state_dict is None and all(hasattr(value, "shape") for value in checkpoint.values()):
            state_dict = checkpoint
        image_size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
        threshold = float(checkpoint.get("threshold", 0.5))
        mean = list(checkpoint.get("normalization_mean", DEFAULT_NORMALIZATION_MEAN))
        std = list(checkpoint.get("normalization_std", DEFAULT_NORMALIZATION_STD))

    return {
        "checkpoint_keys": checkpoint_keys,
        "state_dict": state_dict,
        "image_size": image_size,
        "threshold": threshold,
        "normalization_mean": [float(value) for value in mean],
        "normalization_std": [float(value) for value in std],
    }


def _build_efficientnet_b0(state_dict: Any) -> tuple[Any | None, list[str]]:
    warnings: list[str] = []
    try:
        from torchvision import models  # type: ignore
        import torch  # type: ignore
    except ImportError:
        return None, ["torchvision is not installed; EfficientNet-B0 checkpoint could not be built."]

    try:
        model = models.efficientnet_b0(weights=None)
        classifier_key = next(
            (key for key in state_dict.keys() if key.endswith("classifier.1.weight")),
            None,
        )
        output_features = 1
        if classifier_key is not None:
            output_features = int(state_dict[classifier_key].shape[0])
        input_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(input_features, output_features)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            warnings.append(f"Image checkpoint missing {len(missing)} EfficientNet keys.")
        if unexpected:
            warnings.append(f"Image checkpoint had {len(unexpected)} unexpected keys.")
        return model, warnings
    except Exception as exc:
        return None, [f"EfficientNet-B0 checkpoint build failed: {exc}"]


def _load_image_as_rgb_array(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        array = image
    else:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to preprocess image paths.") from exc
        array = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if array is None:
            raise ValueError(f"Image could not be read: {image}")
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    return array.astype(np.uint8)


def _resize_rgb(image: np.ndarray, image_size: int) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError:
        y_indices = np.linspace(0, image.shape[0] - 1, image_size).astype(int)
        x_indices = np.linspace(0, image.shape[1] - 1, image_size).astype(int)
        return image[y_indices][:, x_indices]
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def _git_lfs_pointer_warning(path: Path) -> str | None:
    if path.stat().st_size > 1024:
        return None
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return None
    if "version https://git-lfs.github.com/spec/v1" in text:
        return f"Image model file at {path} appears to be a Git LFS pointer, not a real checkpoint."
    return None


def _looks_like_torch_module(value: Any) -> bool:
    return hasattr(value, "forward") and hasattr(value, "eval")


def _load_failure(path: Path, warning: str) -> ImageModelLoadResult:
    return ImageModelLoadResult(
        checked=False,
        model_found=True,
        model_path=path,
        warnings=[warning, "The image classifier is treated as a weak/moderate signal."],
    )
