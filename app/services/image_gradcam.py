from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.services.image_model_loader import ImageModelLoadResult, load_image_model_cached, preprocess_frame_for_image_model


def generate_gradcam_for_top_frames(
    frame_results: list[dict[str, Any]],
    image_model: ImageModelLoadResult | None = None,
    enabled: bool = True,
    top_k: int | None = None,
    verification_id: str = "",
) -> dict[str, Any]:
    if not enabled:
        return {
            "checked": False,
            "frames_explained": 0,
            "items": [],
            "warnings": ["Grad-CAM analysis was disabled for this request."],
        }

    model_result = image_model or load_image_model_cached()
    if not model_result.checked or model_result.model is None:
        return {
            "checked": False,
            "frames_explained": 0,
            "items": [],
            "warnings": model_result.warnings or ["Image model is unavailable; Grad-CAM was skipped."],
        }

    max_items = max(0, min(top_k if top_k is not None else settings.gradcam_top_k, settings.gradcam_top_k))
    sorted_frames = sorted(frame_results, key=lambda item: float(item.get("risk_score", 0.0)), reverse=True)[:max_items]
    items: list[dict[str, Any]] = []
    warnings: list[str] = []

    for frame in sorted_frames:
        frame_path = _frame_path(frame)
        if frame_path is None:
            warnings.append(f"Frame {frame.get('frame_id', 'unknown')} has no local path for Grad-CAM.")
            continue
        try:
            overlay_path = _overlay_path(frame_path, verification_id)
            _create_gradcam_overlay(frame_path, overlay_path, model_result)
            items.append(
                {
                    "frame_id": frame.get("frame_id"),
                    "timestamp": frame.get("timestamp", 0.0),
                    "gradcam_overlay_url": f"/outputs/{overlay_path.name}",
                    "target_class": "ai_generated",
                    "note": "Grad-CAM shows model attention and is not proof of manipulation.",
                }
            )
        except Exception as exc:
            warnings.append(f"Grad-CAM failed for frame {frame.get('frame_id', 'unknown')}: {exc}")

    return {
        "checked": True,
        "frames_explained": len(items),
        "items": items,
        "warnings": _dedupe(warnings),
    }


def _create_gradcam_overlay(frame_path: Path, output_path: Path, model_result: ImageModelLoadResult) -> None:
    try:
        import cv2  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV and PyTorch are required for Grad-CAM.") from exc

    target_layer = _find_last_conv_layer(model_result.model)
    if target_layer is None:
        raise RuntimeError("No convolutional layer was found for Grad-CAM.")

    activations: list[Any] = []
    gradients: list[Any] = []

    def forward_hook(_module: Any, _inputs: Any, output: Any) -> None:
        activations.append(output.detach())

    def backward_hook(_module: Any, _grad_input: Any, grad_output: Any) -> None:
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        tensor = torch.from_numpy(
            preprocess_frame_for_image_model(
                frame_path,
                image_size=model_result.image_size,
                mean=model_result.normalization_mean,
                std=model_result.normalization_std,
            )
        ).float()
        model_result.model.zero_grad(set_to_none=True)
        output = model_result.model(tensor)
        flattened = output.reshape(output.shape[0], -1)
        target_score = flattened[:, -1].sum() if flattened.shape[1] == 1 else flattened.max(dim=1).values.sum()
        target_score.backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        activation = activations[-1][0]
        gradient = gradients[-1][0]
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=0)).cpu().numpy()
        cam = _normalize_heatmap(cam)

        original = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if original is None:
            raise RuntimeError("Frame image could not be read for overlay.")
        heatmap = cv2.resize(cam, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.65, heatmap_color, 0.35, 0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), overlay)
    finally:
        forward_handle.remove()
        backward_handle.remove()


def _find_last_conv_layer(model: Any) -> Any | None:
    try:
        import torch  # type: ignore
    except ImportError:
        return None

    last_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_layer = module
    return last_layer


def _normalize_heatmap(cam: np.ndarray) -> np.ndarray:
    cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = float(np.min(cam))
    maximum = float(np.max(cam))
    if maximum - minimum <= 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - minimum) / (maximum - minimum)).astype(np.float32)


def _frame_path(frame: dict[str, Any]) -> Path | None:
    value = frame.get("path") or frame.get("frame_path")
    if value:
        return Path(str(value))
    frame_url = str(frame.get("frame_url") or frame.get("url") or "")
    if frame_url.startswith("/outputs/"):
        return settings.outputs_dir / Path(frame_url).name
    return None


def _overlay_path(frame_path: Path, verification_id: str) -> Path:
    prefix = f"{verification_id}_" if verification_id else ""
    stem = frame_path.stem
    return settings.outputs_dir / f"{prefix}{stem}_gradcam.jpg"


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
