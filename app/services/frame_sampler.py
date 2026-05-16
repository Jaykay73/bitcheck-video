from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings, settings


VALID_SAMPLE_MODES = {"uniform", "scene_change", "hybrid"}


@dataclass
class SampledFrame:
    frame_id: int
    timestamp: float
    path: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 4),
            "path": self.path,
            "url": self.url,
        }


@dataclass
class FrameSamplingResult:
    checked: bool
    sample_mode: str
    frames_requested: int
    frames_extracted: int
    sampled_timestamps: list[float] = field(default_factory=list)
    frames: list[SampledFrame] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "sample_mode": self.sample_mode,
            "frames_requested": self.frames_requested,
            "frames_extracted": self.frames_extracted,
            "sampled_timestamps": [round(timestamp, 4) for timestamp in self.sampled_timestamps],
            "frames": [frame.to_dict() for frame in self.frames],
            "warnings": self.warnings,
        }


def sample_video_frames(
    video_path: Path | str,
    verification_id: str,
    sample_mode: str = "uniform",
    max_frames: int | None = None,
    analyze_first_seconds: int | None = None,
    app_settings: Settings = settings,
) -> FrameSamplingResult:
    mode = sample_mode.lower().strip()
    requested_frames = _clamp_requested_frames(max_frames, app_settings.max_frames_to_analyze)

    if mode not in VALID_SAMPLE_MODES:
        return FrameSamplingResult(
            checked=False,
            sample_mode=mode,
            frames_requested=requested_frames,
            frames_extracted=0,
            warnings=[f"Unsupported sample_mode '{sample_mode}'. Allowed values: hybrid, scene_change, uniform."],
        )

    try:
        import cv2  # type: ignore
    except ImportError:
        return FrameSamplingResult(
            checked=False,
            sample_mode=mode,
            frames_requested=requested_frames,
            frames_extracted=0,
            warnings=["OpenCV is not installed; frame sampling was skipped."],
        )

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return FrameSamplingResult(
                checked=False,
                sample_mode=mode,
                frames_requested=requested_frames,
                frames_extracted=0,
                warnings=["Video could not be opened for frame sampling."],
            )

        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            return FrameSamplingResult(
                checked=False,
                sample_mode=mode,
                frames_requested=requested_frames,
                frames_extracted=0,
                warnings=["Video frame count or FPS is unavailable; frame sampling was skipped."],
            )

        effective_frame_count = frame_count
        if analyze_first_seconds is not None:
            effective_frame_count = min(frame_count, max(1, int(analyze_first_seconds * fps)))

        if mode == "uniform":
            frame_indices = _uniform_indices(effective_frame_count, requested_frames)
        elif mode == "scene_change":
            frame_indices = _scene_change_indices(capture, effective_frame_count, requested_frames, cv2)
        else:
            uniform = _uniform_indices(effective_frame_count, requested_frames)
            scene = _scene_change_indices(capture, effective_frame_count, requested_frames, cv2)
            frame_indices = _dedupe_indices([*uniform, *scene])[:requested_frames]

        frames = _save_frame_indices(
            capture=capture,
            cv2=cv2,
            frame_indices=frame_indices,
            fps=fps,
            verification_id=verification_id,
            output_dir=app_settings.outputs_dir,
        )

        warnings: list[str] = []
        if len(frames) < requested_frames:
            warnings.append(
                "Fewer frames were extracted than requested because the video has limited readable frames."
            )

        return FrameSamplingResult(
            checked=True,
            sample_mode=mode,
            frames_requested=requested_frames,
            frames_extracted=len(frames),
            sampled_timestamps=[frame.timestamp for frame in frames],
            frames=frames,
            warnings=warnings,
        )
    finally:
        capture.release()


def _clamp_requested_frames(max_frames: int | None, hard_limit: int) -> int:
    if max_frames is None:
        return hard_limit
    return max(1, min(int(max_frames), hard_limit))


def _uniform_indices(frame_count: int, requested_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= requested_frames:
        return list(range(frame_count))
    if requested_frames == 1:
        return [0]

    step = (frame_count - 1) / (requested_frames - 1)
    return _dedupe_indices(round(index * step) for index in range(requested_frames))


def _scene_change_indices(capture: Any, frame_count: int, requested_frames: int, cv2: Any) -> list[int]:
    if frame_count <= 0:
        return []

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    previous_hist = None
    scored_indices: list[tuple[float, int]] = []

    for index in range(frame_count):
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        if previous_hist is not None:
            diff = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            scored_indices.append((diff, index))
        previous_hist = hist

    chosen = [0]
    chosen.extend(index for _, index in sorted(scored_indices, reverse=True)[: requested_frames - 1])
    return sorted(_dedupe_indices(chosen))[:requested_frames]


def _save_frame_indices(
    capture: Any,
    cv2: Any,
    frame_indices: list[int],
    fps: float,
    verification_id: str,
    output_dir: Path,
) -> list[SampledFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[SampledFrame] = []

    for position, frame_index in enumerate(frame_indices, start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue

        filename = f"{verification_id}_frame_{position:03d}.jpg"
        saved_path = output_dir / filename
        if not cv2.imwrite(str(saved_path), frame):
            continue

        relative_path = f"outputs/{filename}"
        frames.append(
            SampledFrame(
                frame_id=position,
                timestamp=frame_index / fps if fps else 0.0,
                path=relative_path,
                url=f"/outputs/{filename}",
            )
        )

    return frames


def _dedupe_indices(indices: Any) -> list[int]:
    seen: set[int] = set()
    deduped: list[int] = []
    for index in indices:
        integer_index = int(index)
        if integer_index < 0 or integer_index in seen:
            continue
        seen.add(integer_index)
        deduped.append(integer_index)
    return deduped
