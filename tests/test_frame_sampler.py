from pathlib import Path

import cv2
import numpy as np

from app.config import Settings
from app.services.frame_sampler import sample_video_frames


class SamplerSettings(Settings):
    def __init__(self, output_dir: Path, max_frames: int = 12) -> None:
        self.outputs_dir = output_dir
        self.max_frames_to_analyze = max_frames


def create_synthetic_video(path: Path, frame_count: int = 10, fps: float = 5.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (96, 64),
    )
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            frame[:, :, 0] = (index * 25) % 255
            frame[:, :, 1] = (index * 40) % 255
            frame[:, :, 2] = 80
            writer.write(frame)
    finally:
        writer.release()


def test_extracts_frames_from_synthetic_short_video(tmp_path: Path) -> None:
    video_path = tmp_path / "short.mp4"
    output_dir = tmp_path / "outputs"
    create_synthetic_video(video_path, frame_count=10, fps=5.0)

    result = sample_video_frames(
        video_path,
        verification_id="sample",
        sample_mode="uniform",
        max_frames=8,
        app_settings=SamplerSettings(output_dir),
    )

    assert result.checked is True
    assert result.frames_requested == 8
    assert result.frames_extracted == 8
    assert len(result.sampled_timestamps) == 8
    assert result.frames[0].path.startswith("outputs/")
    assert result.frames[0].url.startswith("/outputs/")
    assert (output_dir / "sample_frame_001.jpg").exists()


def test_max_frames_respected(tmp_path: Path) -> None:
    video_path = tmp_path / "short.mp4"
    output_dir = tmp_path / "outputs"
    create_synthetic_video(video_path, frame_count=20, fps=10.0)

    result = sample_video_frames(
        video_path,
        verification_id="limited",
        sample_mode="uniform",
        max_frames=99,
        app_settings=SamplerSettings(output_dir, max_frames=6),
    )

    assert result.frames_requested == 6
    assert result.frames_extracted <= 6


def test_timestamps_returned_without_duplicates(tmp_path: Path) -> None:
    video_path = tmp_path / "lowfps.mp4"
    output_dir = tmp_path / "outputs"
    create_synthetic_video(video_path, frame_count=3, fps=1.0)

    result = sample_video_frames(
        video_path,
        verification_id="lowfps",
        sample_mode="uniform",
        max_frames=12,
        app_settings=SamplerSettings(output_dir),
    )

    assert result.frames_extracted == 3
    assert result.sampled_timestamps == sorted(result.sampled_timestamps)
    assert len(result.sampled_timestamps) == len(set(result.sampled_timestamps))


def test_no_crash_on_bad_video(tmp_path: Path) -> None:
    video_path = tmp_path / "bad.mp4"
    video_path.write_bytes(b"not a video")

    result = sample_video_frames(
        video_path,
        verification_id="bad",
        sample_mode="uniform",
        max_frames=12,
        app_settings=SamplerSettings(tmp_path / "outputs"),
    )

    assert result.checked is False
    assert result.frames_extracted == 0
    assert result.warnings
