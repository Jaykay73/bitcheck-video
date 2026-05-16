import asyncio
from pathlib import Path

import httpx

from app.config import settings
from main import app
from tests.test_frame_sampler import create_synthetic_video


def test_root() -> None:
    async def run_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/")

    response = asyncio.run(run_request())

    assert response.status_code == 200
    assert response.json() == {
        "service": "BitCheck Video Verification API",
        "status": "running",
        "version": "1.0.0",
    }


def test_health() -> None:
    async def run_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/health")

    response = asyncio.run(run_request())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "BitCheck Video Verification API"
    assert payload["version"] == "1.0.0"
    assert isinstance(payload["audio_model_found"], bool)
    assert isinstance(payload["image_model_found"], bool)
    assert isinstance(payload["ffmpeg_available"], bool)
    assert payload["max_video_duration_seconds"] == 5
    assert payload["max_frames_to_analyze"] == 12


def test_verify_video_rejects_unreadable_video() -> None:
    async def run_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            files = {
                "file": (
                    "clip.mp4",
                    b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16,
                    "video/mp4",
                )
            }
            return await client.post("/verify/video", files=files)

    response = asyncio.run(run_request())

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "BitCheck"
    assert payload["file_type"] == "video"
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "unreadable_video"
    assert payload["input"]["extension"] == ".mp4"
    assert payload["input"]["filename"].endswith(".mp4")
    assert payload["file_validation"] == {"valid": True, "warnings": []}
    assert payload["video_metadata"]["checked"] is False

    saved_path = Path(settings.uploads_dir) / payload["input"]["filename"]
    if saved_path.exists():
        saved_path.unlink()


def test_verify_video_succeeds_with_audio_analysis_false(tmp_path: Path) -> None:
    video_path = tmp_path / "short.mp4"
    create_synthetic_video(video_path, frame_count=10, fps=5.0)

    async def run_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            with video_path.open("rb") as handle:
                files = {"file": ("short.mp4", handle.read(), "video/mp4")}
            data = {"run_audio_analysis": "false", "run_gradcam": "false"}
            return await client.post("/verify/video", files=files, data=data)

    response = asyncio.run(run_request())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"completed", "completed_with_warnings"}
    assert payload["audio_analysis"]["checked"] is False
    assert payload["frame_sampling"]["frames_extracted"] > 0
    assert _paths_are_relative(payload)


def test_verify_video_succeeds_with_image_analysis_false(tmp_path: Path) -> None:
    video_path = tmp_path / "short.mp4"
    create_synthetic_video(video_path, frame_count=8, fps=4.0)

    async def run_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            with video_path.open("rb") as handle:
                files = {"file": ("short.mp4", handle.read(), "video/mp4")}
            data = {
                "run_image_analysis": "false",
                "run_audio_analysis": "false",
                "run_gradcam": "false",
            }
            return await client.post("/verify/video", files=files, data=data)

    response = asyncio.run(run_request())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"completed", "completed_with_warnings"}
    assert payload["visual_analysis"]["checked"] is False
    assert payload["trust"]["decision"] in {"approve", "review", "manual_review"}
    assert _paths_are_relative(payload)


def test_verify_video_rejects_over_five_seconds(tmp_path: Path) -> None:
    video_path = tmp_path / "long.mp4"
    create_synthetic_video(video_path, frame_count=36, fps=6.0)

    async def run_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            with video_path.open("rb") as handle:
                files = {"file": ("long.mp4", handle.read(), "video/mp4")}
            return await client.post("/verify/video", files=files)

    response = asyncio.run(run_request())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "video_too_long"
    assert payload["error"]["max_video_duration_seconds"] == 5


def _paths_are_relative(payload: dict) -> bool:
    text = str(payload)
    assert str(settings.base_dir) not in text
    return "/mnt/" not in text and "\\Users\\" not in text
