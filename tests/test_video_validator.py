import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile

from app.config import Settings
from app.services.file_validator import FileValidationError, save_and_validate_video_upload


class ValidatorSettings(Settings):
    def __init__(self, upload_dir: Path, max_size_mb: int = 50) -> None:
        self.uploads_dir = upload_dir
        self.outputs_dir = upload_dir.parent / "outputs"
        self.models_dir = upload_dir.parent / "models"
        self.max_video_size_mb = max_size_mb


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(content))


def test_valid_mp4_like_file_passes_validation(tmp_path: Path) -> None:
    content = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32
    settings = ValidatorSettings(tmp_path)

    result = asyncio.run(
        save_and_validate_video_upload(make_upload("sample.mp4", content), settings)
    )

    assert result.extension == ".mp4"
    assert result.file_size_bytes == len(content)
    assert result.sha256
    assert result.warnings == []
    assert result.saved_path.exists()


def test_txt_file_is_rejected(tmp_path: Path) -> None:
    settings = ValidatorSettings(tmp_path)

    with pytest.raises(FileValidationError, match="Unsupported video file type"):
        asyncio.run(save_and_validate_video_upload(make_upload("sample.txt", b"text"), settings))


def test_safe_uuid_filename_generated(tmp_path: Path) -> None:
    content = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16
    settings = ValidatorSettings(tmp_path)

    result = asyncio.run(
        save_and_validate_video_upload(make_upload("../../unsafe name.mp4", content), settings)
    )

    assert result.original_filename == "unsafe name.mp4"
    assert result.stored_filename.endswith(".mp4")
    assert result.stored_filename != result.original_filename
    assert result.saved_path.parent == tmp_path.resolve()


def test_file_size_check_rejects_too_large_file(tmp_path: Path) -> None:
    content = b"\x00\x00\x00\x18ftypmp42" + b"0" * 20
    settings = ValidatorSettings(tmp_path, max_size_mb=0)

    with pytest.raises(FileValidationError) as exc:
        asyncio.run(save_and_validate_video_upload(make_upload("sample.mp4", content), settings))

    assert exc.value.status_code == 413
