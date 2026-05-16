from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from fastapi import UploadFile

from app.config import Settings, settings
from app.utils.file_utils import (
    SUPPORTED_VIDEO_EXTENSIONS,
    build_uuid_filename,
    clean_original_filename,
    get_extension,
    resolve_within_directory,
)


class FileValidationError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class ValidatedUpload:
    verification_id: str
    original_filename: str
    stored_filename: str
    saved_path: Path
    sha256: str
    extension: str
    file_size_bytes: int
    warnings: list[str]


def validate_video_extension(filename: str | None) -> str:
    extension = get_extension(filename)
    if extension not in SUPPORTED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise FileValidationError(f"Unsupported video file type. Allowed extensions: {allowed}")
    return extension


def _signature_warning_or_error(extension: str, header: bytes) -> str | None:
    if not header:
        raise FileValidationError("Uploaded file is empty")

    if extension in {".mp4", ".mov"}:
        if len(header) >= 12 and header[4:8] == b"ftyp":
            return None
        raise FileValidationError("File signature does not match an MP4/MOV video")

    if extension == ".avi":
        if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI ":
            return None
        raise FileValidationError("File signature does not match an AVI video")

    if extension in {".mkv", ".webm"}:
        if header.startswith(b"\x1a\x45\xdf\xa3"):
            return None
        raise FileValidationError("File signature does not match an MKV/WEBM video")

    return "No signature rule is available for this extension"


async def save_and_validate_video_upload(
    upload: UploadFile,
    app_settings: Settings = settings,
) -> ValidatedUpload:
    original_filename = clean_original_filename(upload.filename)
    extension = validate_video_extension(original_filename)
    max_bytes = app_settings.max_video_size_mb * 1024 * 1024
    stored_filename = build_uuid_filename(extension)
    saved_path = resolve_within_directory(app_settings.uploads_dir, stored_filename)

    digest = sha256()
    file_size = 0
    header = b""
    warnings: list[str] = []

    app_settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    try:
        with saved_path.open("wb") as output:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                if len(header) < 32:
                    header = (header + chunk)[:32]
                file_size += len(chunk)
                if file_size > max_bytes:
                    raise FileValidationError(
                        f"Video file is too large. Maximum size is {app_settings.max_video_size_mb} MB",
                        status_code=413,
                    )
                digest.update(chunk)
                output.write(chunk)

        signature_warning = _signature_warning_or_error(extension, header)
        if signature_warning:
            warnings.append(signature_warning)

        return ValidatedUpload(
            verification_id=stored_filename.removesuffix(extension),
            original_filename=original_filename,
            stored_filename=stored_filename,
            saved_path=saved_path,
            sha256=digest.hexdigest(),
            extension=extension,
            file_size_bytes=file_size,
            warnings=warnings,
        )
    except Exception:
        if saved_path.exists():
            saved_path.unlink()
        raise
    finally:
        upload.file.close()
