from pathlib import Path
from uuid import uuid4


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def clean_original_filename(filename: str | None) -> str:
    if not filename:
        return "upload"
    return Path(filename).name


def get_extension(filename: str | None) -> str:
    return Path(clean_original_filename(filename)).suffix.lower()


def build_uuid_filename(extension: str) -> str:
    return f"{uuid4()}{extension.lower()}"


def resolve_within_directory(directory: Path, filename: str) -> Path:
    base = directory.resolve()
    target = (base / Path(filename).name).resolve()
    if base not in target.parents and target != base:
        raise ValueError("Resolved path is outside the target directory")
    return target


def has_supported_video_extension(filename: str | None) -> bool:
    return get_extension(filename) in SUPPORTED_VIDEO_EXTENSIONS
