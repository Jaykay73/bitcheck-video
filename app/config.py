from functools import lru_cache
from pathlib import Path
import os
import shutil

from dotenv import load_dotenv


load_dotenv()


class Settings:
    service_name = os.getenv("APP_NAME", "BitCheck Video Verification API")
    version = os.getenv("VERSION", "1.0.0")

    base_dir = Path(__file__).resolve().parent.parent
    uploads_dir = base_dir / os.getenv("UPLOAD_DIR", "uploads")
    outputs_dir = base_dir / os.getenv("OUTPUT_DIR", "outputs")
    models_dir = base_dir / os.getenv("MODEL_DIR", "models")

    expected_audio_model_path = base_dir / os.getenv(
        "AUDIO_MODEL_PATH", "models/BitcheckDeepfake.joblib"
    )
    legacy_audio_model_path = models_dir / "BitcheckDeepFake.joblib"
    image_model_path = base_dir / os.getenv(
        "IMAGE_MODEL_PATH", "models/ai_vs_real_image_detector.pth"
    )

    max_video_duration_seconds = int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "5"))
    max_video_size_mb = int(os.getenv("MAX_VIDEO_SIZE_MB", "50"))
    max_frames_to_analyze = int(os.getenv("MAX_FRAMES_TO_ANALYZE", "12"))
    gradcam_top_k = int(os.getenv("GRADCAM_TOP_K", "3"))
    allow_trim_to_5_seconds = os.getenv("ALLOW_TRIM_TO_5_SECONDS", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "INFO")

    cors_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]

    def ensure_runtime_directories(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @property
    def audio_model_path(self) -> Path:
        if self.expected_audio_model_path.exists():
            return self.expected_audio_model_path
        return self.legacy_audio_model_path

    @property
    def audio_model_found(self) -> bool:
        return self.expected_audio_model_path.exists() or self.legacy_audio_model_path.exists()

    @property
    def image_model_found(self) -> bool:
        return self.image_model_path.exists()

    @property
    def ffmpeg_available(self) -> bool:
        return shutil.which("ffmpeg") is not None


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
