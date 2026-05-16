from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    audio_model_found: bool
    image_model_found: bool
    ffmpeg_available: bool
    max_video_duration_seconds: int
    max_frames_to_analyze: int

