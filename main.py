from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes.verify_video import router as verify_video_router


def create_app() -> FastAPI:
    settings.ensure_runtime_directories()

    app = FastAPI(
        title=settings.service_name,
        version=settings.version,
        description="BitCheck short video verification service.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(verify_video_router)

    return app


app = create_app()
