from time import perf_counter
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import settings
from app.services.audio_analyzer import analyze_audio_deepfake
from app.services.audio_extractor import extract_video_audio
from app.services.audio_feature_extractor import FEATURE_NAMES, extract_audio_features
from app.services.file_validator import FileValidationError, save_and_validate_video_upload
from app.services.frame_sampler import sample_video_frames
from app.services.image_frame_analyzer import analyze_sampled_frames
from app.services.image_gradcam import generate_gradcam_for_top_frames
from app.services.report_builder import build_verification_report
from app.services.temporal_analyzer import analyze_temporal_consistency
from app.services.video_aggregator import aggregate_video_risk
from app.services.video_metadata_analyzer import VideoDurationError, analyze_video_metadata


router = APIRouter()


@router.get("/")
async def root() -> dict:
    return {
        "service": settings.service_name,
        "status": "running",
        "version": settings.version,
    }


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": settings.service_name,
        "version": settings.version,
        "audio_model_found": settings.audio_model_found,
        "image_model_found": settings.image_model_found,
        "ffmpeg_available": settings.ffmpeg_available,
        "max_video_duration_seconds": settings.max_video_duration_seconds,
        "max_frames_to_analyze": settings.max_frames_to_analyze,
    }


@router.get("/outputs/{filename}")
async def get_output_file(filename: str) -> Response:
    if "/" in filename or "\\" in filename or filename in {"", ".", ".."}:
        raise HTTPException(status_code=404, detail="Output file not found.")

    path = (settings.outputs_dir / filename).resolve()
    output_root = settings.outputs_dir.resolve()
    if output_root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Output file not found.")

    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".wav"}:
        raise HTTPException(status_code=404, detail="Output file not found.")
    return Response(content=path.read_bytes(), media_type=_media_type_for_output(path))


@router.post("/verify/video")
async def verify_video(
    file: UploadFile = File(...),
    sample_mode: str = Form("uniform"),
    max_frames: int = Form(12),
    run_image_analysis: bool = Form(True),
    run_gradcam: bool = Form(True),
    run_audio_analysis: bool = Form(True),
    run_forensics: bool = Form(True),
    run_watermark_analysis: bool = Form(True),
    allow_trim_to_5_seconds: bool = Form(False),
) -> dict:
    started_at = perf_counter()
    try:
        validated = await save_and_validate_video_upload(file)
    except FileValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    try:
        metadata = analyze_video_metadata(
            validated.saved_path,
            allow_trim_to_5_seconds=allow_trim_to_5_seconds,
        )
    except VideoDurationError as exc:
        metadata = analyze_video_metadata(
            validated.saved_path,
            allow_trim_to_5_seconds=True,
        )
        metadata.flags = [flag for flag in metadata.flags if flag != "trim_to_demo_limit"]
        metadata.warnings = [
            warning for warning in metadata.warnings if "only the first" not in warning
        ]
        return {
            "verification_id": validated.verification_id,
            "service": "BitCheck",
            "file_type": "video",
            "status": "failed",
            "error": {
                "code": exc.code,
                "message": str(exc),
                "duration_seconds": exc.duration_seconds,
                "max_video_duration_seconds": exc.max_duration_seconds,
            },
            "input": {
                "filename": validated.stored_filename,
                "sha256": validated.sha256,
                "extension": validated.extension,
                "file_size_bytes": validated.file_size_bytes,
            },
            "file_validation": {
                "valid": True,
                "warnings": validated.warnings,
            },
            "video_metadata": metadata.to_dict(),
        }

    input_payload = {
        "filename": validated.stored_filename,
        "sha256": validated.sha256,
        "extension": validated.extension,
        "file_size_bytes": validated.file_size_bytes,
    }
    file_validation = {
        "valid": True,
        "warnings": validated.warnings,
    }

    if not metadata.checked:
        return {
            "verification_id": validated.verification_id,
            "service": "BitCheck",
            "file_type": "video",
            "status": "failed",
            "processing_time_ms": _elapsed_ms(started_at),
            "error": {
                "code": "unreadable_video",
                "message": "Video metadata could not be read; the uploaded file may be corrupt or unsupported.",
            },
            "input": input_payload,
            "file_validation": file_validation,
            "video_metadata": metadata.to_dict(),
        }

    frame_sampling = sample_video_frames(
        validated.saved_path,
        verification_id=validated.verification_id,
        sample_mode=sample_mode,
        max_frames=max_frames,
        analyze_first_seconds=metadata.analyze_first_seconds,
    )

    frame_sampling_payload = frame_sampling.to_dict()
    warnings: list[str] = []
    risk_flags: list[str] = []
    warnings.extend(validated.warnings)
    warnings.extend(metadata.warnings)
    warnings.extend(frame_sampling.warnings)

    if run_image_analysis:
        visual_analysis = analyze_sampled_frames(
            frame_sampling_payload.get("frames", []),
            verification_id=validated.verification_id,
            filename_context=validated.original_filename,
        )
        if not run_forensics:
            visual_analysis["warnings"].append("Visual forensic analysis was disabled after frame analysis.")
        if not run_watermark_analysis:
            visual_analysis["warnings"].append("Watermark analysis was disabled after frame analysis.")
    else:
        visual_analysis = {
            "checked": False,
            "frames_analyzed": 0,
            "high_risk_frames": 0,
            "mean_frame_risk": 0.0,
            "max_frame_risk": 0.0,
            "top_suspicious_frames": [],
            "frames": [],
            "warnings": ["Frame image analysis was skipped by request."],
        }
    warnings.extend(visual_analysis.get("warnings", []))

    gradcam = generate_gradcam_for_top_frames(
        visual_analysis.get("frames", []),
        enabled=bool(run_gradcam and run_image_analysis),
        verification_id=validated.verification_id,
    )
    warnings.extend(gradcam.get("warnings", []))

    audio_extraction_payload: dict[str, Any]
    audio_analysis: dict[str, Any]
    if run_audio_analysis:
        audio_extraction = extract_video_audio(
            validated.saved_path,
            verification_id=validated.verification_id,
            metadata=metadata,
            max_seconds=metadata.analyze_first_seconds,
        )
        audio_extraction_payload = audio_extraction.to_dict()
        warnings.extend(audio_extraction.warnings)
        if audio_extraction.status == "audio_extracted" and audio_extraction.audio_path:
            feature_result = extract_audio_features(
                settings.base_dir / audio_extraction.audio_path,
                max_duration_seconds=metadata.analyze_first_seconds or settings.max_video_duration_seconds,
            )
            warnings.extend(feature_result.warnings)
            model_result = analyze_audio_deepfake(feature_result)
            audio_analysis = {
                **model_result.to_dict(),
                "features_used": FEATURE_NAMES,
                "feature_extraction": feature_result.to_dict(),
            }
            warnings.extend(model_result.warnings)
        else:
            audio_analysis = {
                "checked": audio_extraction.checked,
                "model_found": settings.audio_model_found,
                "risk_score": None,
                "features_used": FEATURE_NAMES,
                "warnings": audio_extraction.warnings,
            }
    else:
        audio_extraction_payload = {
            "checked": False,
            "status": "skipped",
            "audio_present": metadata.audio_present,
            "warnings": ["Audio extraction and analysis were skipped by request."],
        }
        audio_analysis = {
            "checked": False,
            "model_found": settings.audio_model_found,
            "risk_score": None,
            "features_used": FEATURE_NAMES,
            "warnings": ["Audio analysis was skipped by request."],
        }
        warnings.extend(audio_analysis["warnings"])

    temporal_analysis = analyze_temporal_consistency(visual_analysis.get("frames", []))
    warnings.extend(temporal_analysis.get("warnings", []))
    risk_flags.extend(temporal_analysis.get("flags", []))

    aggregation = aggregate_video_risk(
        audio_analysis=audio_analysis,
        visual_analysis=visual_analysis,
        temporal_analysis=temporal_analysis,
        video_metadata=metadata.to_dict(),
        filename_risk=_filename_risk(validated.original_filename),
    )
    risk_flags.extend(metadata.flags)
    risk_flags.extend(aggregation.get("flags", []))
    trust = {
        "trust_score": aggregation["trust_score"],
        "risk_score": aggregation["risk_score"],
        "risk_level": aggregation["risk_level"],
        "decision": aggregation["decision"],
        "summary": aggregation["summary"],
    }
    report = build_verification_report(trust, aggregation, risk_flags, warnings)
    trust["summary"] = report["summary"]
    status = "completed_with_warnings" if warnings else "completed"
    return {
        "verification_id": validated.verification_id,
        "service": "BitCheck",
        "file_type": "video",
        "status": status,
        "processing_time_ms": _elapsed_ms(started_at),
        "input": input_payload,
        "file_validation": file_validation,
        "video_metadata": metadata.to_dict(),
        "frame_sampling": frame_sampling_payload,
        "visual_analysis": visual_analysis,
        "gradcam": gradcam,
        "audio_extraction": audio_extraction_payload,
        "audio_analysis": audio_analysis,
        "temporal_analysis": temporal_analysis,
        "aggregation": aggregation,
        "trust": trust,
        "risk_flags": report["risk_flags"],
        "recommended_actions": report["recommended_actions"],
        "limitations": report["limitations"],
        "warnings": report["warnings"],
    }


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)


def _media_type_for_output(path: Any) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".wav":
        return "audio/wav"
    return "application/octet-stream"


def _filename_risk(filename: str) -> float:
    lowered = filename.lower()
    terms = ["ai", "generated", "deepfake", "runway", "pika", "sora", "synthesia", "heygen"]
    return 0.35 if any(term in lowered for term in terms) else 0.0


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))
