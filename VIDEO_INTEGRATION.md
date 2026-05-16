# BitCheck Video Backend Integration Guide

This document outlines how to integrate the BitCheck Video Verification Service—currently hosted on Hugging Face Spaces—into your existing backend architecture.

## Base URL

When deployed on Hugging Face Spaces, your Space operates as a standard REST API.
The base URL follows this pattern:

```text
https://jaykay73-bitcheck-video.hf.space
```

*(Note: Replace `jaykay73-bitcheck-video` with the actual subdomain provided by Hugging Face if you deploy to a different space name.)*

## Authentication

By default, Hugging Face Spaces are public. If you configured your space as private, you will need to pass a Hugging Face token in the `Authorization` header.
```http
Authorization: Bearer YOUR_HF_TOKEN
```

## Endpoints

### 1. Verify Video
**Endpoint:** `POST /verify/video`  
**Content-Type:** `multipart/form-data`

This is the primary endpoint to submit a video for deepfake and risk verification.

**Form Parameters:**
- `file` *(Required)*: The video file to be analyzed.
- `sample_mode` *(Optional)*: Strategy for extracting frames (e.g., `uniform`). Default: `uniform`.
- `max_frames` *(Optional)*: Maximum number of frames to extract and analyze. Default: `12`.
- `run_image_analysis` *(Optional)*: Whether to run the visual model on extracted frames. Default: `true`.
- `run_gradcam` *(Optional)*: Whether to generate Grad-CAM visual heatmaps. Default: `true`.
- `run_audio_analysis` *(Optional)*: Whether to extract and run Random Forest audio analysis. Default: `true`.
- `run_forensics` *(Optional)*: Whether to run visual forensic analysis. Default: `true`.
- `run_watermark_analysis` *(Optional)*: Whether to check for watermarks. Default: `true`.
- `allow_trim_to_5_seconds` *(Optional)*: If the video exceeds the 5-second limit, trim it instead of throwing an error. Default: `false`.

**Example Request (cURL):**
```bash
curl -X POST https://jaykay73-bitcheck-video.hf.space/verify/video \
  -F "file=@/path/to/suspicious_video.mp4" \
  -F "sample_mode=uniform" \
  -F "max_frames=12" \
  -F "allow_trim_to_5_seconds=true"
```

**Example Response:**
Returns a JSON object detailing trust score, risk level, extracted frames, audio features, and temporal consistency.
```json
{
  "verification_id": "a1b2c3d4-e5f6-7890",
  "service": "BitCheck",
  "file_type": "video",
  "status": "completed",
  "trust": {
    "trust_score": 50,
    "risk_score": 0.5,
    "risk_level": "Suspicious",
    "decision": "review",
    "summary": "..."
  },
  "frame_sampling": { ... },
  "visual_analysis": { ... },
  "audio_analysis": { ... },
  "temporal_analysis": { ... },
  "aggregation": { ... },
  "limitations": [ ... ],
  "warnings": [ ... ]
}
```

### 2. Retrieve Generated Assets (e.g., Grad-CAM Images)
**Endpoint:** `GET /outputs/{filename}`

If you have requested Grad-CAM or other frame analyses, the `POST /verify/video` endpoint will return URLs to those visual assets inside `url` fields (e.g., `url: "/outputs/xxx_frame_001.jpg"`). 

**Example Request:**
```bash
curl -O https://jaykay73-bitcheck-video.hf.space/outputs/uuid_frame_001.jpg
```

### 3. Service Health
**Endpoint:** `GET /health`

Use this endpoint in your backend cron jobs or monitoring services to ensure the video verification service is online and dependencies (like ffmpeg and models) are correctly loaded.

**Example Response:**
```json
{
  "status": "ok",
  "service": "BitCheck Video Verification API",
  "version": "1.0.0",
  "audio_model_found": true,
  "image_model_found": true,
  "ffmpeg_available": true
}
```

## Backend Integration Best Practices

1. **Timeouts:** Video processing takes time (specifically frame extraction and ML inference). Configure your HTTP client timeout appropriately (e.g., 30-60 seconds).
2. **Asynchronous Handling:** For user-facing applications, consider queueing the video on your backend and polling or using webhooks rather than blocking the user's request. However, this API currently processes synchronously, so keep video sizes small (max 50MB and max 5 seconds by default).
3. **Error Handling:** If the video is corrupted or lacks metadata, the API will return a 4xx error. Ensure your backend handles these gracefully.
4. **File Passing:** When receiving a video upload from your frontend, pipe the stream directly to the BitCheck API instead of saving it entirely into memory on your backend, to minimize memory usage.
