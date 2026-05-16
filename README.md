# BitCheck Video Verification Service

BitCheck Video Verification API is a CPU-friendly FastAPI service for short-video AI generation and deepfake risk review. It combines metadata, sampled frames, weak/moderate image inference, visual forensic heuristics, watermark/provenance hints, optional Grad-CAM, audio feature extraction, Random Forest audio inference, temporal analysis, and multi-signal trust scoring.

## Architecture

Upload -> file validation -> video metadata -> frame sampling -> frame image analysis -> optional Grad-CAM -> audio extraction -> audio feature extraction -> Random Forest audio analysis -> temporal analysis -> aggregation -> JSON report.

The service is designed for Hugging Face Spaces and runs with:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

## Five Second Demo Limit

The hackathon MVP enforces `MAX_VIDEO_DURATION_SECONDS=5` so CPU inference, frame extraction, audio extraction, and Grad-CAM remain predictable on small hosted hardware. Longer videos return a clean `video_too_long` failure unless `ALLOW_TRIM_TO_5_SECONDS=true` or the request form field `allow_trim_to_5_seconds=true` is explicitly used.

## Model Files

Place these files in `models/`:

- `models/BitcheckDeepfake.joblib`
- `models/ai_vs_real_image_detector.pth`

The audio loader also checks legacy names such as `models/BitcheckDeepFake.joblib`, but the preferred path is `models/BitcheckDeepfake.joblib`. Model binaries are documented and ignored by Git.

## Audio Feature Pipeline

The audio model expects exactly 26 numeric features in this order:

```text
chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff,
zero_crossing_rate, mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6,
mfcc7, mfcc8, mfcc9, mfcc10, mfcc11, mfcc12, mfcc13, mfcc14,
mfcc15, mfcc16, mfcc17, mfcc18, mfcc19, mfcc20
```

Audio is extracted as mono WAV at 22050 Hz for up to 5 seconds. `class 0` is treated as fake/deepfake and `class 1` as real/human when the model exposes numeric classes `[0, 1]`. String labels are mapped by name when possible.

## Image Model Warning

`ai_vs_real_image_detector.pth` is treated as a weak/moderate frame-level signal. It never dominates the video decision by itself. If the image model is missing or cannot be loaded, BitCheck still uses metadata, visual forensics, watermark hints, audio, and temporal aggregation.

## Risk Scoring

Video risk combines available numeric signals with dynamic weighting:

- audio risk
- visual multi-signal risk
- watermark/provenance risk
- temporal consistency risk
- video metadata risk
- filename context risk

Trust score is `round((1 - risk_score) * 100)`.

- `80-100`: Likely Authentic, `approve`
- `60-79`: Low Risk, `approve`
- `40-59`: Suspicious, `review`
- `20-39`: High Risk, `manual_review`
- `0-19`: Very High Risk, `manual_review`

The demo does not use `block`.

## Local Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

If using `uv`:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

System dependencies should include `ffmpeg` and `libsndfile`.

## Environment Variables

See `.env.example`. Key defaults:

```text
APP_NAME=BitCheck Video Verification API
VERSION=1.0.0
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
MODEL_DIR=models
MAX_VIDEO_DURATION_SECONDS=5
MAX_VIDEO_SIZE_MB=50
MAX_FRAMES_TO_ANALYZE=12
GRADCAM_TOP_K=3
AUDIO_MODEL_PATH=models/BitcheckDeepfake.joblib
IMAGE_MODEL_PATH=models/ai_vs_real_image_detector.pth
ALLOW_TRIM_TO_5_SECONDS=false
LOG_LEVEL=INFO
```

## Running The API

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

Health check:

```bash
curl http://localhost:7860/health
```

Verify a video:

```bash
curl -X POST http://localhost:7860/verify/video \
  -F "file=@sample.mp4" \
  -F "sample_mode=uniform" \
  -F "max_frames=12" \
  -F "run_image_analysis=true" \
  -F "run_gradcam=true" \
  -F "run_audio_analysis=true"
```

Skip audio for a visual-only demo:

```bash
curl -X POST http://localhost:7860/verify/video \
  -F "file=@sample.mp4" \
  -F "run_audio_analysis=false"
```

## Example JSON Response

```json
{
  "verification_id": "uuid",
  "service": "BitCheck",
  "file_type": "video",
  "status": "completed_with_warnings",
  "trust": {
    "trust_score": 50,
    "risk_score": 0.5,
    "risk_level": "Suspicious",
    "decision": "review",
    "summary": "The video has mixed or limited signals and should be reviewed based on the available signals. Trust score: 50. The result is risk-based and should not be treated as absolute proof."
  },
  "frame_sampling": {
    "frames_extracted": 8,
    "frames": [
      {
        "frame_id": 1,
        "timestamp": 0.0,
        "path": "outputs/uuid_frame_001.jpg",
        "url": "/outputs/uuid_frame_001.jpg"
      }
    ]
  },
  "limitations": [
    "BitCheck analyzes sampled frames, not every frame.",
    "Video analysis is probabilistic and should not be treated as absolute proof."
  ]
}
```

## Hugging Face Spaces Deployment

1. Create a Docker Space.
2. Upload this repository.
3. Add the model files under `models/` or configure persistent storage/secrets for model download.
4. Build with the provided `Dockerfile`.
5. The container exposes port `7860` and starts `uvicorn main:app --host 0.0.0.0 --port 7860`.

The Docker image installs `ffmpeg`, `libsndfile1`, `libgl1`, and `libglib2.0-0`.

## Testing

```bash
.venv/bin/python -m compileall .
.venv/bin/python -m pytest -q
```

## Limitations

- BitCheck analyzes sampled frames, not every frame.
- Video analysis is probabilistic and should not be treated as absolute proof.
- The image classifier is treated as a weak/moderate signal because it may not generalize to all generators.
- Audio Random Forest inference depends on matching the exact feature extraction used during training.
- Compressed or noisy audio may affect audio model accuracy.
- Social media compression may affect visual forensic signals.
- Grad-CAM shows model attention, not proof of manipulation.
- Missing metadata does not prove a video is fake.
- High-stakes decisions should involve human review.

## Future Improvements

- Stronger video-native deepfake model
- Face tracking
- Lip-sync consistency analysis
- C2PA/provenance verification
- Video watermark detection
- Stronger SynthID/provenance checks where APIs are available
- Audio CNN/spectrogram model
- Audio Grad-CAM or SHAP report
- Database storage for verification history
- Frontend dashboard
