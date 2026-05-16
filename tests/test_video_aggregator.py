from app.services.video_aggregator import aggregate_video_risk


def test_classifier_only_high_risk_capped_at_suspicious():
    result = aggregate_video_risk(
        visual_analysis={
            "mean_frame_risk": 0.82,
            "max_frame_risk": 0.92,
            "frames": [
                {
                    "risk_score": 0.92,
                    "classifier": {"classifier_risk": 0.95},
                    "forensics": {"risk_score": 0.0},
                    "watermark": {"risk_score": 0.0, "possible_watermark_found": False},
                }
            ],
        },
        temporal_analysis={
            "mean_frame_risk": 0.82,
            "top_20_percent_frame_risk_mean": 0.92,
            "suspicious_frame_ratio": 1.0,
            "max_frame_risk": 0.92,
            "repeated_signal_consistency": 0.0,
            "temporal_consistency_risk": 0.0,
        },
    )

    assert result["risk_level"] == "Suspicious"
    assert result["decision"] == "review"
    assert result["risk_score"] <= 0.60


def test_audio_visual_agreement_gives_high_risk():
    result = aggregate_video_risk(
        audio_analysis={"risk_score": 0.88, "fake_probability": 0.88},
        visual_analysis={"mean_frame_risk": 0.58, "max_frame_risk": 0.80},
        temporal_analysis={
            "mean_frame_risk": 0.58,
            "top_20_percent_frame_risk_mean": 0.80,
            "suspicious_frame_ratio": 0.75,
            "max_frame_risk": 0.80,
            "repeated_signal_consistency": 0.60,
            "temporal_consistency_risk": 0.66,
        },
    )

    assert result["risk_level"] in {"High Risk", "Very High Risk"}
    assert result["decision"] == "manual_review"


def test_missing_audio_redistributes_weights():
    result = aggregate_video_risk(
        audio_analysis={"risk_score": None},
        visual_analysis={"mean_frame_risk": 0.55, "max_frame_risk": 0.72},
        temporal_analysis={
            "mean_frame_risk": 0.55,
            "top_20_percent_frame_risk_mean": 0.72,
            "suspicious_frame_ratio": 0.66,
            "max_frame_risk": 0.72,
            "repeated_signal_consistency": 0.40,
            "temporal_consistency_risk": 0.57,
        },
    )

    assert "audio_risk" not in result["weighted_contributions"]
    assert result["risk_score"] > 0.0


def test_watermark_override_sets_high_risk():
    frames = [
        {"watermark": {"possible_watermark_found": True, "risk_score": 0.65, "flags": ["possible visible watermark"]}},
        {"watermark": {"possible_watermark_found": True, "risk_score": 0.65, "flags": ["bottom-right watermark-like artifact"]}},
    ]

    result = aggregate_video_risk(
        visual_analysis={"mean_frame_risk": 0.30, "max_frame_risk": 0.40, "frames": frames},
        temporal_analysis={"temporal_consistency_risk": 0.25},
    )

    assert result["risk_level"] in {"High Risk", "Very High Risk"}
    assert result["decision"] == "manual_review"


def test_no_evidence_returns_suspicious_review():
    result = aggregate_video_risk()

    assert result["risk_level"] == "Suspicious"
    assert result["decision"] == "review"
    assert result["trust_score"] == 50
