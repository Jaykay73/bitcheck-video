from app.services.temporal_analyzer import analyze_temporal_consistency


def test_isolated_high_risk_frame_does_not_produce_extreme_temporal_risk():
    frames = [
        {"risk_score": 0.92, "flags": ["noise_inconsistency"]},
        {"risk_score": 0.12, "flags": []},
        {"risk_score": 0.18, "flags": []},
        {"risk_score": 0.22, "flags": []},
        {"risk_score": 0.20, "flags": []},
    ]

    result = analyze_temporal_consistency(frames)

    assert result["high_risk_frame_count"] == 1
    assert result["temporal_consistency_risk"] <= 0.55
    assert "not consistent" in result["flags"][-1]


def test_repeated_high_risk_frames_increase_temporal_risk():
    frames = [
        {"risk_score": 0.78, "flags": ["noise_inconsistency"]},
        {"risk_score": 0.82, "flags": ["noise_inconsistency"]},
        {"risk_score": 0.74, "flags": ["noise_inconsistency"]},
        {"risk_score": 0.66, "flags": ["noise_inconsistency"]},
        {"risk_score": 0.30, "flags": []},
    ]

    result = analyze_temporal_consistency(frames)

    assert result["high_risk_frame_count"] == 4
    assert result["repeated_signal_consistency"] >= 0.60
    assert result["temporal_consistency_risk"] >= 0.70
    assert result["flags"]


def test_empty_frame_list_handled():
    result = analyze_temporal_consistency([])

    assert result["checked"] is True
    assert result["frames_analyzed"] == 0
    assert result["temporal_consistency_risk"] == 0.0
    assert result["warnings"]
