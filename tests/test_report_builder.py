from app.services.report_builder import LIMITATIONS, build_verification_report


def test_report_includes_limitations():
    report = build_verification_report(
        {"risk_level": "Suspicious", "trust_score": 50, "decision": "review"},
        {"signal_scores": {}},
    )

    assert report["limitations"] == LIMITATIONS
    assert "sampled frames" in report["limitations"][0]


def test_report_avoids_absolute_claims():
    report = build_verification_report(
        {"risk_level": "High Risk", "trust_score": 30, "decision": "manual_review"},
        {"signal_scores": {"audio_risk": 0.9}},
    )

    summary = report["summary"].lower()
    assert ("definitely " + "fake") not in summary
    assert ("definitely " + "real") not in summary
    assert "absolute proof" in summary


def test_risk_flags_deduplicated():
    report = build_verification_report(
        {"risk_level": "Suspicious", "trust_score": 50, "decision": "review"},
        {"signal_scores": {}},
        risk_flags=["same", "same", "other"],
    )

    assert report["risk_flags"] == ["same", "other"]


def test_summary_changes_based_on_evidence():
    audio_report = build_verification_report(
        {"risk_level": "High Risk", "trust_score": 25, "decision": "manual_review"},
        {"signal_scores": {"audio_risk": 0.9}},
    )
    visual_report = build_verification_report(
        {"risk_level": "Suspicious", "trust_score": 45, "decision": "review"},
        {"signal_scores": {"visual_multisignal_risk": 0.7}},
    )

    assert audio_report["summary"] != visual_report["summary"]
    assert "audio model" in audio_report["summary"]
    assert "sampled frames" in visual_report["summary"]
