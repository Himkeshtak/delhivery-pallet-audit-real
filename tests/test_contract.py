from pallet_audit.contracts import (
    CheckResult,
    PalletAssessment,
    PoseEstimate,
    Verdict,
    Verifiability,
)


def test_assessment_serializes_enums_as_contract_values() -> None:
    checks = [
        CheckResult(
            f"SOP-PAL-03-{index}",
            "check",
            Verifiability.PARTIAL,
            Verdict.MANUAL_REVIEW,
            0.2,
            "insufficient evidence",
            {},
        )
        for index in range(1, 9)
    ]
    assessment = PalletAssessment(
        "1.0.0",
        "assessment-1",
        "2026-08-23T00:00:00+00:00",
        "frame-1",
        "track-1",
        PoseEstimate.unavailable("missing calibration"),
        checks,
        Verdict.MANUAL_REVIEW,
        0.0,
        ["manual review required"],
    )
    payload = assessment.to_dict()
    assert payload["overall_verdict"] == "MANUAL_REVIEW"
    assert payload["checks"][0]["scope"] == "PARTIAL"
