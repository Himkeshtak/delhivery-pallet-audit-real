from pallet_audit.contracts import Evidence, PoseEstimate, Verdict
from pallet_audit.sop import aggregate_verdict, evaluate_sop


def _pose() -> PoseEstimate:
    return PoseEstimate(
        available=True,
        x_m=1.0,
        y_m=2.0,
        theta_deg=30.0,
        visible_face="front",
        covariance_xy_m2=[[0.0001, 0.0], [0.0, 0.0001]],
        theta_std_deg=1.0,
        confidence=0.9,
        reprojection_rmse_px=0.4,
    )


def _passing_evidence() -> dict[str, Evidence]:
    return {
        "overhang_m": Evidence(0.005, 0.9, standard_deviation=0.003, source="masks"),
        "height_m": Evidence(1.5, 0.9, standard_deviation=0.03, source="geometry"),
        "maximum_box_rotation_deg": Evidence(
            3.0, 0.8, standard_deviation=1.0, source="box edges"
        ),
        "size_inversion_probability": Evidence(0.05, 0.8, coverage=0.9, source="boxes"),
        "wrap_probability": Evidence(0.95, 0.9, coverage=0.9, source="classifier"),
        "box_damage_probability": Evidence(0.05, 0.9, coverage=0.9, source="EfficientAD"),
        "centroid_offset_m": Evidence(0.02, 0.8, standard_deviation=0.01, source="masks"),
        "pallet_damage_probability": Evidence(
            0.05, 0.9, coverage=0.9, source="EfficientAD"
        ),
    }


def test_all_eight_checks_pass_with_strong_evidence() -> None:
    checks = evaluate_sop(_passing_evidence(), _pose())
    assert len(checks) == 8
    assert all(check.verdict is Verdict.PASS for check in checks)
    overall, confidence, _ = aggregate_verdict(checks, _pose())
    assert overall is Verdict.PASS
    assert confidence > 0


def test_uncertain_threshold_crossing_requires_manual_review() -> None:
    evidence = _passing_evidence()
    evidence["overhang_m"] = Evidence(
        0.028, 0.95, standard_deviation=0.005, source="masks"
    )
    checks = evaluate_sop(evidence, _pose())
    assert checks[0].verdict is Verdict.MANUAL_REVIEW


def test_confirmed_visible_damage_fails_even_when_pose_is_unavailable() -> None:
    evidence = _passing_evidence()
    evidence["box_damage_probability"] = Evidence(
        0.9, 0.95, coverage=0.9, source="EfficientAD"
    )
    unavailable = PoseEstimate.unavailable("occluded")
    checks = evaluate_sop(evidence, unavailable)
    overall, _, reasons = aggregate_verdict(checks, unavailable)
    assert checks[5].verdict is Verdict.FAIL
    assert overall is Verdict.FAIL
    assert "SOP-PAL-03-6" in reasons[0]

