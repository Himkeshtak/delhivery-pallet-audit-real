from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from .contracts import CheckResult, Evidence, PoseEstimate, Verdict, Verifiability


@dataclass(frozen=True)
class SopPolicy:
    uncertainty_z: float = 1.96
    overhang_limit_m: float = 0.03
    height_limit_m: float = 1.80
    box_rotation_limit_deg: float = 15.0
    centroid_offset_limit_m: float = 0.10
    minimum_visible_coverage: float = 0.70
    appearance_pass_probability: float = 0.15
    appearance_fail_probability: float = 0.65
    wrap_pass_probability: float = 0.80
    wrap_fail_probability: float = 0.20
    manual_blocks_overall_pass: bool = True


CHECKS = (
    ("SOP-PAL-03-1", "No box overhang above 3 cm", Verifiability.PARTIAL),
    ("SOP-PAL-03-2", "Load height at most 1.8 m", Verifiability.PARTIAL),
    ("SOP-PAL-03-3", "Columns aligned within 15 degrees", Verifiability.PARTIAL),
    ("SOP-PAL-03-4", "No box-size inversion", Verifiability.PARTIAL),
    ("SOP-PAL-03-5", "Load stretch-wrapped", Verifiability.PARTIAL),
    ("SOP-PAL-03-6", "No visibly damaged or crushed box", Verifiability.PARTIAL),
    ("SOP-PAL-03-7", "Load centroid within 10 cm", Verifiability.PARTIAL),
    ("SOP-PAL-03-8", "No visible pallet damage", Verifiability.PARTIAL),
)

FEATURES = (
    "overhang_m",
    "height_m",
    "maximum_box_rotation_deg",
    "size_inversion_probability",
    "wrap_probability",
    "box_damage_probability",
    "centroid_offset_m",
    "pallet_damage_probability",
)

POSE_DEPENDENT = {0, 1, 2, 6}


def _evidence_payload(evidence: Evidence | None) -> dict[str, object]:
    return {} if evidence is None else asdict(evidence)


def _manual(index: int, reason: str, evidence: Evidence | None = None) -> CheckResult:
    check_id, name, scope = CHECKS[index]
    confidence = 0.0 if evidence is None else float(np.clip(evidence.confidence, 0.0, 1.0))
    return CheckResult(
        check_id,
        name,
        scope,
        Verdict.MANUAL_REVIEW,
        confidence,
        reason,
        _evidence_payload(evidence),
    )


def _bounded_check(
    index: int,
    evidence: Evidence | None,
    limit: float,
    policy: SopPolicy,
    pose: PoseEstimate,
) -> CheckResult:
    if evidence is None or not isinstance(evidence.value, (int, float)):
        return _manual(index, "required metric evidence is missing", evidence)
    if evidence.coverage < policy.minimum_visible_coverage:
        return _manual(index, "visible coverage is below policy", evidence)
    if index in POSE_DEPENDENT and not pose.available:
        return _manual(index, "reliable metric pose is unavailable", evidence)

    value = float(evidence.value)
    deviation = evidence.standard_deviation
    if deviation is None:
        return _manual(index, "measurement uncertainty is missing", evidence)
    lower = value - policy.uncertainty_z * deviation
    upper = value + policy.uncertainty_z * deviation
    if upper <= limit:
        verdict = Verdict.PASS
        reason = f"upper {policy.uncertainty_z:.2f}σ bound {upper:.4f} <= limit {limit:.4f}"
    elif lower > limit:
        verdict = Verdict.FAIL
        reason = f"lower {policy.uncertainty_z:.2f}σ bound {lower:.4f} > limit {limit:.4f}"
    else:
        return _manual(
            index,
            f"uncertainty interval [{lower:.4f}, {upper:.4f}] crosses limit",
            evidence,
        )
    confidence = evidence.confidence * (pose.confidence if index in POSE_DEPENDENT else 1.0)
    check_id, name, scope = CHECKS[index]
    return CheckResult(
        check_id,
        name,
        scope,
        verdict,
        float(np.clip(confidence, 0.0, 1.0)),
        reason,
        _evidence_payload(evidence),
    )


def _probability_check(
    index: int,
    evidence: Evidence | None,
    pass_at_or_below: float,
    fail_at_or_above: float,
    policy: SopPolicy,
) -> CheckResult:
    if evidence is None or not isinstance(evidence.value, (int, float)):
        return _manual(index, "required probability evidence is missing", evidence)
    if evidence.coverage < policy.minimum_visible_coverage:
        return _manual(index, "visible coverage is below policy", evidence)
    if evidence.metadata.get("score_not_calibrated"):
        return _manual(index, "risk score has not been calibrated as a probability", evidence)
    probability = float(evidence.value)
    if not 0.0 <= probability <= 1.0:
        return _manual(index, "probability is outside [0, 1]", evidence)
    if probability <= pass_at_or_below:
        verdict, reason = Verdict.PASS, f"probability {probability:.3f} <= {pass_at_or_below:.3f}"
    elif probability >= fail_at_or_above:
        verdict, reason = Verdict.FAIL, f"probability {probability:.3f} >= {fail_at_or_above:.3f}"
    else:
        return _manual(index, "probability lies in the abstention band", evidence)
    check_id, name, scope = CHECKS[index]
    confidence = evidence.confidence * evidence.coverage
    return CheckResult(
        check_id,
        name,
        scope,
        verdict,
        float(np.clip(confidence, 0.0, 1.0)),
        reason,
        _evidence_payload(evidence),
    )


def _wrap_check(evidence: Evidence | None, policy: SopPolicy) -> CheckResult:
    index = 4
    if evidence is None or not isinstance(evidence.value, (int, float)):
        return _manual(index, "stretch-wrap probability is missing", evidence)
    if evidence.coverage < policy.minimum_visible_coverage:
        return _manual(index, "not enough visible surface to judge wrap", evidence)
    probability = float(evidence.value)
    if probability >= policy.wrap_pass_probability:
        verdict, reason = Verdict.PASS, "visible surfaces show continuous stretch wrap"
    elif probability <= policy.wrap_fail_probability:
        verdict, reason = Verdict.FAIL, "visible surfaces show no stretch wrap"
    else:
        return _manual(index, "wrap classifier lies in the abstention band", evidence)
    check_id, name, scope = CHECKS[index]
    return CheckResult(
        check_id,
        name,
        scope,
        verdict,
        float(np.clip(evidence.confidence * evidence.coverage, 0.0, 1.0)),
        reason,
        _evidence_payload(evidence),
    )


def evaluate_sop(
    evidence: Mapping[str, Evidence],
    pose: PoseEstimate,
    policy: SopPolicy | None = None,
) -> list[CheckResult]:
    policy = policy or SopPolicy()
    return [
        _bounded_check(0, evidence.get(FEATURES[0]), policy.overhang_limit_m, policy, pose),
        _bounded_check(1, evidence.get(FEATURES[1]), policy.height_limit_m, policy, pose),
        _bounded_check(2, evidence.get(FEATURES[2]), policy.box_rotation_limit_deg, policy, pose),
        _probability_check(
            3,
            evidence.get(FEATURES[3]),
            policy.appearance_pass_probability,
            policy.appearance_fail_probability,
            policy,
        ),
        _wrap_check(evidence.get(FEATURES[4]), policy),
        _probability_check(
            5,
            evidence.get(FEATURES[5]),
            policy.appearance_pass_probability,
            policy.appearance_fail_probability,
            policy,
        ),
        _bounded_check(6, evidence.get(FEATURES[6]), policy.centroid_offset_limit_m, policy, pose),
        _probability_check(
            7,
            evidence.get(FEATURES[7]),
            policy.appearance_pass_probability,
            policy.appearance_fail_probability,
            policy,
        ),
    ]


def aggregate_verdict(
    checks: list[CheckResult],
    pose: PoseEstimate,
    policy: SopPolicy | None = None,
) -> tuple[Verdict, float, list[str]]:
    policy = policy or SopPolicy()
    failures = [check for check in checks if check.verdict is Verdict.FAIL]
    manual = [check for check in checks if check.verdict is Verdict.MANUAL_REVIEW]
    if failures:
        confidence = min(check.confidence for check in failures)
        return (
            Verdict.FAIL,
            confidence,
            [f"confirmed failure: {check.check_id} — {check.reason}" for check in failures],
        )
    if not pose.available:
        return Verdict.MANUAL_REVIEW, 0.0, ["placement pose is not reliable", *pose.reasons]
    if manual and policy.manual_blocks_overall_pass:
        confidence = min([pose.confidence, *(check.confidence for check in manual)])
        return (
            Verdict.MANUAL_REVIEW,
            confidence,
            [f"unresolved check: {check.check_id} — {check.reason}" for check in manual],
        )
    confidence = min([pose.confidence, *(check.confidence for check in checks)])
    return Verdict.PASS, confidence, ["pose and every required SOP check passed policy"]
