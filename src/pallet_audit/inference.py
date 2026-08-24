from __future__ import annotations

import uuid
from datetime import datetime, timezone

from .contracts import PalletAssessment, PoseEstimate
from .sop import SopPolicy, aggregate_verdict, evaluate_sop


def unavailable_perception_assessment(
    *,
    frame_id: str,
    track_id: str,
    model_versions: dict[str, str],
    reasons: list[str],
    calibration_id: str | None = None,
    policy: SopPolicy | None = None,
) -> PalletAssessment:
    """Build a schema-valid fail-safe assessment when metric evidence is absent."""
    pose = PoseEstimate.unavailable(*reasons)
    checks = evaluate_sop({}, pose, policy)
    verdict, confidence, reasoning = aggregate_verdict(checks, pose, policy)
    return PalletAssessment(
        schema_version="1.0.0",
        assessment_id=str(uuid.uuid4()),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        frame_id=frame_id,
        track_id=track_id,
        pose=pose,
        checks=checks,
        overall_verdict=verdict,
        overall_confidence=confidence,
        reasoning=reasoning,
        model_versions=model_versions,
        calibration_id=calibration_id,
    )

