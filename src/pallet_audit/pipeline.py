from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .contracts import Evidence, KeypointObservation, PalletAssessment, PoseEstimate
from .geometry import HomographyCalibration, PosePolicy, estimate_floor_pose
from .sop import SopPolicy, aggregate_verdict, evaluate_sop


@dataclass(frozen=True)
class PalletObservation:
    frame_id: str
    track_id: str
    footprint_keypoints: Mapping[str, KeypointObservation]
    sop_evidence: Mapping[str, Evidence]
    model_versions: dict[str, str] = field(default_factory=dict)


class AssessmentEngine:
    """Deterministic geometry, abstention, and SOP fusion behind vision adapters."""

    def __init__(
        self,
        calibration: HomographyCalibration,
        pose_policy: PosePolicy | None = None,
        sop_policy: SopPolicy | None = None,
    ) -> None:
        self.calibration = calibration
        self.pose_policy = pose_policy or PosePolicy()
        self.sop_policy = sop_policy or SopPolicy()

    def assess(self, observation: PalletObservation) -> PalletAssessment:
        try:
            pose = estimate_floor_pose(
                observation.footprint_keypoints, self.calibration, self.pose_policy
            )
        except (ValueError, FloatingPointError) as error:
            pose = PoseEstimate.unavailable("pose_estimation_exception", type(error).__name__)
        checks = evaluate_sop(observation.sop_evidence, pose, self.sop_policy)
        verdict, confidence, reasoning = aggregate_verdict(checks, pose, self.sop_policy)
        return PalletAssessment(
            schema_version="1.0.0",
            assessment_id=str(uuid.uuid4()),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            frame_id=observation.frame_id,
            track_id=observation.track_id,
            pose=pose,
            checks=checks,
            overall_verdict=verdict,
            overall_confidence=confidence,
            reasoning=reasoning,
            model_versions=dict(observation.model_versions),
            calibration_id=self.calibration.calibration_id,
        )
