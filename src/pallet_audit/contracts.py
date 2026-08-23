from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class Verifiability(str, Enum):
    VERIFIABLE = "VERIFIABLE"
    PARTIAL = "PARTIAL"
    NOT_VERIFIABLE = "NOT_VERIFIABLE"


@dataclass(frozen=True)
class KeypointObservation:
    x_px: float
    y_px: float
    confidence: float


@dataclass(frozen=True)
class PoseEstimate:
    available: bool
    x_m: float | None = None
    y_m: float | None = None
    theta_deg: float | None = None
    visible_face: str | None = None
    covariance_xy_m2: list[list[float]] | None = None
    theta_std_deg: float | None = None
    confidence: float = 0.0
    reprojection_rmse_px: float | None = None
    reasons: list[str] = field(default_factory=list)

    @classmethod
    def unavailable(cls, *reasons: str) -> PoseEstimate:
        return cls(available=False, confidence=0.0, reasons=list(reasons))


@dataclass(frozen=True)
class Evidence:
    value: float | bool | str | None
    confidence: float
    coverage: float = 1.0
    standard_deviation: float | None = None
    source: str = "unknown"
    units: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    name: str
    scope: Verifiability
    verdict: Verdict
    confidence: float
    reason: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class PalletAssessment:
    schema_version: str
    assessment_id: str
    created_at_utc: str
    frame_id: str
    track_id: str
    pose: PoseEstimate
    checks: list[CheckResult]
    overall_verdict: Verdict
    overall_confidence: float
    reasoning: list[str]
    model_versions: dict[str, str] = field(default_factory=dict)
    calibration_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        def normalize(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, list):
                return [normalize(item) for item in value]
            return value

        return normalize(asdict(self))

