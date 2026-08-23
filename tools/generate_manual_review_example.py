#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

from pallet_audit.geometry import HomographyCalibration
from pallet_audit.pipeline import AssessmentEngine, PalletObservation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate and schema-check the honest missing-evidence assessment example."
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("schemas/pallet_assessment.schema.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/manual_review_assessment.json"),
    )
    args = parser.parse_args()

    placeholder = HomographyCalibration(
        calibration_id="missing-physical-calibration",
        image_to_floor=np.eye(3),
        reprojection_rmse_px=999.0,
        floor_frame="UNAVAILABLE",
    )
    observation = PalletObservation(
        frame_id="example-frame-without-physical-evidence",
        track_id="example-track-001",
        footprint_keypoints={},
        sop_evidence={},
        model_versions={
            "detector": "detect-real-v1@36273b60fb817a0d",
            "segmenter": "segment-real-v1@79aaaa6c147ef566",
            "pose": "UNAVAILABLE:no-keypoint-labels",
            "damage": "UNAVAILABLE:no-damage-labels",
        },
    )
    payload = AssessmentEngine(placeholder).assess(observation).to_dict()
    payload["assessment_id"] = "example-manual-review-001"
    payload["created_at_utc"] = "2026-08-24T00:00:00+05:30"

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(payload)
    if payload["overall_verdict"] != "MANUAL_REVIEW":
        raise SystemExit("missing evidence must never produce PASS or FAIL")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
