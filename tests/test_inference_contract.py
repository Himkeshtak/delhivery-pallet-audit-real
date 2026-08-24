import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from pallet_audit.inference import unavailable_perception_assessment


def test_unavailable_perception_assessment_is_schema_valid() -> None:
    assessment = unavailable_perception_assessment(
        frame_id="frame.jpg",
        track_id="pallet-001",
        model_versions={"detector": "real@123"},
        reasons=["no_calibration", "no_pose_keypoints"],
    )
    payload = assessment.to_dict()
    schema = json.loads(
        Path("schemas/pallet_assessment.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(payload)
    assert payload["overall_verdict"] == "MANUAL_REVIEW"
    assert len(payload["checks"]) == 8
    assert all(check["verdict"] == "MANUAL_REVIEW" for check in payload["checks"])

