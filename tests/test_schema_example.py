import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


def test_committed_manual_review_example_matches_contract() -> None:
    schema = json.loads(
        Path("schemas/pallet_assessment.schema.json").read_text(encoding="utf-8")
    )
    payload = json.loads(
        Path("examples/manual_review_assessment.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(payload)
    assert payload["overall_verdict"] == "MANUAL_REVIEW"
    assert len(payload["checks"]) == 8
    assert all(check["verdict"] == "MANUAL_REVIEW" for check in payload["checks"])
