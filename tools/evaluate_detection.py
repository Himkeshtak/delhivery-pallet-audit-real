#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pallet_audit.evaluation import evaluate_coco_predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report COCO-box detection and localization distributions separately."
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/detection_evaluation.json"))
    args = parser.parse_args()
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    report = evaluate_coco_predictions(ground_truth, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

