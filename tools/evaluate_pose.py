#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pallet_audit.evaluation import evaluate_pose_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate metric pallet poses as distributions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/pose_evaluation.json"))
    args = parser.parse_args()
    with args.predictions.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    report = evaluate_pose_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

