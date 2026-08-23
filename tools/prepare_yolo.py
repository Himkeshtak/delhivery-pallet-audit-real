#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pallet_audit.data.prepare import PreparationFailure, prepare_yolo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a leakage-controlled YOLO dataset from audited COCO data."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("detect", "segment"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = prepare_yolo_dataset(
            args.source,
            args.output,
            args.task,
            seed=args.seed,
            ratios={
                "train": args.train_ratio,
                "valid": args.valid_ratio,
                "test": args.test_ratio,
            },
        )
    except PreparationFailure as error:
        raise SystemExit(f"preparation failed: {error}") from error
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
