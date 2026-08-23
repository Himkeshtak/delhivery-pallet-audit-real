#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pallet_audit.data.prepare import PreparationFailure, validate_prepared_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a prepared YOLO dataset.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--task", choices=("detect", "segment"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = validate_prepared_yolo(args.input, args.task)
    except PreparationFailure as error:
        raise SystemExit(f"validation failed: {error}") from error
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
