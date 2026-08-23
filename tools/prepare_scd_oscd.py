#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pallet_audit.data.prepare import validate_prepared_yolo
from pallet_audit.data.scd import prepare_oscd_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit official OSCD COCO masks and prepare leakage-controlled YOLO data."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--validation-fraction", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = prepare_oscd_dataset(
        args.source,
        args.output,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        archive=args.archive,
        expected_archive_sha256=args.expected_archive_sha256,
    )
    report["validation"] = validate_prepared_yolo(args.output, "segment")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

