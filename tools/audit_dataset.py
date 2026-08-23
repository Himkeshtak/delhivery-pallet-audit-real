#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pallet_audit.data.audit import AuditFailure, audit_many


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate downloaded COCO exports and report leakage."
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_audit.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = sorted(path for path in args.input.iterdir() if path.is_dir())
    if not roots:
        raise SystemExit(f"no downloaded sources under {args.input}")
    try:
        report = audit_many(roots)
    except AuditFailure as error:
        raise SystemExit(f"dataset audit failed: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
