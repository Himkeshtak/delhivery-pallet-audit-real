#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pallet_audit.data.audit import AuditFailure, audit_many
from pallet_audit.data.config import load_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate downloaded COCO exports and report leakage."
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_audit.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/data_sources.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_sources(args.config)
    roots = [
        args.input / source.id
        for source in config.sources
        if (args.input / source.id).is_dir()
    ]
    if not roots:
        raise SystemExit(f"no downloaded sources under {args.input}")
    try:
        report = audit_many(roots)
    except AuditFailure as error:
        raise SystemExit(f"dataset audit failed: {error}") from error
    expected_tasks = {source.id: source.expected_task for source in config.sources}
    mismatches = {
        source_id: {
            "expected": expected_tasks[source_id],
            "inferred": source_report["inferred_task"],
        }
        for source_id, source_report in report["sources"].items()
        if source_report["inferred_task"] != expected_tasks[source_id]
    }
    report["gates"] = {
        "expected_tasks_match": not mismatches,
        "task_mismatches": mismatches,
    }
    if mismatches:
        raise SystemExit(f"dataset audit failed: task mismatch: {mismatches}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
