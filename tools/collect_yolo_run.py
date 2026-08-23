#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _number(value: str) -> int | float | str:
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish a portable, weight-hashed summary of a YOLO training run."
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--published-weight", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.run / "run_manifest.json").read_text(encoding="utf-8"))
    with (args.run / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        curve: list[dict[str, Any]] = [
            {key.strip(): _number(value.strip()) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    started = datetime.fromisoformat(manifest["started_at_utc"])
    finished = datetime.fromisoformat(manifest["finished_at_utc"])
    best_weight = dict(manifest["best_weight"])
    best_weight["path"] = "weights/best.pt"
    if args.published_weight:
        published_hash = _sha256(args.published_weight)
        if published_hash != best_weight["sha256"]:
            raise SystemExit("published weight hash does not match the selected checkpoint")
        best_weight["repository_path"] = args.published_weight.as_posix()
    payload = {
        "training_report_schema_version": 1,
        "run": args.run.name,
        "base_commit": args.base_commit,
        "working_tree_note": (
            "Training CLI options were uncommitted at run start and are included in the "
            "next stage commit; dataset preparation was exactly base_commit."
        ),
        "task": manifest["task"],
        "dataset": Path(manifest["data_yaml"]).parent.name,
        "pretrained_model": manifest["pretrained_model"],
        "configuration": {
            key: manifest[key]
            for key in (
                "epochs",
                "image_size",
                "batch",
                "workers",
                "patience",
                "freeze",
                "cache",
                "device",
                "seed",
            )
        },
        "started_at_utc": manifest["started_at_utc"],
        "finished_at_utc": manifest["finished_at_utc"],
        "duration_seconds": (finished - started).total_seconds(),
        "hardware": manifest["hardware"],
        "software": manifest["software"],
        "selected_checkpoint_metrics": manifest["metrics"],
        "best_weight": best_weight,
        "training_curve": curve,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
