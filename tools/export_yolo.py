#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO and record the exact artifact.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--format", choices=("onnx", "engine"), required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--data", type=Path, help="required calibration data YAML for INT8")
    parser.add_argument("--output", type=Path, default=Path("artifacts/exports"))
    args = parser.parse_args()
    if args.int8 and args.data is None:
        raise SystemExit("INT8 export requires --data for representative calibration")
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("install the ML extras: python -m pip install -e '.[ml]'") from error

    model = YOLO(str(args.model))
    exported = Path(
        model.export(
            format=args.format,
            imgsz=args.image_size,
            half=args.half,
            int8=args.int8,
            data=str(args.data) if args.data else None,
            dynamic=False,
            simplify=True,
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "export_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_model": {
            "path": str(args.model.resolve()),
            "sha256": _sha256(args.model),
            "bytes": args.model.stat().st_size,
        },
        "exported_model": {
            "path": str(exported.resolve()),
            "sha256": _sha256(exported),
            "bytes": exported.stat().st_size,
        },
        "format": args.format,
        "image_size": args.image_size,
        "half": args.half,
        "int8": args.int8,
        "representative_data": str(args.data.resolve()) if args.data else None,
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "software": {"ultralytics": ultralytics.__version__},
        "accuracy_cost": (
            "PENDING: run the same held-out evaluator on source and exported artifacts"
        ),
    }
    manifest_path = args.output / f"{exported.stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
