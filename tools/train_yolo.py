#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODELS = {
    "detect": "yolo11n.pt",
    "segment": "yolo11n-seg.pt",
    "pose": "yolo11n-pose.pt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_plan(task: str, data_path: Path, model: str) -> dict[str, Any]:
    if task not in DEFAULT_MODELS:
        raise ValueError(f"unsupported task: {task}")
    if not data_path.is_file():
        raise ValueError(f"dataset YAML does not exist: {data_path}")
    payload = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "train" not in payload or "val" not in payload:
        raise ValueError("dataset YAML must define train and val")
    if task == "pose" and "kpt_shape" not in payload:
        raise ValueError(
            "pose training requires human-verified keypoint labels and kpt_shape; "
            "boxes are not pose labels"
        )
    if task == "segment" and "seg" not in Path(model).stem.lower():
        raise ValueError("segment task requires a segmentation checkpoint such as yolo11n-seg.pt")
    if task == "pose" and "pose" not in Path(model).stem.lower():
        raise ValueError("pose task requires a pose checkpoint such as yolo11n-pose.pt")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO11 task with an auditable manifest."
    )
    parser.add_argument("--task", choices=sorted(DEFAULT_MODELS), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project", type=Path, default=Path("runs/yolo"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_name = args.model or DEFAULT_MODELS[args.task]
    validate_training_plan(args.task, args.data, model_name)
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("install the ML extras: python -m pip install -e '.[ml]'") from error

    device = args.device
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    started = datetime.now(timezone.utc)
    model = YOLO(model_name)
    results = model.train(
        task=args.task,
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=device,
        seed=args.seed,
        deterministic=True,
        project=str(args.project),
        name=args.run_name,
        exist_ok=False,
        plots=True,
        save=True,
    )
    run_directory = Path(results.save_dir)
    best_weight = run_directory / "weights" / "best.pt"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    manifest = {
        "manifest_schema_version": 1,
        "task": args.task,
        "data_yaml": str(args.data.resolve()),
        "data_yaml_sha256": sha256_file(args.data),
        "pretrained_model": model_name,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch": args.batch,
        "device": str(device),
        "seed": args.seed,
        "git_commit": git_commit,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "software": {"ultralytics": ultralytics.__version__},
        "metrics": {key: float(value) for key, value in results.results_dict.items()},
        "best_weight": {
            "path": str(best_weight),
            "sha256": sha256_file(best_weight) if best_weight.exists() else None,
            "bytes": best_weight.stat().st_size if best_weight.exists() else None,
        },
    }
    (run_directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
