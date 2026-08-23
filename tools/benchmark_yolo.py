#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from pallet_audit.evaluation import distribution


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure actual YOLO end-to-end and stage latency on declared hardware."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("reports/runtime_yolo.json"))
    args = parser.parse_args()
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("install the ML extras: python -m pip install -e '.[ml]'") from error

    image_paths = sorted(
        path
        for path in args.images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        raise SystemExit("benchmark image directory is empty")
    frames = [cv2.imread(str(path)) for path in image_paths]
    if any(frame is None for frame in frames):
        raise SystemExit("one or more benchmark images could not be decoded")

    model = YOLO(str(args.model))
    for index in range(args.warmup):
        model.predict(
            frames[index % len(frames)],
            imgsz=args.image_size,
            device=args.device,
            verbose=False,
        )

    wall_ms: list[float] = []
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    for _ in range(args.repeat):
        for frame in frames:
            started = time.perf_counter_ns()
            result = model.predict(
                frame, imgsz=args.image_size, device=args.device, verbose=False
            )[0]
            wall_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            preprocess_ms.append(float(result.speed.get("preprocess", 0.0)))
            inference_ms.append(float(result.speed.get("inference", 0.0)))
            postprocess_ms.append(float(result.speed.get("postprocess", 0.0)))

    payload = {
        "benchmark_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "images": len(frames),
        "warmup_frames": args.warmup,
        "measured_frames": len(wall_ms),
        "image_size": args.image_size,
        "device_requested": args.device,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "software": {"ultralytics": ultralytics.__version__},
        "latency_ms": {
            "end_to_end": distribution(wall_ms),
            "preprocess": distribution(preprocess_ms),
            "inference": distribution(inference_ms),
            "postprocess": distribution(postprocess_ms),
        },
        "throughput_fps_from_total_wall_time": 1000.0 / (sum(wall_ms) / len(wall_ms)),
        "scope": (
            "single YOLO component including in-memory frame input; video decode and "
            "full fusion excluded"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
