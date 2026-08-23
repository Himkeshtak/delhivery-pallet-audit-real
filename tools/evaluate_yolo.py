#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import yaml

from pallet_audit.yolo_evaluation import localization_report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _component_report(component: Any, names: dict[int, str]) -> dict[str, Any]:
    indexes = [int(value) for value in component.ap_class_index.tolist()]
    classes: dict[str, Any] = {}
    for position, class_id in enumerate(indexes):
        classes[names[class_id]] = {
            "class_id": class_id,
            "precision": float(component.p[position]),
            "recall": float(component.r[position]),
            "f1": float(component.f1[position]),
            "ap50": float(component.ap50[position]),
            "ap_by_iou_0_50_to_0_95": [
                float(value) for value in component.all_ap[position].tolist()
            ],
            "map50_95": float(component.ap[position]),
        }
    return {
        "precision_mean": float(component.mp),
        "recall_mean": float(component.mr),
        "map50": float(component.map50),
        "map75": float(component.map75),
        "map50_95": float(component.map),
        "classes": classes,
    }


def _ground_truth(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        class_id = int(tokens[0])
        center_x, center_y, box_width, box_height = (float(value) for value in tokens[1:5])
        x1 = (center_x - box_width / 2) * width
        y1 = (center_y - box_height / 2) * height
        x2 = (center_x + box_width / 2) * width
        y2 = (center_y + box_height / 2) * height
        boxes.append({"class_id": class_id, "box_xyxy": [x1, y1, x2, y2]})
    return boxes


def _draw_worst_cases(
    samples: list[dict[str, Any]],
    worst: list[dict[str, Any]],
    names: dict[int, str],
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    by_image = {str(sample["image"]): sample for sample in samples}
    for rank, summary in enumerate(worst, start=1):
        sample = by_image[str(summary["image"])]
        frame = cv2.imread(str(sample["absolute_image"]))
        if frame is None:
            continue
        for truth in sample["ground_truth"]:
            x1, y1, x2, y2 = (int(round(value)) for value in truth["box_xyxy"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                frame,
                f"GT {names[int(truth['class_id'])]}",
                (x1, max(15, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 200, 0),
                1,
                cv2.LINE_AA,
            )
        for prediction in sample["predictions"]:
            x1, y1, x2, y2 = (int(round(value)) for value in prediction["box_xyxy"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220), 2)
            label = (
                f"P {names[int(prediction['class_id'])]} "
                f"{float(prediction['confidence']):.2f}"
            )
            cv2.putText(
                frame,
                label,
                (x1, min(frame.shape[0] - 5, y2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 220),
                1,
                cv2.LINE_AA,
            )
        file_name = f"worst_{rank}_{Path(str(sample['image'])).name}"
        cv2.imwrite(str(output / file_name), frame)
        summary["overlay"] = (output / file_name).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO detection/segmentation and localization distributions."
    )
    parser.add_argument("--task", choices=("detect", "segment"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlays", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("install the ML extras: python -m pip install -e '.[ml]'") from error

    data = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(str(data["path"]))
    image_key = "val" if args.split == "val" else "test"
    image_directory = root / str(data[image_key])
    label_directory = root / "labels" / ("valid" if args.split == "val" else "test")
    names = {index: str(name) for index, name in enumerate(data["names"])}
    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        workers=0,
        plots=True,
        project="runs/evaluation",
        name=f"{args.task}-{args.split}",
        exist_ok=True,
        verbose=False,
    )
    task_metrics: dict[str, Any] = {"boxes": _component_report(metrics.box, names)}
    if args.task == "segment":
        task_metrics["masks"] = _component_report(metrics.seg, names)

    samples: list[dict[str, Any]] = []
    if args.task == "detect":
        image_paths = sorted(
            path
            for path in image_directory.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise SystemExit(f"could not decode {image_path}")
            height, width = frame.shape[:2]
            result = model.predict(
                frame,
                imgsz=args.image_size,
                conf=args.confidence,
                device=args.device,
                verbose=False,
            )[0]
            predictions = [
                {
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                    "box_xyxy": [float(value) for value in box],
                }
                for box, class_id, confidence in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    strict=True,
                )
            ]
            samples.append(
                {
                    "image": image_path.relative_to(root).as_posix(),
                    "absolute_image": image_path,
                    "ground_truth": _ground_truth(
                        label_directory / f"{image_path.stem}.txt", width, height
                    ),
                    "predictions": predictions,
                }
            )
        localization = localization_report(samples)
        task_metrics["localization_at_confidence"] = args.confidence
        task_metrics["localization"] = localization
        if args.overlays:
            _draw_worst_cases(samples, localization["worst_cases"], names, args.overlays)

    payload = {
        "evaluation_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "split": args.split,
        "model": {
            "file": args.model.name,
            "bytes": args.model.stat().st_size,
            "sha256": _sha256(args.model),
        },
        "dataset": root.name,
        "image_size": args.image_size,
        "batch": args.batch,
        "device": args.device,
        "metrics": task_metrics,
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
        },
        "software": {"ultralytics": ultralytics.__version__},
    }
    for sample in samples:
        sample.pop("absolute_image", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
