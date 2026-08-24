#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pallet_audit.inference import unavailable_perception_assessment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_id(path: Path) -> str:
    return f"{path.stem}@{_sha256(path)[:16]}"


def _prediction_records(result: Any) -> list[dict[str, Any]]:
    if result.boxes is None:
        return []
    records = []
    polygons = result.masks.xy if result.masks is not None else []
    for index, (box, class_id, confidence) in enumerate(zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        strict=True,
    )):
        class_index = int(class_id)
        record = {
            "class_id": class_index,
            "class_name": str(result.names[class_index]),
            "confidence": float(confidence),
            "box_xyxy": [float(value) for value in box],
        }
        if index < len(polygons):
            record["mask_polygon_xy"] = [
                [float(point[0]), float(point[1])] for point in polygons[index]
            ]
        records.append(record)
    return records


def _run(model: Any, frame: np.ndarray, args: argparse.Namespace) -> tuple[Any, float]:
    started = time.perf_counter()
    result = model.predict(
        frame,
        imgsz=args.image_size,
        conf=args.confidence,
        device=args.device,
        verbose=False,
    )[0]
    return result, (time.perf_counter() - started) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run real checkpoints and emit fail-safe per-pallet assessment JSON."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--detector", type=Path, default=Path("weights/releases/detect-real-v1.pt")
    )
    parser.add_argument(
        "--structural-segmenter",
        type=Path,
        default=Path("weights/releases/segment-real-v1.pt"),
    )
    parser.add_argument(
        "--carton-segmenter",
        type=Path,
        default=Path("weights/releases/carton-seg-scd-v1.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    for path in (args.image, args.detector, args.structural_segmenter, args.carton_segmenter):
        if not path.is_file():
            raise SystemExit(f"required file does not exist: {path}")
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"could not decode image: {args.image}")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("install ML dependencies with python -m pip install -e '.[ml]'") from error

    total_started = time.perf_counter()
    detector_result, detector_ms = _run(YOLO(str(args.detector)), frame, args)
    structure_result, structure_ms = _run(YOLO(str(args.structural_segmenter)), frame, args)
    carton_result, carton_ms = _run(YOLO(str(args.carton_segmenter)), frame, args)
    detector_predictions = _prediction_records(detector_result)
    structure_predictions = _prediction_records(structure_result)
    carton_predictions = _prediction_records(carton_result)
    pallets = [
        prediction
        for prediction in detector_predictions
        if prediction["class_name"].strip().lower() == "pallet"
    ]

    overlay = frame.copy()
    for prediction in pallets:
        x1, y1, x2, y2 = (int(round(value)) for value in prediction["box_xyxy"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 200, 40), 3)
        cv2.putText(
            overlay,
            f"pallet {prediction['confidence']:.2f}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 200, 40),
            2,
            cv2.LINE_AA,
        )
    if carton_result.masks is not None:
        mask_layer = overlay.copy()
        for polygon in carton_result.masks.xy:
            points = np.rint(polygon).astype(np.int32)
            if len(points) >= 3:
                cv2.fillPoly(mask_layer, [points], (0, 150, 255))
                cv2.polylines(overlay, [points], True, (0, 100, 255), 2, cv2.LINE_AA)
        overlay = cv2.addWeighted(mask_layer, 0.25, overlay, 0.75, 0)

    args.output.mkdir(parents=True, exist_ok=True)
    overlay_path = args.output / "overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    model_versions = {
        "pallet_detector": _model_id(args.detector),
        "structural_segmenter": _model_id(args.structural_segmenter),
        "carton_segmenter": _model_id(args.carton_segmenter),
        "pose": "UNAVAILABLE:no-trained-keypoint-checkpoint",
        "damage": "UNAVAILABLE:no-reviewed-damage-labels",
        "wrap": "UNAVAILABLE:no-reviewed-wrap-labels",
    }
    assessment_files = []
    for index, _ in enumerate(pallets, start=1):
        assessment = unavailable_perception_assessment(
            frame_id=args.image.name,
            track_id=f"single-frame-pallet-{index:03d}",
            model_versions=model_versions,
            reasons=[
                "metric_pose_unavailable_no_keypoint_checkpoint",
                "assignment_camera_calibration_not_supplied",
                "single_view_load_evidence_incomplete",
            ],
            calibration_id=None,
        )
        assessment_path = args.output / f"assessment_{index:03d}.json"
        assessment_path.write_text(
            json.dumps(assessment.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        assessment_files.append(assessment_path.name)
    total_ms = (time.perf_counter() - total_started) * 1000.0
    summary = {
        "inference_schema_version": 1,
        "image": str(args.image),
        "status": "ASSESSMENTS_WRITTEN" if pallets else "NO_PALLET_DETECTED",
        "assessment_files": assessment_files,
        "counts": {
            "pallets": len(pallets),
            "detector_instances": len(detector_predictions),
            "structural_instances": len(structure_predictions),
            "carton_instances": len(carton_predictions),
        },
        "predictions": {
            "pallet_detector": detector_predictions,
            "structural_segmenter": structure_predictions,
            "carton_segmenter": carton_predictions,
        },
        "latency_ms_single_frame": {
            "pallet_detector": detector_ms,
            "structural_segmenter": structure_ms,
            "carton_segmenter": carton_ms,
            "total_including_model_load_and_serialization": total_ms,
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "software": {"ultralytics": ultralytics.__version__},
        "model_versions": model_versions,
        "overlay": overlay_path.name,
        "limitations": [
            "carton masks are component pretraining outputs, not assignment-domain validation",
            "cartons are not associated to pallets without calibrated geometry/tracking",
            "no metric or appearance SOP check is auto-passed from this evidence",
        ],
    }
    (args.output / "frame_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
