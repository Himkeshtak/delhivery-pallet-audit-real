#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from pallet_audit.evaluation import distribution


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics with a planar chessboard."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--columns", type=int, required=True, help="inner-corner columns")
    parser.add_argument("--rows", type=int, required=True, help="inner-corner rows")
    parser.add_argument("--square-size-m", type=float, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/calibration/camera.json"))
    args = parser.parse_args()

    image_paths = sorted(
        path
        for path in args.images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    object_template = np.zeros((args.rows * args.columns, 3), np.float32)
    object_template[:, :2] = np.mgrid[0 : args.columns, 0 : args.rows].T.reshape(-1, 2)
    object_template *= args.square_size_m
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    accepted_paths: list[Path] = []
    image_size: tuple[int, int] | None = None
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is not None and image_size != gray.shape[::-1]:
            raise SystemExit("all calibration images must have the same resolution")
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCornersSB(gray, (args.columns, args.rows))
        if found:
            object_points.append(object_template.copy())
            image_points.append(corners)
            accepted_paths.append(path)
    if len(object_points) < 8 or image_size is None:
        raise SystemExit("at least 8 successful, varied chessboard views are required")

    rms, camera_matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    view_errors: list[float] = []
    for object_set, image_set, rotation, translation in zip(
        object_points, image_points, rotations, translations, strict=True
    ):
        projected, _ = cv2.projectPoints(
            object_set, rotation, translation, camera_matrix, distortion
        )
        residual = np.linalg.norm(image_set.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
        view_errors.append(float(np.sqrt(np.mean(residual**2))))
    payload = {
        "calibration_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Zhang planar chessboard via OpenCV calibrateCamera",
        "image_size_px": list(image_size),
        "board_inner_corners": [args.columns, args.rows],
        "square_size_m": args.square_size_m,
        "accepted_images": [str(path) for path in accepted_paths],
        "rejected_image_count": len(image_paths) - len(accepted_paths),
        "opencv_rms_px": float(rms),
        "per_view_reprojection_rmse_px": distribution(view_errors),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.ravel().tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
