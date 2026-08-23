#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from pallet_audit.evaluation import distribution


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit image-to-floor homography from surveyed, marker-free floor correspondences."
        )
    )
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--floor-frame", default="slot-floor")
    parser.add_argument("--ransac-floor-threshold-m", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("artifacts/calibration/floor.json"))
    args = parser.parse_args()
    with args.points.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_x_px", "image_y_px", "floor_x_m", "floor_y_m"}
    if len(rows) < 6 or not rows or not required.issubset(rows[0]):
        raise SystemExit("provide at least 6 correspondences with the documented CSV columns")
    image_points = np.array(
        [[float(row["image_x_px"]), float(row["image_y_px"])] for row in rows], dtype=float
    )
    floor_points = np.array(
        [[float(row["floor_x_m"]), float(row["floor_y_m"])] for row in rows], dtype=float
    )
    image_to_floor, inliers = cv2.findHomography(
        image_points, floor_points, cv2.RANSAC, args.ransac_floor_threshold_m
    )
    if image_to_floor is None or inliers is None:
        raise SystemExit("homography fit failed; check point geometry and units")
    floor_to_image = np.linalg.inv(image_to_floor)
    projected_floor = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2), image_to_floor
    ).reshape(-1, 2)
    projected_image = cv2.perspectiveTransform(
        floor_points.reshape(-1, 1, 2), floor_to_image
    ).reshape(-1, 2)
    floor_errors = np.linalg.norm(projected_floor - floor_points, axis=1)
    image_errors = np.linalg.norm(projected_image - image_points, axis=1)
    inlier_mask = inliers.ravel().astype(bool)
    payload = {
        "calibration_schema_version": 1,
        "calibration_id": args.calibration_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "floor_frame": args.floor_frame,
        "correspondence_count": len(rows),
        "inlier_count": int(inlier_mask.sum()),
        "image_to_floor": image_to_floor.tolist(),
        "floor_to_image": floor_to_image.tolist(),
        "floor_residual_m": distribution(floor_errors[inlier_mask]),
        "image_reprojection_error_px": distribution(image_errors[inlier_mask]),
        "correspondences": [
            {
                **row,
                "inlier": bool(inlier_mask[index]),
                "floor_residual_m": float(floor_errors[index]),
                "image_reprojection_error_px": float(image_errors[index]),
            }
            for index, row in enumerate(rows)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
