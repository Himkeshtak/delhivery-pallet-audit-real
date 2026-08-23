#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def _rank(path: Path, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{path.as_posix()}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render deterministic visual-QA overlays for a YOLO segmentation dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-split", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples_per_split < 1:
        raise SystemExit("samples-per-split must be positive")

    data = yaml.safe_load((args.dataset / "data.yaml").read_text(encoding="utf-8"))
    names = [str(name) for name in data["names"]]
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for split in ("train", "valid", "test"):
        images = sorted(
            (
                path
                for path in (args.dataset / "images" / split).iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ),
            key=lambda path: _rank(path, args.seed),
        )[: args.samples_per_split]
        for index, image_path in enumerate(images, start=1):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise SystemExit(f"Could not decode {image_path}")
            height, width = frame.shape[:2]
            overlay = frame.copy()
            label_path = args.dataset / "labels" / split / f"{image_path.stem}.txt"
            instances = 0
            for line in label_path.read_text(encoding="utf-8").splitlines():
                tokens = line.split()
                class_id = int(tokens[0])
                values = np.asarray([float(value) for value in tokens[1:]], dtype=np.float32)
                points = values.reshape(-1, 2)
                points[:, 0] *= width
                points[:, 1] *= height
                polygon = np.rint(points).astype(np.int32)
                cv2.fillPoly(overlay, [polygon], (0, 190, 255))
                cv2.polylines(frame, [polygon], True, (0, 110, 255), 2, cv2.LINE_AA)
                x, y = polygon.min(axis=0)
                cv2.putText(
                    frame,
                    names[class_id],
                    (int(x), max(14, int(y) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 80, 220),
                    1,
                    cv2.LINE_AA,
                )
                instances += 1
            rendered = cv2.addWeighted(overlay, 0.28, frame, 0.72, 0)
            output_path = args.output / f"{split}_{index}.jpg"
            cv2.imwrite(str(output_path), rendered)
            records.append(
                {
                    "split": split,
                    "source_image": image_path.relative_to(args.dataset).as_posix(),
                    "instances": instances,
                    "overlay": output_path.as_posix(),
                }
            )
    (args.output / "index.json").write_text(
        json.dumps({"seed": args.seed, "samples": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

