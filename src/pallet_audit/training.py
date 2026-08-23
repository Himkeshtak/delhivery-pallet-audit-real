from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SUPPORTED_YOLO_TASKS = {"detect", "segment", "pose"}


def validate_yolo_training_plan(task: str, data_path: Path, model: str) -> dict[str, Any]:
    if task not in SUPPORTED_YOLO_TASKS:
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


def validate_anomaly_folder(root: Path) -> dict[str, Path]:
    paths = {
        "normal_train": root / "train" / "good",
        "normal_test": root / "test" / "good",
        "abnormal_test": root / "test" / "bad",
    }
    missing = [name for name, path in paths.items() if not path.is_dir()]
    if missing:
        raise ValueError(f"anomaly dataset is missing required directories: {missing}")
    if not any(paths["normal_train"].iterdir()):
        raise ValueError("normal training directory is empty")
    if not any(paths["abnormal_test"].iterdir()):
        raise ValueError("abnormal test directory is empty; AUROC/F1 cannot be measured")
    return paths
