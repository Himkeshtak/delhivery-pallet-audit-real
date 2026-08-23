from pathlib import Path

import pytest

from tools.train_yolo import validate_training_plan


def test_pose_training_refuses_box_only_yaml(tmp_path: Path) -> None:
    data = tmp_path / "data.yaml"
    data.write_text("train: images/train\nval: images/val\nnames: [pallet]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="boxes are not pose labels"):
        validate_training_plan("pose", data, "yolo11n-pose.pt")


def test_valid_segmentation_plan(tmp_path: Path) -> None:
    data = tmp_path / "data.yaml"
    data.write_text("train: images/train\nval: images/val\nnames: [pallet]\n", encoding="utf-8")
    assert validate_training_plan("segment", data, "yolo11n-seg.pt")["names"] == ["pallet"]

