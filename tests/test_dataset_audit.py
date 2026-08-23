import json
from pathlib import Path

from PIL import Image

from pallet_audit.data.audit import audit_coco_export


def _write_split(root: Path, split: str, image_bytes_from: Path | None = None) -> Path:
    directory = root / split
    directory.mkdir(parents=True)
    image_path = directory / f"{split}.png"
    if image_bytes_from:
        image_path.write_bytes(image_bytes_from.read_bytes())
    else:
        Image.new("RGB", (32, 24), color=(30, 60, 90)).save(image_path)
    coco = {
        "images": [{"id": 1, "file_name": image_path.name, "width": 32, "height": 24}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2, 3, 10, 8],
                "area": 80,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "pallet"}],
    }
    (directory / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    return image_path


def test_audit_detects_cross_split_exact_duplicates(tmp_path: Path) -> None:
    train_image = _write_split(tmp_path, "train")
    _write_split(tmp_path, "test", image_bytes_from=train_image)

    report = audit_coco_export(tmp_path)

    assert report["inferred_task"] == "object-detection"
    assert report["images_by_split"] == {"test": 1, "train": 1}
    assert report["instances_by_class"] == {"pallet": 2}
    assert report["gates"]["cross_split_exact_duplicates"] == 1
    assert report["gates"]["cross_split_visual_hash_duplicates"] == 1


def test_audit_recognizes_polygon_signal(tmp_path: Path) -> None:
    _write_split(tmp_path, "train")
    path = tmp_path / "train" / "_annotations.coco.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["annotations"][0]["segmentation"] = [[2, 3, 12, 3, 12, 11, 2, 11]]
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_coco_export(tmp_path)

    assert report["inferred_task"] == "instance-segmentation"
    assert report["annotation_signals"]["polygon"] == 1
