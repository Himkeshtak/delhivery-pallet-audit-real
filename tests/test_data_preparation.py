import json
import random
from pathlib import Path

from PIL import Image

from pallet_audit.data.prepare import (
    _origin_family,
    prepare_yolo_dataset,
    validate_prepared_yolo,
)


def _write_coco_image(root: Path, split: str, name: str, color: tuple[int, int, int]) -> None:
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / name
    generator = random.Random(sum((index + 1) * value for index, value in enumerate(color)))
    image = Image.new("RGB", (20, 10))
    image.putdata(
        [tuple(generator.randrange(256) for _ in range(3)) for _ in range(20 * 10)]
    )
    image.save(image_path)
    annotation_path = directory / "_annotations.coco.json"
    if annotation_path.exists():
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    else:
        payload = {"images": [], "annotations": [], "categories": [{"id": 2, "name": "pallet"}]}
    image_id = len(payload["images"]) + 1
    payload["images"].append(
        {"id": image_id, "file_name": name, "width": 20, "height": 10}
    )
    payload["annotations"].append(
        {
            "id": image_id,
            "image_id": image_id,
            "category_id": 2,
            "bbox": [2, 1, 10, 5],
            "area": 50,
            "iscrowd": 0,
        }
    )
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")


def test_origin_family_groups_capture_dates_and_augmentations() -> None:
    assert _origin_family("IMG_20250729_083703_jpg.rf.abc.jpg") == "capture-date:20250729"
    assert _origin_family("train_4_640_aug_05_flip_jpg.rf.abc.jpg") == (
        "augmentation-family:train_4"
    )


def test_preparation_keeps_duplicate_family_in_one_split(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_coco_image(source, "train", "IMG_20250101_010101_jpg.rf.a.jpg", (1, 2, 3))
    _write_coco_image(source, "valid", "IMG_20250101_010102_jpg.rf.b.jpg", (4, 5, 6))
    _write_coco_image(source, "test", "IMG_20250202_010103_jpg.rf.c.jpg", (7, 8, 9))
    _write_coco_image(source, "test", "IMG_20250303_010104_jpg.rf.d.jpg", (10, 11, 12))

    report = prepare_yolo_dataset(source, tmp_path / "prepared", "detect")

    matching = [
        record for record in report["records"] if record["origin_family"] == "capture-date:20250101"
    ]
    assert len({record["split"] for record in matching}) == 1
    assert report["groups"]["cross_split"] == 0
    assert (tmp_path / "prepared" / "data.yaml").is_file()
    validation = validate_prepared_yolo(tmp_path / "prepared", "detect")
    assert validation["gates"]["all_coordinates_normalized"] is True
    assert validation["gates"]["cross_split_groups"] == 0
