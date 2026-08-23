import json
import random
from pathlib import Path

from PIL import Image

from pallet_audit.data.prepare import validate_prepared_yolo
from pallet_audit.data.scd import prepare_oscd_dataset


def _noise_image(path: Path, seed: int) -> None:
    generator = random.Random(seed)
    image = Image.new("RGB", (24, 16))
    image.putdata(
        [tuple(generator.randrange(256) for _ in range(3)) for _ in range(24 * 16)]
    )
    image.save(path)


def _partition(root: Path, stem: str, seeds: list[int], invalid_index: int | None = None) -> None:
    image_directory = root / "images" / f"{stem}2017"
    annotation_directory = root / "annotations"
    image_directory.mkdir(parents=True, exist_ok=True)
    annotation_directory.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    for index, seed in enumerate(seeds, start=1):
        name = f"{stem}-{index}.jpg"
        _noise_image(image_directory / name, seed)
        images.append({"id": index, "file_name": name, "width": 24, "height": 16})
        polygon = [2, 2, 14, 2, 14, 10, 2, 10]
        if invalid_index == index:
            polygon = [2, 2, 14, 2]
        annotations.append(
            {
                "id": index,
                "image_id": index,
                "category_id": 1,
                "bbox": [2, 2, 12, 8],
                "area": 96,
                "iscrowd": 0,
                "segmentation": [polygon],
            }
        )
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "Carton"}],
    }
    (annotation_directory / f"instances_{stem}2017.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_oscd_preserves_test_and_excludes_cross_split_visual_duplicate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _partition(source, "train", [1, 2, 3, 4, 99], invalid_index=4)
    _partition(source, "val", [99, 100])

    output = tmp_path / "prepared"
    report = prepare_oscd_dataset(source, output, validation_fraction=0.25, seed=7)

    assert report["authors_partition"]["train_images"] == 5
    assert report["authors_partition"]["test_images"] == 2
    assert report["cross_author_split_visual_hash_groups"] == 1
    assert report["excluded_author_train_images"] == 1
    assert report["rejected_annotations"] == {"degenerate-polygon": 1}
    assert report["images_by_split"]["test"] == 2
    assert sum(report["images_by_split"].values()) == 6
    assert report["groups"]["cross_split"] == 0
    validation = validate_prepared_yolo(output, "segment")
    assert validation["gates"]["cross_split_groups"] == 0
    assert validation["gates"]["all_coordinates_normalized"] is True
