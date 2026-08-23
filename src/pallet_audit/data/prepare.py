from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .audit import _dhash, find_coco_annotations


class PreparationFailure(ValueError):
    """Raised when audited COCO data cannot be converted without label loss."""


@dataclass(frozen=True)
class ImageRecord:
    source_split: str
    annotation_path: Path
    image_path: Path
    file_name: str
    width: int
    height: int
    annotations: tuple[dict[str, Any], ...]

    @property
    def provenance_key(self) -> str:
        return f"{self.source_split}/{self.file_name}"


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _filesystem_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def _resolve_image(annotation_path: Path, file_name: str) -> Path:
    candidates = (
        annotation_path.parent / file_name,
        annotation_path.parent.parent / file_name,
        annotation_path.parent / Path(file_name).name,
    )
    for candidate in candidates:
        if os.path.isfile(_filesystem_path(candidate)):
            return candidate
    raise PreparationFailure(f"Missing image: {file_name}")


def _origin_family(file_name: str) -> str:
    base = re.sub(r"\.rf\.[^.]+(?=\.[^.]+$)", "", Path(file_name).name)
    base = re.sub(r"_(?:jpg|jpeg|png)$", "", Path(base).stem, flags=re.IGNORECASE)
    date_match = re.search(r"IMG_(\d{8})", base, flags=re.IGNORECASE)
    if date_match:
        return f"capture-date:{date_match.group(1)}"
    generated_match = re.match(r"^(train_\d+)(?:_|$)", base, flags=re.IGNORECASE)
    if generated_match:
        return f"augmentation-family:{generated_match.group(1).lower()}"
    return f"source-image:{base.lower()}"


def _load_records(source_root: Path) -> tuple[list[ImageRecord], dict[int, str]]:
    records: list[ImageRecord] = []
    categories: dict[int, str] = {}
    for annotation_file in find_coco_annotations(source_root):
        payload = json.loads(annotation_file.path.read_text(encoding="utf-8"))
        for category in payload["categories"]:
            category_id = int(category["id"])
            name = str(category["name"]).strip()
            previous = categories.get(category_id)
            if previous is not None and previous != name:
                raise PreparationFailure(f"Inconsistent category {category_id}")
            categories[category_id] = name
        by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload["annotations"]:
            by_image[int(annotation["image_id"])].append(annotation)
        for image in payload["images"]:
            file_name = str(image["file_name"])
            records.append(
                ImageRecord(
                    source_split=annotation_file.split,
                    annotation_path=annotation_file.path,
                    image_path=_resolve_image(annotation_file.path, file_name),
                    file_name=file_name,
                    width=int(image["width"]),
                    height=int(image["height"]),
                    annotations=tuple(by_image[int(image["id"])]),
                )
            )
    if not records:
        raise PreparationFailure(f"No images found under {source_root}")
    return records, categories


def _build_groups(records: list[ImageRecord]) -> list[list[int]]:
    sets = _DisjointSet(len(records))
    by_family: dict[str, list[int]] = defaultdict(list)
    by_visual_hash: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_family[_origin_family(record.file_name)].append(index)
        by_visual_hash[_dhash(record.image_path)].append(index)
    for members in (*by_family.values(), *by_visual_hash.values()):
        anchor = members[0]
        for member in members[1:]:
            sets.union(anchor, member)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[sets.find(index)].append(index)
    return list(groups.values())


def _assign_groups(
    groups: list[list[int]], records: list[ImageRecord], ratios: dict[str, float], seed: int
) -> dict[int, str]:
    if set(ratios) != {"train", "valid", "test"}:
        raise PreparationFailure("Ratios must define train, valid, and test")
    if abs(sum(ratios.values()) - 1.0) > 1e-9 or any(value <= 0 for value in ratios.values()):
        raise PreparationFailure("Split ratios must be positive and sum to 1")

    total = sum(len(group) for group in groups)
    targets = {split: ratio * total for split, ratio in ratios.items()}
    counts = Counter[str]()
    assignment: dict[int, str] = {}

    def stable_group_key(group: list[int]) -> str:
        members = "\n".join(sorted(records[index].provenance_key for index in group))
        return hashlib.sha256(f"{seed}\n{members}".encode()).hexdigest()

    ordered = sorted(groups, key=lambda group: (-len(group), stable_group_key(group)))
    split_order = ("train", "valid", "test")
    for group in ordered:
        split = max(
            split_order,
            key=lambda name: (
                (targets[name] - counts[name]) / targets[name],
                -split_order.index(name),
            ),
        )
        for index in group:
            assignment[index] = split
        counts[split] += len(group)
    return assignment


def _normalise(value: float, scale: int, context: str) -> float:
    tolerance = 1.0 / scale
    normalised = value / scale
    if normalised < -tolerance or normalised > 1.0 + tolerance:
        raise PreparationFailure(f"Out-of-bounds coordinate at {context}: {value}/{scale}")
    return min(1.0, max(0.0, normalised))


def _labels_for_record(
    record: ImageRecord, task: str, class_ids: dict[int, int]
) -> tuple[list[str], Counter[int]]:
    lines: list[str] = []
    counts: Counter[int] = Counter()
    for annotation in record.annotations:
        raw_category = int(annotation["category_id"])
        if raw_category not in class_ids:
            continue
        class_id = class_ids[raw_category]
        context = f"{record.provenance_key}/annotation-{annotation.get('id')}"
        if task == "detect":
            x, y, width, height = (float(value) for value in annotation["bbox"])
            values = (
                _normalise(x + width / 2, record.width, context),
                _normalise(y + height / 2, record.height, context),
                width / record.width,
                height / record.height,
            )
            lines.append(f"{class_id} " + " ".join(f"{value:.8f}" for value in values))
        elif task == "segment":
            polygons = annotation.get("segmentation")
            if not isinstance(polygons, list) or len(polygons) != 1:
                raise PreparationFailure(f"Expected exactly one polygon at {context}")
            polygon = polygons[0]
            coordinates: list[float] = []
            for offset in range(0, len(polygon), 2):
                coordinates.extend(
                    (
                        _normalise(float(polygon[offset]), record.width, context),
                        _normalise(float(polygon[offset + 1]), record.height, context),
                    )
                )
            lines.append(
                f"{class_id} " + " ".join(f"{value:.8f}" for value in coordinates)
            )
        else:
            raise PreparationFailure(f"Unsupported task: {task}")
        counts[class_id] += 1
    return lines, counts


def prepare_yolo_dataset(
    source_root: Path,
    destination: Path,
    task: str,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    if task not in {"detect", "segment"}:
        raise PreparationFailure("Only detect and segment conversion is supported")
    if destination.exists():
        raise PreparationFailure(f"Destination already exists: {destination}")
    ratios = ratios or {"train": 0.8, "valid": 0.1, "test": 0.1}
    records, categories = _load_records(source_root)

    populated_categories = {
        int(annotation["category_id"])
        for record in records
        for annotation in record.annotations
    }
    selected_categories = sorted(populated_categories)
    class_ids = {raw_id: output_id for output_id, raw_id in enumerate(selected_categories)}
    class_names = [categories[raw_id] for raw_id in selected_categories]
    groups = _build_groups(records)
    assignment = _assign_groups(groups, records, ratios, seed)

    destination.mkdir(parents=True, exist_ok=False)
    image_counts: Counter[str] = Counter()
    label_counts: dict[str, Counter[int]] = defaultdict(Counter)
    empty_images: Counter[str] = Counter()
    source_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    manifest_records: list[dict[str, Any]] = []
    group_ids: dict[int, str] = {}
    try:
        for group in groups:
            members = "\n".join(sorted(records[index].provenance_key for index in group))
            group_id = hashlib.sha256(members.encode()).hexdigest()[:16]
            for index in group:
                group_ids[index] = group_id

        for index, record in enumerate(records):
            split = assignment[index]
            digest = hashlib.sha256(record.provenance_key.encode()).hexdigest()[:20]
            suffix = record.image_path.suffix.lower() or ".jpg"
            output_name = f"{source_root.name}_{digest}{suffix}"
            image_directory = destination / "images" / split
            label_directory = destination / "labels" / split
            image_directory.mkdir(parents=True, exist_ok=True)
            label_directory.mkdir(parents=True, exist_ok=True)
            output_image = image_directory / output_name
            try:
                os.link(_filesystem_path(record.image_path), _filesystem_path(output_image))
            except OSError:
                shutil.copy2(_filesystem_path(record.image_path), _filesystem_path(output_image))

            lines, counts = _labels_for_record(record, task, class_ids)
            label_path = label_directory / f"{Path(output_name).stem}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            image_counts[split] += 1
            label_counts[split].update(counts)
            if not lines:
                empty_images[split] += 1
            source_split_counts[split][record.source_split] += 1
            manifest_records.append(
                {
                    "output": f"images/{split}/{output_name}",
                    "source": record.image_path.relative_to(source_root).as_posix(),
                    "source_split": record.source_split,
                    "split": split,
                    "group_id": group_ids[index],
                    "origin_family": _origin_family(record.file_name),
                }
            )

        data_yaml = {
            "path": destination.resolve().as_posix(),
            "train": "images/train",
            "val": "images/valid",
            "test": "images/test",
            "names": class_names,
        }
        (destination / "data.yaml").write_text(
            yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
        )
        report: dict[str, Any] = {
            "preparation_schema_version": 1,
            "source": source_root.name,
            "task": task,
            "seed": seed,
            "strategy": {
                "ratios": ratios,
                "group_keys": [
                    "capture date when encoded in IMG_YYYYMMDD filenames",
                    "known augmentation family train_N",
                    "Roboflow pre-augmentation source filename",
                    "identical 64-bit difference hash",
                ],
                "assignment": "deterministic largest-group deficit balancing",
            },
            "classes": [
                {"id": output_id, "name": class_names[output_id], "source_id": raw_id}
                for raw_id, output_id in class_ids.items()
            ],
            "images_by_split": dict(sorted(image_counts.items())),
            "instances_by_split_and_class": {
                split: {
                    class_names[class_id]: count
                    for class_id, count in sorted(counts.items())
                }
                for split, counts in sorted(label_counts.items())
            },
            "empty_images_by_split": dict(sorted(empty_images.items())),
            "raw_split_distribution_within_processed_split": {
                split: dict(sorted(counts.items()))
                for split, counts in sorted(source_split_counts.items())
            },
            "groups": {
                "count": len(groups),
                "largest_images": max(len(group) for group in groups),
                "cross_split": 0,
            },
            "records": manifest_records,
        }
        (destination / "preparation_manifest.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return report
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def validate_prepared_yolo(root: Path, task: str) -> dict[str, Any]:
    """Validate label syntax, coordinate ranges, file pairs, and group isolation."""
    if task not in {"detect", "segment"}:
        raise PreparationFailure(f"Unsupported task: {task}")
    data_yaml_path = root / "data.yaml"
    manifest_path = root / "preparation_manifest.json"
    if not data_yaml_path.is_file() or not manifest_path.is_file():
        raise PreparationFailure(f"Prepared dataset is missing metadata: {root}")
    data_yaml = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    names = data_yaml.get("names") if isinstance(data_yaml, dict) else None
    if not isinstance(names, list) or not names:
        raise PreparationFailure("data.yaml names must be a non-empty list")

    image_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    instance_counts: Counter[str] = Counter()
    for split in ("train", "valid", "test"):
        images = {
            path.stem: path
            for path in (root / "images" / split).iterdir()
            if path.is_file()
        }
        labels = {
            path.stem: path
            for path in (root / "labels" / split).glob("*.txt")
            if path.is_file()
        }
        if images.keys() != labels.keys():
            missing_labels = sorted(images.keys() - labels.keys())[:5]
            missing_images = sorted(labels.keys() - images.keys())[:5]
            raise PreparationFailure(
                f"Unpaired files in {split}: missing labels={missing_labels}, "
                f"missing images={missing_images}"
            )
        image_counts[split] = len(images)
        label_counts[split] = len(labels)
        for label_path in labels.values():
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                tokens = line.split()
                minimum = 5 if task == "detect" else 7
                if len(tokens) < minimum or (task == "detect" and len(tokens) != 5):
                    raise PreparationFailure(
                        f"Invalid {task} label at {label_path}:{line_number}"
                    )
                if task == "segment" and (len(tokens) - 1) % 2:
                    raise PreparationFailure(
                        f"Unpaired polygon coordinate at {label_path}:{line_number}"
                    )
                class_id = int(tokens[0])
                if class_id < 0 or class_id >= len(names):
                    raise PreparationFailure(
                        f"Invalid class id at {label_path}:{line_number}: {class_id}"
                    )
                coordinates = [float(value) for value in tokens[1:]]
                if any(value < 0.0 or value > 1.0 for value in coordinates):
                    raise PreparationFailure(
                        f"Coordinate outside [0,1] at {label_path}:{line_number}"
                    )
                instance_counts[split] += 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in manifest["records"]:
        group_splits[str(record["group_id"])].add(str(record["split"]))
    leaking_groups = sorted(
        group_id for group_id, splits in group_splits.items() if len(splits) > 1
    )
    if leaking_groups:
        raise PreparationFailure(f"Groups cross split boundaries: {leaking_groups[:5]}")

    return {
        "validation_schema_version": 1,
        "dataset": root.name,
        "task": task,
        "images_by_split": dict(sorted(image_counts.items())),
        "labels_by_split": dict(sorted(label_counts.items())),
        "instances_by_split": dict(sorted(instance_counts.items())),
        "groups": len(group_splits),
        "gates": {
            "all_images_have_labels": True,
            "all_labels_have_images": True,
            "all_class_ids_valid": True,
            "all_coordinates_normalized": True,
            "cross_split_groups": 0,
        },
    }
