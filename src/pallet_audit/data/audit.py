from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


class AuditFailure(ValueError):
    """Raised when an export cannot safely enter model preparation."""


@dataclass(frozen=True)
class AnnotationFile:
    split: str
    path: Path


def _filesystem_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(_filesystem_path(path), "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _dhash(path: Path, size: int = 8) -> str:
    with Image.open(_filesystem_path(path)) as image:
        gray = image.convert("L").resize((size + 1, size))
        pixels = list(gray.getdata())
    bits = []
    for row in range(size):
        offset = row * (size + 1)
        bits.extend(pixels[offset + col] > pixels[offset + col + 1] for col in range(size))
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:0{size * size // 4}x}"


def find_coco_annotations(root: Path) -> list[AnnotationFile]:
    candidates = sorted(root.rglob("*.json"))
    found: list[AnnotationFile] = []
    for path in candidates:
        if path.name == "manifest.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if all(key in payload for key in ("images", "annotations", "categories")):
            relative_parts = [part.lower() for part in path.relative_to(root).parts]
            split = next(
                (
                    name
                    for name in ("train", "valid", "validation", "test")
                    if name in relative_parts
                ),
                "unspecified",
            )
            if split == "validation":
                split = "valid"
            found.append(AnnotationFile(split, path))
    if not found:
        raise AuditFailure(f"No COCO annotation JSON found under {root}")
    return found


def _resolve_image(annotation_path: Path, file_name: str) -> Path:
    candidates = [
        annotation_path.parent / file_name,
        annotation_path.parent.parent / file_name,
        annotation_path.parent / Path(file_name).name,
    ]
    for candidate in candidates:
        if os.path.isfile(_filesystem_path(candidate)):
            return candidate
    raise AuditFailure(f"Missing image referenced by {annotation_path}: {file_name}")


def _validate_bbox(bbox: Any, width: int, height: int, context: str) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise AuditFailure(f"Invalid COCO bbox at {context}")
    x, y, box_width, box_height = (float(value) for value in bbox)
    if box_width <= 0 or box_height <= 0:
        raise AuditFailure(f"Non-positive bbox at {context}")
    tolerance = 1.0
    outside_image = (
        x < -tolerance
        or y < -tolerance
        or x + box_width > width + tolerance
        or y + box_height > height + tolerance
    )
    if outside_image:
        raise AuditFailure(f"Out-of-bounds bbox at {context}")


def _valid_polygon(segmentation: Any) -> bool:
    if not isinstance(segmentation, list) or not segmentation:
        return False
    return all(
        isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0
        for polygon in segmentation
    )


def audit_coco_export(root: Path) -> dict[str, Any]:
    annotation_files = find_coco_annotations(root)
    category_names: dict[int, str] = {}
    class_counts: Counter[str] = Counter()
    split_images: Counter[str] = Counter()
    split_annotations: Counter[str] = Counter()
    task_signals: Counter[str] = Counter()
    exact_hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    visual_hashes: dict[str, list[dict[str, str]]] = defaultdict(list)

    for annotation_file in annotation_files:
        payload = json.loads(annotation_file.path.read_text(encoding="utf-8"))
        for category in payload["categories"]:
            category_id = int(category["id"])
            name = str(category["name"]).strip()
            existing = category_names.get(category_id)
            if existing is not None and existing != name:
                raise AuditFailure(
                    f"Category id {category_id} maps to both {existing!r} and {name!r}"
                )
            category_names[category_id] = name

        images_by_id: dict[int, dict[str, Any]] = {}
        for image in payload["images"]:
            image_id = int(image["id"])
            if image_id in images_by_id:
                raise AuditFailure(f"Duplicate image id {image_id} in {annotation_file.path}")
            images_by_id[image_id] = image
            image_path = _resolve_image(annotation_file.path, str(image["file_name"]))
            record = {
                "split": annotation_file.split,
                "path": image_path.relative_to(root).as_posix(),
            }
            exact_hashes[_sha256(image_path)].append(record)
            visual_hashes[_dhash(image_path)].append(record)
            split_images[annotation_file.split] += 1

        for annotation in payload["annotations"]:
            image_id = int(annotation["image_id"])
            category_id = int(annotation["category_id"])
            if image_id not in images_by_id:
                raise AuditFailure(f"Annotation references missing image id {image_id}")
            if category_id not in category_names:
                raise AuditFailure(f"Annotation references missing category id {category_id}")
            image = images_by_id[image_id]
            _validate_bbox(
                annotation.get("bbox"),
                int(image["width"]),
                int(image["height"]),
                f"annotation {annotation.get('id')} in {annotation_file.path}",
            )
            class_counts[category_names[category_id]] += 1
            split_annotations[annotation_file.split] += 1
            task_signals["bbox"] += 1
            if _valid_polygon(annotation.get("segmentation")):
                task_signals["polygon"] += 1
            keypoints = annotation.get("keypoints")
            if isinstance(keypoints, list) and keypoints:
                task_signals["keypoints"] += 1

    def duplicate_groups(mapping: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
        return [
            {"fingerprint": fingerprint, "files": records}
            for fingerprint, records in mapping.items()
            if len(records) > 1
        ]

    exact_groups = duplicate_groups(exact_hashes)
    visual_groups = duplicate_groups(visual_hashes)
    cross_split_exact = [
        group for group in exact_groups if len({record["split"] for record in group["files"]}) > 1
    ]
    cross_split_visual = [
        group
        for group in visual_groups
        if len({record["split"] for record in group["files"]}) > 1
    ]
    inferred_task = "object-detection"
    if task_signals["polygon"]:
        inferred_task = "instance-segmentation"
    if task_signals["keypoints"]:
        inferred_task = "keypoint-detection"

    return {
        "audit_schema_version": 1,
        "root": root.name,
        "annotation_files": [
            {"split": item.split, "path": item.path.relative_to(root).as_posix()}
            for item in annotation_files
        ],
        "inferred_task": inferred_task,
        "categories": [
            {"id": category_id, "name": name}
            for category_id, name in sorted(category_names.items())
        ],
        "images_by_split": dict(sorted(split_images.items())),
        "annotations_by_split": dict(sorted(split_annotations.items())),
        "instances_by_class": dict(sorted(class_counts.items())),
        "annotation_signals": dict(sorted(task_signals.items())),
        "duplicates": {
            "exact_groups": exact_groups,
            "visual_hash_groups": visual_groups,
            "cross_split_exact_groups": cross_split_exact,
            "cross_split_visual_hash_groups": cross_split_visual,
        },
        "gates": {
            "all_references_resolve": True,
            "all_boxes_valid": True,
            "cross_split_exact_duplicates": len(cross_split_exact),
            "cross_split_visual_hash_duplicates": len(cross_split_visual),
        },
    }


def audit_many(roots: Iterable[Path]) -> dict[str, Any]:
    return {
        "audit_schema_version": 1,
        "sources": {root.name: audit_coco_export(root) for root in roots},
    }
