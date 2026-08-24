from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .audit import AuditFailure, _dhash, _sha256, _validate_bbox
from .prepare import PreparationFailure, _filesystem_path, _normalise


@dataclass(frozen=True)
class OSCDRecord:
    author_split: str
    image_id: int
    image_path: Path
    file_name: str
    width: int
    height: int
    annotations: tuple[dict[str, Any], ...]
    visual_hash: str

    @property
    def provenance_key(self) -> str:
        return f"{self.author_split}/{self.file_name}"


def _load_partition(root: Path, json_stem: str, author_split: str) -> list[OSCDRecord]:
    annotation_path = root / "annotations" / f"instances_{json_stem}2017.json"
    image_directory = root / "images" / f"{json_stem}2017"
    if not annotation_path.is_file() or not image_directory.is_dir():
        raise PreparationFailure(f"OSCD partition is incomplete: {json_stem}")
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {
        int(category["id"]): str(category["name"]).strip()
        for category in payload.get("categories", [])
    }
    if categories != {1: "Carton"}:
        raise PreparationFailure(f"Unexpected OSCD categories: {categories}")
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    records: list[OSCDRecord] = []
    seen_ids: set[int] = set()
    for image in payload.get("images", []):
        image_id = int(image["id"])
        if image_id in seen_ids:
            raise PreparationFailure(f"Duplicate OSCD image id in {json_stem}: {image_id}")
        seen_ids.add(image_id)
        file_name = str(image["file_name"])
        image_path = image_directory / file_name
        if not image_path.is_file():
            raise PreparationFailure(f"Missing OSCD image: {image_path}")
        records.append(
            OSCDRecord(
                author_split=author_split,
                image_id=image_id,
                image_path=image_path,
                file_name=file_name,
                width=int(image["width"]),
                height=int(image["height"]),
                annotations=tuple(annotations_by_image[image_id]),
                visual_hash=_dhash(image_path),
            )
        )
    return records


def _assign_train_validation(
    records: list[OSCDRecord], validation_fraction: float, seed: int
) -> dict[str, str]:
    if not 0.0 < validation_fraction < 0.5:
        raise PreparationFailure("validation_fraction must be between zero and 0.5")
    by_hash: dict[str, list[OSCDRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.visual_hash].append(record)
    target = round(len(records) * validation_fraction)

    def stable_key(item: tuple[str, list[OSCDRecord]]) -> str:
        fingerprint, members = item
        names = "\n".join(sorted(member.provenance_key for member in members))
        return hashlib.sha256(f"{seed}\n{fingerprint}\n{names}".encode()).hexdigest()

    assignments: dict[str, str] = {}
    validation_count = 0
    for fingerprint, members in sorted(by_hash.items(), key=stable_key):
        remaining = target - validation_count
        split = "valid" if remaining > 0 and len(members) <= 2 * remaining else "train"
        assignments[fingerprint] = split
        if split == "valid":
            validation_count += len(members)
    return assignments


def _polygon_line(
    record: OSCDRecord, annotation: dict[str, Any]
) -> tuple[str | None, str | None]:
    context = f"{record.provenance_key}/annotation-{annotation.get('id')}"
    if int(annotation.get("category_id", -1)) != 1:
        raise PreparationFailure(f"Unexpected category at {context}")
    try:
        _validate_bbox(annotation.get("bbox"), record.width, record.height, context)
    except (AuditFailure, TypeError, ValueError):
        return None, "invalid-bbox"
    polygons = annotation.get("segmentation")
    if not isinstance(polygons, list) or len(polygons) != 1:
        return None, "not-exactly-one-polygon"
    polygon = polygons[0]
    if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
        return None, "degenerate-polygon"
    coordinates: list[float] = []
    try:
        for offset in range(0, len(polygon), 2):
            coordinates.extend(
                (
                    _normalise(float(polygon[offset]), record.width, context),
                    _normalise(float(polygon[offset + 1]), record.height, context),
                )
            )
    except (PreparationFailure, TypeError, ValueError):
        return None, "out-of-bounds-polygon"
    return "0 " + " ".join(f"{value:.8f}" for value in coordinates), None


def _segment_box_key(line: str) -> tuple[float, float, float, float]:
    """Return the normalized segment envelope used by Ultralytics to de-duplicate labels."""
    values = [float(value) for value in line.split()[1:]]
    xs = values[0::2]
    ys = values[1::2]
    return (
        round(min(xs), 8),
        round(min(ys), 8),
        round(max(xs), 8),
        round(max(ys), 8),
    )


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(_filesystem_path(source), _filesystem_path(destination))
    except OSError:
        shutil.copy2(_filesystem_path(source), _filesystem_path(destination))


def prepare_oscd_dataset(
    source_root: Path,
    destination: Path,
    *,
    validation_fraction: float = 0.12,
    seed: int = 42,
    archive: Path | None = None,
    expected_archive_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit canonical OSCD and prepare leakage-controlled YOLO segmentation data.

    The authors' ``val2017`` partition is treated as the immutable test set.
    Any author-training record with the same 64-bit difference hash as a test
    record is excluded instead of contaminating that test set.
    """
    source_root = source_root.resolve()
    destination = destination.resolve()
    if destination.exists() or destination.with_name(destination.name + ".building").exists():
        raise PreparationFailure(f"Destination already exists: {destination}")
    archive_report: dict[str, Any] | None = None
    if archive is not None:
        archive = archive.resolve()
        if not archive.is_file():
            raise PreparationFailure(f"Archive does not exist: {archive}")
        digest = _sha256(archive)
        if expected_archive_sha256 and digest.lower() != expected_archive_sha256.lower():
            raise PreparationFailure("OSCD archive SHA-256 does not match the pinned source")
        archive_report = {
            "path": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": digest,
        }

    author_train = _load_partition(source_root, "train", "author-train")
    author_test = _load_partition(source_root, "val", "author-test")
    test_hashes = {record.visual_hash for record in author_test}
    excluded = [record for record in author_train if record.visual_hash in test_hashes]
    eligible_train = [record for record in author_train if record.visual_hash not in test_hashes]
    train_assignments = _assign_train_validation(eligible_train, validation_fraction, seed)

    work = destination.with_name(destination.name + ".building")
    image_counts: Counter[str] = Counter()
    instance_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    manifest_records: list[dict[str, Any]] = []
    work.mkdir(parents=True, exist_ok=False)
    for record in (*eligible_train, *author_test):
        split = (
            "test"
            if record.author_split == "author-test"
            else train_assignments[record.visual_hash]
        )
        digest = hashlib.sha256(record.provenance_key.encode()).hexdigest()[:20]
        suffix = record.image_path.suffix.lower() or ".jpg"
        output_name = f"scd_oscd_{digest}{suffix}"
        image_directory = work / "images" / split
        label_directory = work / "labels" / split
        image_directory.mkdir(parents=True, exist_ok=True)
        label_directory.mkdir(parents=True, exist_ok=True)
        _link_or_copy(record.image_path, image_directory / output_name)
        lines: list[str] = []
        seen_segment_boxes: set[tuple[float, float, float, float]] = set()
        for annotation in record.annotations:
            line, rejection = _polygon_line(record, annotation)
            if rejection:
                rejection_counts[rejection] += 1
            elif line:
                box_key = _segment_box_key(line)
                if box_key in seen_segment_boxes:
                    rejection_counts["duplicate-segment-box"] += 1
                else:
                    seen_segment_boxes.add(box_key)
                    lines.append(line)
        (label_directory / f"{Path(output_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        image_counts[split] += 1
        instance_counts[split] += len(lines)
        manifest_records.append(
            {
                "output": f"images/{split}/{output_name}",
                "source": record.image_path.relative_to(source_root).as_posix(),
                "source_split": record.author_split,
                "split": split,
                "group_id": record.visual_hash,
            }
        )

    data_yaml = {
        "path": destination.as_posix(),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": ["carton"],
    }
    (work / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )
    cross_author_hashes = sorted(
        {record.visual_hash for record in author_train} & test_hashes
    )
    report: dict[str, Any] = {
        "preparation_schema_version": 1,
        "source": "scd_oscd_official",
        "task": "segment",
        "seed": seed,
        "archive": archive_report,
        "license": "CC BY-NC-SA 4.0",
        "authors_partition": {
            "train_images": len(author_train),
            "test_images": len(author_test),
            "train_instances_raw": sum(len(record.annotations) for record in author_train),
            "test_instances_raw": sum(len(record.annotations) for record in author_test),
        },
        "strategy": {
            "authors_val2017_role": "immutable-test",
            "validation_fraction_of_eligible_author_train": validation_fraction,
            "group_key": "identical 64-bit difference hash",
            "cross-author-test-policy": "exclude matching author-training records",
            "duplicate-annotation-policy": (
                "retain the first polygon for each class/envelope pair, matching "
                "the Ultralytics segmentation loader"
            ),
        },
        "cross_author_split_visual_hash_groups": len(cross_author_hashes),
        "excluded_author_train_images": len(excluded),
        "rejected_annotations": dict(sorted(rejection_counts.items())),
        "images_by_split": dict(sorted(image_counts.items())),
        "instances_by_split": dict(sorted(instance_counts.items())),
        "groups": {
            "count": len({record["group_id"] for record in manifest_records}),
            "cross_split": 0,
        },
        "records": manifest_records,
    }
    (work / "preparation_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    work.rename(destination)
    return report
