import json
import zipfile
from pathlib import Path

import pytest

from pallet_audit.data.config import Source
from pallet_audit.data.roboflow import LocalArchiveError, import_source_archive


def _source() -> Source:
    return Source(
        id="fixture",
        workspace="workspace",
        project="project",
        version=1,
        url="https://example.test/dataset/1",
        expected_task="object-detection",
        allowed_licenses=("CC BY 4.0",),
        local_archive="fixture.zip",
    )


def test_import_local_archive_writes_hash_manifest(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("train/_annotations.coco.json", "{}")
        bundle.writestr("train/image.jpg", b"image-bytes")

    destination = import_source_archive(_source(), archive, tmp_path / "raw")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["source"]["id"] == "fixture"
    assert len(manifest["archive"]["sha256"]) == 64
    assert {item["path"] for item in manifest["files"]} == {
        "train/_annotations.coco.json",
        "train/image.jpg",
    }


def test_import_rejects_zip_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "unsafe")

    with pytest.raises(LocalArchiveError, match="Unsafe ZIP member"):
        import_source_archive(_source(), archive, tmp_path / "raw")
