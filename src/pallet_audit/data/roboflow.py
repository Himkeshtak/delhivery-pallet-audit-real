from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Source, load_sources


class RoboflowDownloadError(RuntimeError):
    """Raised when an authenticated, pinned Roboflow export cannot be obtained."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _request_json(url: str, timeout: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "delhivery-pallet-audit/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(detail).get("error", {}).get("message", detail)
        except json.JSONDecodeError:
            message = detail
        if error.code in {401, 403}:
            raise RoboflowDownloadError(
                "Roboflow rejected the export credential. Set a valid private "
                "ROBOFLOW_API_KEY; the value is never written to disk."
            ) from error
        raise RoboflowDownloadError(f"Roboflow API HTTP {error.code}: {message}") from error
    except urllib.error.URLError as error:
        raise RoboflowDownloadError(f"Roboflow API unavailable: {error.reason}") from error


def _export_link(source: Source, api_key: str, model_format: str, polls: int = 120) -> str:
    quoted_key = urllib.parse.quote(api_key, safe="")
    url = (
        f"https://api.roboflow.com/{source.workspace}/{source.project}/"
        f"{source.version}/{model_format}?api_key={quoted_key}"
    )
    for _ in range(polls):
        payload = _request_json(url)
        export = payload.get("export")
        if isinstance(export, dict) and export.get("link"):
            return str(export["link"])
        if payload.get("ready") is False:
            time.sleep(1)
            continue
        raise RoboflowDownloadError(
            f"Unexpected export response for {source.id}; no download link was returned"
        )
    raise RoboflowDownloadError(f"Timed out while Roboflow generated export for {source.id}")


def _download(url: str, destination: Path, timeout: int = 180) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "delhivery-pallet-audit/0.1"})
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            partial.open("wb") as out,
        ):
            shutil.copyfileobj(response, out, length=1024 * 1024)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = (destination / member.filename).resolve()
            within_destination = (
                destination_resolved in member_path.parents
                or member_path == destination_resolved
            )
            if not within_destination:
                raise RoboflowDownloadError(f"Unsafe ZIP member rejected: {member.filename}")
        bundle.extractall(destination)


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "manifest.json":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def download_source(source: Source, output_root: Path, api_key: str, overwrite: bool) -> Path:
    destination = output_root / source.id
    archive = destination / "source.coco.zip"
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and not overwrite:
        return destination

    if destination.exists() and overwrite:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    link = _export_link(source, api_key, "coco")
    _download(link, archive)
    _safe_extract(archive, destination)
    archive_hash = sha256_file(archive)

    manifest = {
        "manifest_schema_version": 1,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": asdict(source),
        "format": "coco",
        "archive": {
            "path": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": archive_hash,
        },
        "files": _file_manifest(destination),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination


def download_sources(
    config_path: Path,
    output_root: Path,
    api_key: str | None = None,
    selected: Iterable[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    config = load_sources(config_path)
    credential = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not credential:
        raise RoboflowDownloadError(
            "ROBOFLOW_API_KEY is missing. Create a private key in Roboflow settings, "
            "set it in the environment, and rerun; do not commit it."
        )

    selected_set = set(selected or [])
    unknown = selected_set - {source.id for source in config.sources}
    if unknown:
        raise RoboflowDownloadError(f"Unknown source ids: {sorted(unknown)}")
    sources = [source for source in config.sources if not selected_set or source.id in selected_set]
    return [download_source(source, output_root, credential, overwrite) for source in sources]
