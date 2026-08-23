from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class SourceConfigError(ValueError):
    """Raised when the data-source configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class Source:
    id: str
    workspace: str
    project: str
    version: int
    url: str
    expected_task: str
    allowed_licenses: tuple[str, ...]
    local_archive: str | None = None


@dataclass(frozen=True)
class DataSources:
    schema_version: int
    canonical_format: str
    sources: tuple[Source, ...]


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "":
        raise SourceConfigError(f"Missing {key!r} in {context}")
    return value


def load_sources(path: Path) -> DataSources:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SourceConfigError("Data-source config must be a mapping")

    schema_version = int(_required(raw, "schema_version", "root"))
    if schema_version != 1:
        raise SourceConfigError(f"Unsupported schema_version: {schema_version}")
    canonical_format = str(_required(raw, "canonical_format", "root"))
    if canonical_format != "coco":
        raise SourceConfigError("Canonical source format must remain 'coco'")

    raw_sources = _required(raw, "sources", "root")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise SourceConfigError("sources must be a non-empty list")

    sources: list[Source] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_sources):
        context = f"sources[{index}]"
        if not isinstance(item, dict):
            raise SourceConfigError(f"{context} must be a mapping")
        source_id = str(_required(item, "id", context))
        if source_id in seen:
            raise SourceConfigError(f"Duplicate source id: {source_id}")
        seen.add(source_id)
        sources.append(
            Source(
                id=source_id,
                workspace=str(_required(item, "workspace", context)),
                project=str(_required(item, "project", context)),
                version=int(_required(item, "version", context)),
                url=str(_required(item, "url", context)),
                expected_task=str(_required(item, "expected_task", context)),
                allowed_licenses=tuple(str(x) for x in item.get("allowed_licenses", [])),
                local_archive=(
                    str(item["local_archive"]) if item.get("local_archive") else None
                ),
            )
        )
    return DataSources(schema_version, canonical_format, tuple(sources))
