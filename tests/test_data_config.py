from pathlib import Path

import pytest

from pallet_audit.data.config import SourceConfigError, load_sources


def test_load_pinned_sources() -> None:
    config = load_sources(Path("configs/data_sources.yaml"))
    assert config.canonical_format == "coco"
    assert {source.id for source in config.sources} == {"pallet_detect_v1", "plh_c_1_v1"}
    assert all(source.version == 1 for source in config.sources)


def test_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "sources.yaml"
    config_path.write_text(
        """schema_version: 1
canonical_format: coco
sources:
  - &source
    id: duplicate
    workspace: w
    project: p
    version: 1
    url: https://example.test/one
    expected_task: object-detection
  - <<: *source
    url: https://example.test/two
""",
        encoding="utf-8",
    )
    with pytest.raises(SourceConfigError, match="Duplicate source id"):
        load_sources(config_path)

