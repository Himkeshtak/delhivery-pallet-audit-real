#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pallet_audit.data.config import load_sources
from pallet_audit.data.roboflow import LocalArchiveError, import_source_archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and import user-downloaded pinned Roboflow COCO ZIP archives."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/data_sources.yaml"))
    parser.add_argument("--downloads", type=Path, default=Path("data/raw/downloads"))
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_sources(args.config)
    selected = set(args.sources or [])
    known = {source.id for source in config.sources}
    unknown = selected - known
    if unknown:
        print(f"import failed: unknown source ids: {sorted(unknown)}", file=sys.stderr)
        return 2

    sources = [source for source in config.sources if not selected or source.id in selected]
    try:
        for source in sources:
            if not source.local_archive:
                raise LocalArchiveError(f"No local_archive configured for {source.id}")
            destination = import_source_archive(
                source,
                args.downloads / source.local_archive,
                args.output,
                overwrite=args.overwrite,
            )
            print(destination)
    except LocalArchiveError as error:
        print(f"import failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
