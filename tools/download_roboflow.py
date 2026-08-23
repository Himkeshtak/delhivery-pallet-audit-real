#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pallet_audit.data.roboflow import RoboflowDownloadError, download_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pinned Roboflow sources as canonical COCO exports."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/data_sources.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        destinations = download_sources(
            args.config, args.output, selected=args.sources, overwrite=args.overwrite
        )
    except RoboflowDownloadError as error:
        print(f"download failed: {error}", file=sys.stderr)
        return 2
    for destination in destinations:
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

