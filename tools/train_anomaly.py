#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_folder_layout(root: Path) -> dict[str, Path]:
    paths = {
        "normal_train": root / "train" / "good",
        "normal_test": root / "test" / "good",
        "abnormal_test": root / "test" / "bad",
    }
    missing = [name for name, path in paths.items() if not path.is_dir()]
    if missing:
        raise ValueError(f"anomaly dataset is missing required directories: {missing}")
    if not any(paths["normal_train"].iterdir()):
        raise ValueError("normal training directory is empty")
    if not any(paths["abnormal_test"].iterdir()):
        raise ValueError("abnormal test directory is empty; AUROC/F1 cannot be measured")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and test the pinned EfficientAD or PatchCore visible-damage experiment."
    )
    parser.add_argument("--model", choices=("efficientad", "patchcore"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/anomaly"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--imagenette", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_folder_layout(args.data)
    try:
        import anomalib
        import torch
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import EfficientAd, Patchcore
        from anomalib.pre_processing import PreProcessor
        from torchvision.transforms.v2 import Compose, Resize
    except ImportError as error:
        raise SystemExit(
            "Anomalib 2.6 requires Python <=3.12. Use a Python 3.11/3.12 ML environment "
            "and install: python -m pip install -e '.[ml]'"
        ) from error

    datamodule = Folder(
        name="pallet_visible_damage",
        root=args.data,
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/bad",
        mask_dir="ground_truth/bad" if (args.data / "ground_truth" / "bad").is_dir() else None,
        train_batch_size=1,
        eval_batch_size=8,
        num_workers=0,
        seed=args.seed,
    )
    pre_processor = PreProcessor(
        transform=Compose([Resize(size=(args.image_size, args.image_size))])
    )
    if args.model == "efficientad":
        kwargs: dict[str, Any] = {"model_size": "small", "pre_processor": pre_processor}
        if args.imagenette:
            kwargs["imagenet_dir"] = str(args.imagenette)
        model = EfficientAd(**kwargs)
        maximum_epochs = 20
    else:
        model = Patchcore(pre_processor=pre_processor)
        maximum_epochs = 1

    run_root = args.output / args.model
    engine = Engine(
        default_root_dir=run_root,
        max_epochs=maximum_epochs,
        accelerator=args.accelerator,
        devices=1,
        deterministic=True,
    )
    started = datetime.now(timezone.utc)
    engine.fit(model=model, datamodule=datamodule)
    test_results = engine.test(model=model, datamodule=datamodule)
    checkpoint_candidates = sorted(run_root.rglob("*.ckpt"))
    manifest = {
        "manifest_schema_version": 1,
        "model": args.model,
        "anomalib_version": anomalib.__version__,
        "data_root": str(args.data.resolve()),
        "seed": args.seed,
        "image_size_requested": args.image_size,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "test_results": test_results,
        "checkpoints": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in checkpoint_candidates
        ],
        "scope_warning": (
            "Scores describe visible cropped surfaces only; hidden pallet/load damage "
            "remains unknown."
        ),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
