"""Compute per-band normalization statistics from *training-split scenes only*.

Usage::

    python -m ml.compute_stats --data-root /path/to/38-Cloud --out artifacts/normalization.json

The validation scenes (see ``ml.config.SplitConfig``) and all official test scenes are
excluded, so no information from held-out data leaks into preprocessing.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from ml.config import REPO_ROOT, SplitConfig, TrainConfig, data_root_from_env
from ml.dataset import (
    DatasetLayout,
    build_patch_index,
    filter_nonempty,
    load_patch,
    read_names_csv,
    records_for_scenes,
    split_scenes,
    unique_scene_ids,
)
from ml.preprocessing import RunningStats, compute_valid_mask, scale_to_unit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "normalization.json")
    parser.add_argument("--max-patches", type=int, default=None, help="Debug only: limit patches.")
    parser.add_argument(
        "--all-patches", action="store_true", help="Ignore the nonempty-patch list."
    )
    args = parser.parse_args(argv)

    data_root = args.data_root or data_root_from_env()
    layout = DatasetLayout.discover(data_root, "train")
    records = build_patch_index(layout, with_gt=True)

    use_nonempty = TrainConfig().use_nonempty_patches_only and not args.all_patches
    if use_nonempty:
        if layout.nonempty_csv is None:
            raise SystemExit("No *_nonempty.csv found; pass --all-patches to proceed without it.")
        records = filter_nonempty(records, read_names_csv(layout.nonempty_csv))

    train_scenes, val_scenes = split_scenes(unique_scene_ids(records), SplitConfig())
    train_records = records_for_scenes(records, train_scenes)
    if args.max_patches:
        train_records = train_records[: args.max_patches]

    acc = RunningStats()
    for i, record in enumerate(train_records, 1):
        raw, _ = load_patch(record)
        acc.update(scale_to_unit(raw), compute_valid_mask(raw))
        if i % 500 == 0:
            print(f"  processed {i}/{len(train_records)} patches")

    stats = acc.finalize(
        {
            "dataset": "38-Cloud (training split)",
            "train_scenes": train_scenes,
            "excluded_validation_scenes": val_scenes,
            "split_config": SplitConfig().__dict__,
            "n_patches": len(train_records),
            "nonempty_patches_only": use_nonempty,
            "debug_limited": bool(args.max_patches),
            "computed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    stats.save(args.out)
    print(f"Saved {args.out}\n  mean={stats.mean}\n  std ={stats.std}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
