"""Dataset inspection helpers used by ``notebooks/01_data_exploration.ipynb``.

Kept in the package (not the notebook) so the logic is unit-tested and reusable.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from ml.config import BAND_ORDER, PATCH_SIZE
from ml.dataset import DatasetLayout, PatchRecord, load_patch
from ml.preprocessing import compute_valid_mask
from ml.scene import group_by_scene


def layout_summary(layout: DatasetLayout, records: Sequence[PatchRecord]) -> dict:
    """Counts and resolved paths for one split, for the notebook's first report."""
    return {
        "split": layout.split,
        "split_root": str(layout.split_root),
        "patches_csv": layout.patches_csv.name,
        "sceneids_csv": layout.sceneids_csv.name if layout.sceneids_csv else None,
        "nonempty_csv": layout.nonempty_csv.name if layout.nonempty_csv else None,
        "gt_dir": layout.gt_dir.name if layout.gt_dir else None,
        "scene_gt_dir": layout.scene_gt_dir.name if layout.scene_gt_dir else None,
        "band_dirs": {b: p.name for b, p in layout.band_dirs.items()},
        "n_patches": len(records),
        "n_scenes": len({r.scene_id for r in records}),
    }


def scene_table(records: Sequence[PatchRecord]) -> list[dict]:
    """Per-scene patch count and tile-grid size (rows x cols)."""
    rows = []
    for scene_id, parsed in group_by_scene(r.patch.name for r in records).items():
        n_rows, n_cols = max(p.row for p in parsed), max(p.col for p in parsed)
        rows.append(
            {
                "scene_id": scene_id,
                "n_patches": len(parsed),
                "grid_rows": n_rows,
                "grid_cols": n_cols,
                "grid_complete": len(parsed) == n_rows * n_cols,
                "padded_height": n_rows * PATCH_SIZE,
                "padded_width": n_cols * PATCH_SIZE,
            }
        )
    return rows


def patch_stats(record: PatchRecord) -> dict:
    """Facts about one patch: dtype/range per band, valid fraction, label encoding, cloud share."""
    raw, mask = load_patch(record)
    valid = compute_valid_mask(raw)
    valid_fraction = float(valid.mean())
    stats = {
        "name": record.patch.name,
        "scene_id": record.scene_id,
        "dtype": str(raw.dtype),
        "valid_fraction": valid_fraction,
    }
    for i, band in enumerate(BAND_ORDER):
        stats[f"{band}_min"] = int(raw[i][valid].min()) if valid.any() else None
        stats[f"{band}_max"] = int(raw[i][valid].max()) if valid.any() else None
    if mask is not None:
        stats["cloud_fraction_of_valid"] = float(mask[valid].mean()) if valid.any() else None
        stats["cloud_on_invalid_pixels"] = int(mask[~valid].sum())
    return stats


def sample_records(records: Sequence[PatchRecord], n: int, seed: int = 0) -> list[PatchRecord]:
    """Reproducible random sample of records (for quick inspection before a full scan)."""
    rng = random.Random(seed)
    return rng.sample(list(records), min(n, len(records)))


def scan(records: Sequence[PatchRecord], progress=None) -> list[dict]:
    """Compute ``patch_stats`` for every record (one full pass over the dataset)."""
    iterator = progress(records) if progress else records
    return [patch_stats(r) for r in iterator]


def summarize_scan(rows: Sequence[dict]) -> dict:
    """Aggregate a scan: how many patches are mostly empty, and overall cloud share."""
    valid = np.array([r["valid_fraction"] for r in rows])
    out = {
        "n_patches": len(rows),
        "n_fully_empty": int((valid == 0).sum()),
        "n_below_80pct_valid": int((valid < 0.8).sum()),
        "n_above_80pct_valid": int((valid >= 0.8).sum()),
    }
    clouds = [
        r["cloud_fraction_of_valid"] for r in rows if r.get("cloud_fraction_of_valid") is not None
    ]
    if clouds:
        out["mean_cloud_fraction_of_valid"] = float(np.mean(clouds))
    out["cloud_labels_on_invalid_pixels"] = int(
        sum(r.get("cloud_on_invalid_pixels", 0) for r in rows)
    )
    return out
