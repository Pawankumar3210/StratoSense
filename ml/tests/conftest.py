"""Synthetic 38-Cloud-shaped fixture for tests.

This is NOT real satellite data. It reproduces the *structure* documented by the dataset
authors (folder names, patch naming, CSV format, dtypes, label encoding) so the pipeline
can be unit-tested without downloading the dataset. Band identity is encoded in the pixel
values (``base[band] + patch_index*10 + noise``) so any band mix-up is detectable.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ml.config import BAND_ORDER, PATCH_SIZE

SCENES = [
    "LC08_L1TP_000001_20160101_20170101_01_T1",
    "LC08_L1TP_000002_20160102_20170102_01_T1",
    "LC08_L1TP_000003_20160103_20170103_01_T1",
    "LC08_L1TP_000004_20160104_20170104_01_T1",
    "LC08_L1TP_000005_20160105_20170105_01_T1",
]
GRID = (1, 2)  # rows x cols of patches per fixture scene (kept small for speed)
BAND_BASE = {"red": 10000, "green": 20000, "blue": 30000, "nir": 40000}
MARGIN = 100  # zero-valued margin (rows) on the first patch of each scene


def _write_tif(path: Path, array: np.ndarray) -> None:
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": str(array.dtype),
        "crs": "EPSG:32633",
        "transform": from_origin(500000, 4000000, 30, 30),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def make_fake_dataset(root: Path, *, gt_dirname: str = "train_gt", nest: str | None = None) -> Path:
    """Create a miniature training split and return the dataset root."""
    base = root / nest if nest else root
    split_root = base / "38-Cloud_training"
    for band in BAND_ORDER:
        (split_root / f"train_{band}").mkdir(parents=True)
    (split_root / gt_dirname).mkdir(parents=True)
    (split_root / "Entire_scene_gts").mkdir(parents=True)

    rng = np.random.default_rng(0)
    names, nonempty = [], []
    index = 0
    for scene in SCENES:
        for row in range(1, GRID[0] + 1):
            for col in range(1, GRID[1] + 1):
                index += 1
                name = f"patch_{index}_{row}_by_{col}_{scene}"
                names.append(name)
                margin = row == 1 and col == 1
                if not margin:
                    nonempty.append(name)
                valid = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=bool)
                if margin:
                    valid[:MARGIN, :] = False
                for band in BAND_ORDER:
                    noise = rng.integers(0, 10, (PATCH_SIZE, PATCH_SIZE))
                    data = (BAND_BASE[band] + index * 10 + noise).astype(np.uint16)
                    data[~valid] = 0
                    _write_tif(split_root / f"train_{band}" / f"{band}_{name}.TIF", data)
                gt = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                gt[150:250, 100:300] = 255  # a cloud block, always on valid pixels
                _write_tif(split_root / gt_dirname / f"gt_{name}.TIF", gt)

    with (split_root / "training_patches_38-Cloud.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        writer.writerows([[n] for n in names])
    with (split_root / "training_patches_38-Cloud_nonempty.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        writer.writerows([[n] for n in nonempty])
    with (split_root / "training_sceneids_38-Cloud.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        writer.writerows([[s] for s in SCENES])
    return root


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory) -> Path:
    return make_fake_dataset(tmp_path_factory.mktemp("fake38"))
