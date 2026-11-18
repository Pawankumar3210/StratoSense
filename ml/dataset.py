"""38-Cloud dataset access: discovery, name-based band matching, loading and splitting.

Conventions verified against the official dataset README and the authors' Cloud-Net code:

* The patch CSVs have a ``name`` column, e.g. ``patch_192_10_by_12_LC08_L1TP_..._T1``.
* A band file is ``{band}_{name}.TIF`` inside ``{split}_{band}/``; the ground truth is
  ``gt_{name}.TIF``. The README calls the training GT folder ``train_gt`` while the authors'
  code uses ``train_gts``; both spellings are accepted.
* ``patch_{index}_{row}_by_{col}_{scene_id}``: ``row``/``col`` are 1-indexed grid positions
  of the 384x384 tile inside the (zero-padded) scene.
* Images are 16-bit; ground-truth masks are 0 (clear) / 255 (cloud).

Bands are matched strictly through patch names. Directory ordering is never relied upon.
This module is NumPy/rasterio only (no PyTorch).
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.config import (
    BAND_ORDER,
    MASK_CLEAR_VALUE,
    MASK_CLOUD_VALUE,
    PATCH_SIZE,
    RAW_DTYPE,
    SplitConfig,
)


class DatasetError(RuntimeError):
    """Raised when the dataset on disk does not match the documented structure."""


# --------------------------------------------------------------------------------------
# Patch names
# --------------------------------------------------------------------------------------
_PATCH_NAME_RE = re.compile(r"^patch_(?P<index>\d+)_(?P<row>\d+)_by_(?P<col>\d+)_(?P<scene>LC.+)$")
_TIF_SUFFIXES = (".tif", ".tiff")


def strip_tif_suffix(name: str) -> str:
    """Remove a trailing ``.TIF``/``.tif`` (case-insensitive) if present."""
    return name[:-4] if name.lower().endswith(".tif") else name


@dataclass(frozen=True)
class PatchName:
    """Parsed ``patch_{index}_{row}_by_{col}_{scene_id}`` identifier."""

    name: str
    index: int
    row: int
    col: int
    scene_id: str

    @classmethod
    def parse(cls, name: str) -> PatchName:
        stem = strip_tif_suffix(name.strip())
        match = _PATCH_NAME_RE.match(stem)
        if match is None:
            raise DatasetError(f"Not a valid 38-Cloud patch name: {name!r}")
        row, col = int(match["row"]), int(match["col"])
        if row < 1 or col < 1:
            raise DatasetError(f"Patch row/col must be 1-indexed, got {row}/{col} in {name!r}")
        return cls(stem, int(match["index"]), row, col, match["scene"])


def band_file_stem(band: str, patch_name: str) -> str:
    return f"{band}_{patch_name}"


def gt_file_stem(patch_name: str) -> str:
    return f"gt_{patch_name}"


# --------------------------------------------------------------------------------------
# Layout discovery
# --------------------------------------------------------------------------------------
def _find_dir(root: Path, names: Sequence[str], max_depth: int = 3) -> Path | None:
    """Breadth-first search for a directory whose name matches (case-insensitive)."""
    wanted = {n.lower() for n in names}
    level = [root]
    for _ in range(max_depth + 1):
        next_level: list[Path] = []
        matches = []
        for directory in level:
            if not directory.is_dir():
                continue
            if directory.name.lower() in wanted:
                matches.append(directory)
            next_level.extend(sorted(p for p in directory.iterdir() if p.is_dir()))
        if len(matches) > 1:
            raise DatasetError(f"Ambiguous directory {names} under {root}: {matches}")
        if matches:
            return matches[0]
        level = next_level
    return None


def _index_tifs(directory: Path) -> dict[str, Path]:
    """Map file stem -> path for every TIFF in ``directory``."""
    index: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in _TIF_SUFFIXES:
            if path.stem in index:
                raise DatasetError(f"Duplicate stem {path.stem!r} in {directory}")
            index[path.stem] = path
    return index


@dataclass(frozen=True)
class DatasetLayout:
    """Resolved locations of one 38-Cloud split (``train`` or ``test``)."""

    split: str
    split_root: Path
    band_dirs: dict[str, Path]
    patches_csv: Path
    sceneids_csv: Path | None
    nonempty_csv: Path | None
    gt_dir: Path | None  # per-patch GT (train only)
    scene_gt_dir: Path | None  # complete-scene GT (``Entire_scene_gts``)

    @classmethod
    def discover(cls, data_root: str | Path, split: str) -> DatasetLayout:
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        root = Path(data_root).expanduser()
        if not root.is_dir():
            raise DatasetError(f"Dataset root does not exist: {root}")

        folder_names = ("38-cloud_training",) if split == "train" else ("38-cloud_test",)
        split_root = _find_dir(root, folder_names)
        if split_root is None:  # allow pointing directly at the split folder
            if (root / f"{split}_{BAND_ORDER[0]}").is_dir():
                split_root = root
            else:
                raise DatasetError(
                    f"Could not find a '{folder_names[0]}' folder (or {split}_red) under {root}."
                )

        band_dirs: dict[str, Path] = {}
        for band in BAND_ORDER:
            found = _find_dir(split_root, (f"{split}_{band}",), max_depth=1)
            if found is None:
                raise DatasetError(f"Missing band directory '{split}_{band}' in {split_root}")
            band_dirs[band] = found

        # The full patch list is distinct from the optional "nonempty" list.
        csvs = sorted(split_root.glob("*.csv"))
        prefix = "training_patches" if split == "train" else "test_patches"
        full = [
            c
            for c in csvs
            if c.name.lower().startswith(prefix) and "nonempty" not in c.name.lower()
        ]
        nonempty = [c for c in csvs if "nonempty" in c.name.lower()]
        scene_csv = [c for c in csvs if "sceneid" in c.name.lower()]
        if len(full) != 1:
            raise DatasetError(
                f"Expected exactly one '{prefix}*.csv' in {split_root}, found {full}"
            )
        patches_csv = full[0]
        if len(nonempty) > 1 or len(scene_csv) > 1:
            raise DatasetError(f"Ambiguous CSV files in {split_root}: {nonempty + scene_csv}")

        gt_dir = None
        if split == "train":
            gt_dir = _find_dir(split_root, ("train_gt", "train_gts"), max_depth=1)
            if gt_dir is None:
                raise DatasetError(
                    f"Missing ground-truth folder (train_gt/train_gts) in {split_root}"
                )

        return cls(
            split=split,
            split_root=split_root,
            band_dirs=band_dirs,
            patches_csv=patches_csv,
            sceneids_csv=scene_csv[0] if scene_csv else None,
            nonempty_csv=nonempty[0] if nonempty else None,
            gt_dir=gt_dir,
            scene_gt_dir=_find_dir(split_root, ("entire_scene_gts",), max_depth=1),
        )


# --------------------------------------------------------------------------------------
# CSV reading and patch index
# --------------------------------------------------------------------------------------
def read_names_csv(path: str | Path) -> list[str]:
    """Read patch names from a CSV with a ``name`` column. Duplicates are an error."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetError(f"{path} is empty")
        column = next((f for f in reader.fieldnames if f.strip().lower() == "name"), None)
        if column is None:
            raise DatasetError(f"{path} has no 'name' column (columns: {reader.fieldnames})")
        names = [strip_tif_suffix(row[column].strip()) for row in reader if row[column].strip()]
    if len(set(names)) != len(names):
        raise DatasetError(f"{path} contains duplicate patch names")
    return names


@dataclass(frozen=True)
class PatchRecord:
    """One patch: parsed name, band file paths in model order, and optional GT path."""

    patch: PatchName
    band_paths: tuple[Path, ...]
    gt_path: Path | None = None

    @property
    def scene_id(self) -> str:
        return self.patch.scene_id


def build_patch_index(
    layout: DatasetLayout,
    names: Iterable[str] | None = None,
    *,
    with_gt: bool | None = None,
) -> list[PatchRecord]:
    """Match each patch name to its four band files (and GT) by *name*.

    Raises ``DatasetError`` listing missing files rather than silently skipping them.
    """
    if names is None:
        names = read_names_csv(layout.patches_csv)
    names = list(names)
    if with_gt is None:
        with_gt = layout.gt_dir is not None
    if with_gt and layout.gt_dir is None:
        raise DatasetError("Per-patch ground truth requested but this split has none.")

    band_index = {band: _index_tifs(layout.band_dirs[band]) for band in BAND_ORDER}
    gt_index = _index_tifs(layout.gt_dir) if with_gt and layout.gt_dir else {}

    records: list[PatchRecord] = []
    missing: list[str] = []
    for name in names:
        parsed = PatchName.parse(name)
        paths: list[Path] = []
        for band in BAND_ORDER:
            path = band_index[band].get(band_file_stem(band, parsed.name))
            if path is None:
                missing.append(f"{band}: {parsed.name}")
            else:
                paths.append(path)
        gt_path = None
        if with_gt:
            gt_path = gt_index.get(gt_file_stem(parsed.name))
            if gt_path is None:
                missing.append(f"gt: {parsed.name}")
        if len(paths) == len(BAND_ORDER) and (gt_path is not None or not with_gt):
            records.append(PatchRecord(parsed, tuple(paths), gt_path))
    if missing:
        raise DatasetError(
            f"{len(missing)} expected files are missing (first 5: {missing[:5]}). "
            "Check that the dataset was fully extracted."
        )
    return records


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _read_band(path: Path, size: int = PATCH_SIZE) -> np.ndarray:
    import rasterio  # imported lazily so the module stays cheap to import

    with rasterio.open(path) as src:
        if src.count != 1:
            raise DatasetError(f"{path.name}: expected 1 band, found {src.count}")
        data = src.read(1)
    if data.shape != (size, size):
        raise DatasetError(f"{path.name}: expected {(size, size)}, got {data.shape}")
    return data


def decode_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a 0/255 ground-truth array to uint8 0/1. Any other encoding is an error."""
    values = np.unique(mask)
    if not set(values.tolist()) <= {MASK_CLEAR_VALUE, MASK_CLOUD_VALUE}:
        raise DatasetError(
            f"Unexpected label values {values[:10].tolist()}; expected only "
            f"{{{MASK_CLEAR_VALUE}, {MASK_CLOUD_VALUE}}}. Inspect the data before training."
        )
    return (mask == MASK_CLOUD_VALUE).astype(np.uint8)


def load_patch(record: PatchRecord) -> tuple[np.ndarray, np.ndarray | None]:
    """Load one patch as ``raw (4, 384, 384) uint16`` and ``mask (384, 384) uint8 {0,1}``."""
    bands = [_read_band(p) for p in record.band_paths]
    for band, arr in zip(BAND_ORDER, bands, strict=True):
        if arr.dtype != np.dtype(RAW_DTYPE):
            raise DatasetError(
                f"{band} band of {record.patch.name} is {arr.dtype}, expected {RAW_DTYPE}"
            )
    raw = np.stack(bands, axis=0)
    mask = decode_mask(_read_band(record.gt_path)) if record.gt_path is not None else None
    return raw, mask


# --------------------------------------------------------------------------------------
# Scene-level splitting (prevents spatial leakage between train and validation)
# --------------------------------------------------------------------------------------
def unique_scene_ids(records: Iterable[PatchRecord]) -> list[str]:
    return sorted({r.scene_id for r in records})


def split_scenes(
    scene_ids: Sequence[str], config: SplitConfig | None = None
) -> tuple[list[str], list[str]]:
    """Deterministically split *scenes* (never patches) into train and validation.

    Scenes are ranked by ``sha256(f"{seed}:{scene_id}")`` so the result is identical across
    machines, Python versions and input ordering.
    """
    config = config or SplitConfig()
    unique = sorted(set(scene_ids))
    if len(unique) < 2:
        raise DatasetError("Need at least two scenes for a train/validation split.")
    if not 0.0 < config.val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    ranked = sorted(unique, key=lambda s: hashlib.sha256(f"{config.seed}:{s}".encode()).hexdigest())
    n_val = min(max(1, round(len(unique) * config.val_fraction)), len(unique) - 1)
    val = sorted(ranked[:n_val])
    train = sorted(ranked[n_val:])
    assert_disjoint_scenes(train, val)
    return train, val


def assert_disjoint_scenes(train: Iterable[str], val: Iterable[str]) -> None:
    overlap = set(train) & set(val)
    if overlap:
        raise DatasetError(f"Scene leakage between splits: {sorted(overlap)}")


def records_for_scenes(records: Iterable[PatchRecord], scenes: Iterable[str]) -> list[PatchRecord]:
    wanted = set(scenes)
    return [r for r in records if r.scene_id in wanted]


def filter_nonempty(
    records: Iterable[PatchRecord], nonempty_names: Iterable[str]
) -> list[PatchRecord]:
    """Keep only patches listed in the official ``*_nonempty.csv`` (>80% informative pixels)."""
    keep = {strip_tif_suffix(n) for n in nonempty_names}
    return [r for r in records if r.patch.name in keep]
