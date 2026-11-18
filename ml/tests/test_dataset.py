import csv

import numpy as np
import pytest

from ml.config import BAND_ORDER, SplitConfig
from ml.dataset import (
    DatasetError,
    DatasetLayout,
    PatchName,
    build_patch_index,
    decode_mask,
    filter_nonempty,
    load_patch,
    read_names_csv,
    records_for_scenes,
    split_scenes,
    unique_scene_ids,
)
from ml.tests.conftest import BAND_BASE, MARGIN, SCENES, make_fake_dataset

NAME = "patch_192_10_by_12_LC08_L1TP_002053_20160520_20170324_01_T1"


# ---- patch names ---------------------------------------------------------------------
def test_parse_patch_name_matches_official_sample():
    p = PatchName.parse(NAME)
    assert (p.index, p.row, p.col) == (192, 10, 12)
    assert p.scene_id == "LC08_L1TP_002053_20160520_20170324_01_T1"


def test_parse_accepts_tif_suffix():
    assert PatchName.parse(NAME + ".TIF").name == NAME
    assert PatchName.parse(NAME + ".tif").name == NAME


@pytest.mark.parametrize(
    "bad",
    ["red_" + NAME, "patch_1_0_by_1_LC08_X", "patch_1_1_by_1_notascene", "foo", ""],
)
def test_parse_rejects_malformed_names(bad):
    with pytest.raises(DatasetError):
        PatchName.parse(bad)


# ---- layout discovery ----------------------------------------------------------------
def test_discover_standard_layout(fake_root):
    layout = DatasetLayout.discover(fake_root, "train")
    assert set(layout.band_dirs) == set(BAND_ORDER)
    assert layout.gt_dir.name == "train_gt"
    assert layout.patches_csv.name == "training_patches_38-Cloud.csv"
    assert layout.nonempty_csv.name.endswith("nonempty.csv")
    assert layout.scene_gt_dir.name == "Entire_scene_gts"


def test_discover_accepts_train_gts_spelling(tmp_path):
    root = make_fake_dataset(tmp_path, gt_dirname="train_gts")
    assert DatasetLayout.discover(root, "train").gt_dir.name == "train_gts"


def test_discover_finds_nested_split_folder(tmp_path):
    root = make_fake_dataset(tmp_path, nest="kaggle_download/extra")
    assert DatasetLayout.discover(root, "train").split_root.name == "38-Cloud_training"


def test_discover_missing_root_raises(tmp_path):
    with pytest.raises(DatasetError):
        DatasetLayout.discover(tmp_path / "nope", "train")


def test_discover_missing_band_dir_raises(tmp_path):
    root = make_fake_dataset(tmp_path)
    (root / "38-Cloud_training" / "train_nir").rename(root / "38-Cloud_training" / "oops")
    with pytest.raises(DatasetError, match="train_nir"):
        DatasetLayout.discover(root, "train")


# ---- CSV + index ---------------------------------------------------------------------
def test_read_names_csv_rejects_duplicates_and_bad_header(tmp_path):
    dup = tmp_path / "dup.csv"
    dup.write_text("name\nA\nA\n")
    with pytest.raises(DatasetError, match="duplicate"):
        read_names_csv(dup)
    bad = tmp_path / "bad.csv"
    bad.write_text("id\nA\n")
    with pytest.raises(DatasetError, match="'name' column"):
        read_names_csv(bad)


def test_index_matches_all_patches_and_band_order(fake_root):
    layout = DatasetLayout.discover(fake_root, "train")
    records = build_patch_index(layout)
    assert len(records) == len(SCENES) * 2
    for rec in records:
        assert len(rec.band_paths) == 4
        for band, path in zip(BAND_ORDER, rec.band_paths, strict=True):
            assert path.name == f"{band}_{rec.patch.name}.TIF"
        assert rec.gt_path.name == f"gt_{rec.patch.name}.TIF"


def test_index_raises_on_missing_file(tmp_path):
    root = make_fake_dataset(tmp_path)
    split = root / "38-Cloud_training"
    victim = next((split / "train_green").iterdir())
    victim.unlink()
    layout = DatasetLayout.discover(root, "train")
    with pytest.raises(DatasetError, match="missing"):
        build_patch_index(layout)


def test_index_is_independent_of_csv_order(fake_root):
    layout = DatasetLayout.discover(fake_root, "train")
    names = read_names_csv(layout.patches_csv)
    forward = build_patch_index(layout, names)
    backward = build_patch_index(layout, list(reversed(names)))
    assert {r.patch.name: r.band_paths for r in forward} == {
        r.patch.name: r.band_paths for r in backward
    }


# ---- loading -------------------------------------------------------------------------
def test_load_patch_band_identity_dtype_and_shape(fake_root):
    layout = DatasetLayout.discover(fake_root, "train")
    record = build_patch_index(layout)[1]  # not a margin patch
    raw, mask = load_patch(record)
    assert raw.shape == (4, 384, 384) and raw.dtype == np.uint16
    assert mask.shape == (384, 384) and set(np.unique(mask)) <= {0, 1}
    for i, band in enumerate(BAND_ORDER):  # band identity survives the stack
        assert raw[i].min() >= BAND_BASE[band]
        assert raw[i].max() < BAND_BASE[band] + 10000


def test_load_patch_preserves_zero_margin_and_gt_alignment(fake_root):
    layout = DatasetLayout.discover(fake_root, "train")
    record = build_patch_index(layout)[0]  # row 1, col 1 -> has margin
    raw, mask = load_patch(record)
    assert (raw[:, :MARGIN, :] == 0).all()
    assert (raw[:, MARGIN:, :] > 0).all()
    assert mask[150:250, 100:300].all() and mask.sum() == 100 * 200


def test_decode_mask_accepts_0_255_and_rejects_other_encodings():
    assert decode_mask(np.array([[0, 255]], dtype=np.uint8)).tolist() == [[0, 1]]
    with pytest.raises(DatasetError):
        decode_mask(np.array([[0, 1]], dtype=np.uint8))  # unexpected encoding
    with pytest.raises(DatasetError):
        decode_mask(np.array([[0, 128, 255]], dtype=np.uint8))


# ---- scene-level split ---------------------------------------------------------------
def test_split_is_deterministic_disjoint_and_complete():
    a = split_scenes(SCENES, SplitConfig(0.2, 7))
    b = split_scenes(list(reversed(SCENES)), SplitConfig(0.2, 7))
    assert a == b
    train, val = a
    assert not set(train) & set(val)
    assert sorted(train + val) == sorted(SCENES)
    assert len(val) == 1


def test_split_changes_with_seed_but_never_leaks():
    outcomes = {tuple(split_scenes(SCENES, SplitConfig(0.4, s))[1]) for s in range(10)}
    assert len(outcomes) > 1
    for s in range(10):
        train, val = split_scenes(SCENES, SplitConfig(0.4, s))
        assert val and train and not set(train) & set(val)


def test_split_requires_two_scenes():
    with pytest.raises(DatasetError):
        split_scenes(["only_one"])


def test_no_patch_from_validation_scene_in_training(fake_root):
    records = build_patch_index(DatasetLayout.discover(fake_root, "train"))
    train_s, val_s = split_scenes(unique_scene_ids(records), SplitConfig())
    train_r, val_r = records_for_scenes(records, train_s), records_for_scenes(records, val_s)
    assert {r.scene_id for r in train_r}.isdisjoint({r.scene_id for r in val_r})
    assert len(train_r) + len(val_r) == len(records)


def test_filter_nonempty_uses_official_list(fake_root):
    layout = DatasetLayout.discover(fake_root, "train")
    records = build_patch_index(layout)
    kept = filter_nonempty(records, read_names_csv(layout.nonempty_csv))
    assert len(kept) == len(records) - len(SCENES)  # one margin patch per scene removed
    assert all(not (r.patch.row == 1 and r.patch.col == 1) for r in kept)


def test_csv_with_bom_and_tif_suffix(tmp_path):
    p = tmp_path / "x.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([["name"], [NAME + ".TIF"]])
    assert read_names_csv(p) == [NAME]
