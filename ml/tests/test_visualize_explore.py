import numpy as np

from ml.dataset import DatasetLayout, build_patch_index
from ml.explore import (
    layout_summary,
    patch_stats,
    sample_records,
    scan,
    scene_table,
    summarize_scan,
)
from ml.preprocessing import compute_valid_mask
from ml.visualize import rgb_composite


def test_rgb_composite_shape_dtype_and_black_nodata():
    rng = np.random.default_rng(0)
    raw = rng.integers(1, 60000, (4, 32, 32), dtype=np.uint16)
    raw[:, :4, :] = 0
    valid = compute_valid_mask(raw)
    rgb = rgb_composite(raw, valid)
    assert rgb.shape == (32, 32, 3) and rgb.dtype == np.uint8
    assert (rgb[~valid] == 0).all()
    assert rgb[valid].max() == 255 and rgb[valid].min() == 0  # stretch used full range


def test_rgb_uses_red_green_blue_not_nir():
    raw = np.zeros((4, 8, 8), np.uint16)
    raw[0], raw[1], raw[2], raw[3] = 1000, 2000, 3000, 65000
    raw[0, 0, 0] = 50000  # ensure red has dynamic range
    rgb = rgb_composite(raw, np.ones((8, 8), bool), low=0, high=100)
    assert rgb[0, 0, 0] == 255 and rgb[0, 0, 1] == 0  # red channel is band 0


def test_rgb_composite_all_invalid_returns_black():
    raw = np.zeros((4, 4, 4), np.uint16)
    assert not rgb_composite(raw, np.zeros((4, 4), bool)).any()


def test_explore_pipeline_on_fixture(fake_root):
    layout = DatasetLayout.discover(fake_root, "train")
    records = build_patch_index(layout)
    summary = layout_summary(layout, records)
    assert summary["n_patches"] == 10 and summary["n_scenes"] == 5

    table = scene_table(records)
    assert all(
        row["grid_rows"] == 1 and row["grid_cols"] == 2 and row["grid_complete"] for row in table
    )

    stats = patch_stats(records[0])  # the margin patch
    assert stats["dtype"] == "uint16"
    assert abs(stats["valid_fraction"] - (384 - 100) / 384) < 1e-9
    assert stats["cloud_on_invalid_pixels"] == 0
    assert stats["red_min"] >= 10000 and stats["nir_min"] >= 40000

    rows = scan(records)
    agg = summarize_scan(rows)
    # one margin patch per scene is 74% valid (< 80%), mirroring the official "nonempty" rule
    assert agg["n_patches"] == 10
    assert agg["n_below_80pct_valid"] == 5 and agg["n_above_80pct_valid"] == 5
    assert agg["n_fully_empty"] == 0
    assert len(sample_records(records, 3, seed=1)) == 3
    assert sample_records(records, 3, seed=1) == sample_records(records, 3, seed=1)
