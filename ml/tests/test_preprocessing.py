import json

import numpy as np
import pytest

from ml.config import BAND_ORDER
from ml.preprocessing import (
    NormalizationStats,
    PreprocessingError,
    RunningStats,
    compute_valid_mask,
    normalize,
    scale_to_unit,
    validate_raw_array,
)


def make_raw(h=6, w=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 65535, (4, h, w), dtype=np.uint16)


def test_band_order_contract():
    assert BAND_ORDER == ("red", "green", "blue", "nir")


def test_valid_mask_requires_all_bands_zero_for_invalid():
    raw = make_raw()
    raw[:, 0, 0] = 0  # all bands zero -> invalid
    raw[2, 1, 1] = 0  # only blue zero -> still valid
    valid = compute_valid_mask(raw)
    assert not valid[0, 0] and valid[1, 1] and valid.sum() == raw.shape[1] * raw.shape[2] - 1


@pytest.mark.parametrize(
    "arr",
    [
        np.zeros((3, 4, 4), np.uint16),  # wrong band count
        np.zeros((4, 4), np.uint16),  # wrong ndim
        np.zeros((4, 4, 4), np.float32),  # wrong dtype
        np.zeros((4, 4, 4), np.uint8),  # wrong dtype
        np.zeros((4, 0, 4), np.uint16),  # empty
    ],
)
def test_validate_rejects_unsupported_arrays(arr):
    with pytest.raises(PreprocessingError):
        validate_raw_array(arr)


def test_scale_to_unit_range():
    raw = np.full((4, 2, 2), 65535, np.uint16)
    raw[0] = 0
    out = scale_to_unit(raw)
    assert out.dtype == np.float32 and out[0].max() == 0 and np.allclose(out[1:], 1.0)


def test_running_stats_match_numpy_over_valid_pixels():
    rng = np.random.default_rng(1)
    acc, chunks = RunningStats(), []
    for _ in range(4):
        raw = rng.integers(0, 65535, (4, 20, 20), dtype=np.uint16)
        raw[:, :5, :] = 0
        valid = compute_valid_mask(raw)
        unit = scale_to_unit(raw)
        acc.update(unit, valid)
        chunks.append(unit[:, valid])
    stats = acc.finalize()
    allpix = np.concatenate(chunks, axis=1).astype(np.float64)
    assert np.allclose(stats.mean, allpix.mean(axis=1), atol=1e-6)
    assert np.allclose(stats.std, allpix.std(axis=1), atol=1e-6)
    assert stats.provenance["valid_pixel_count"] == allpix.shape[1]


def test_running_stats_empty_raises():
    with pytest.raises(PreprocessingError):
        RunningStats().finalize()


def _stats():
    return NormalizationStats(mean=(0.1, 0.2, 0.3, 0.4), std=(0.5, 0.5, 0.25, 0.25))


def test_normalize_hand_computed_and_nodata_neutral():
    raw = np.zeros((4, 2, 2), np.uint16)
    raw[:, 0, 0] = 65535  # unit value 1.0 in every band
    raw[:, 0, 1] = 32768  # ~0.5
    x, valid = normalize(raw, _stats())
    assert x.dtype == np.float32 and x.shape == raw.shape
    assert np.allclose(
        x[:, 0, 0], [(1 - 0.1) / 0.5, (1 - 0.2) / 0.5, (1 - 0.3) / 0.25, (1 - 0.4) / 0.25]
    )
    assert valid.tolist() == [[True, True], [False, False]]
    assert (x[:, ~valid] == 0).all()  # invalid pixels are neutral


def test_normalization_is_image_independent():
    """The same pixel value normalizes identically regardless of the rest of the image."""
    a, b = make_raw(seed=1), make_raw(seed=2)
    a[:, 0, 0] = b[:, 0, 0] = [1000, 2000, 3000, 4000]
    xa, _ = normalize(a, _stats())
    xb, _ = normalize(b, _stats())
    assert np.array_equal(xa[:, 0, 0], xb[:, 0, 0])


def test_stats_roundtrip(tmp_path):
    s = NormalizationStats(
        mean=(0.1, 0.2, 0.3, 0.4), std=(0.5, 0.5, 0.25, 0.25), provenance={"a": 1}
    )
    path = tmp_path / "n.json"
    s.save(path)
    assert NormalizationStats.load(path) == s


def test_stats_validation_errors(tmp_path):
    with pytest.raises(PreprocessingError):
        NormalizationStats(mean=(0, 0, 0, 0), std=(1, 1, 1, 0))
    with pytest.raises(PreprocessingError):
        NormalizationStats(mean=(0, 0, 0), std=(1, 1, 1))
    with pytest.raises(PreprocessingError):
        NormalizationStats(
            mean=(0, 0, 0, 0), std=(1, 1, 1, 1), band_order=("blue", "green", "red", "nir")
        )
    with pytest.raises(PreprocessingError, match="not found"):
        NormalizationStats.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"mean": [0, 0, 0, 0]}))
    with pytest.raises(PreprocessingError, match="missing key"):
        NormalizationStats.load(bad)
