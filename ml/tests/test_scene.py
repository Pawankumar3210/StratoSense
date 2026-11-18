"""Tests for official-protocol scene reconstruction.

The synthetic scenes are padded and cut following the rules read from the official
``evaluation.m`` (centred floor padding, 1-indexed row/col, non-overlapping tiles). These
tests verify our implementation is self-consistent with that reading; the real padding
geometry is asserted against actual reference masks when the dataset is available.
"""

import numpy as np
import pytest

from ml.config import PATCH_SIZE
from ml.dataset import DatasetError
from ml.scene import group_by_scene, reconstruct_scene, stitch_patches, unpad_center

SCENE = "LC08_L1TP_002053_20160520_20170324_01_T1"
P = PATCH_SIZE


def cut_into_patches(scene: np.ndarray, scene_id=SCENE, suffix=""):
    """Pad `scene` (centred, floor offsets) to a multiple of 384 and cut official-named tiles."""
    h, w = scene.shape
    rows, cols = -(-h // P), -(-w // P)
    padded = np.zeros((rows * P, cols * P), dtype=scene.dtype)
    top, left = (rows * P - h) // 2, (cols * P - w) // 2
    padded[top : top + h, left : left + w] = scene
    patches, idx = {}, 0
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            idx += 1
            tile = padded[(r - 1) * P : r * P, (c - 1) * P : c * P]
            patches[f"patch_{idx}_{r}_by_{c}_{scene_id}{suffix}"] = tile.copy()
    return patches


@pytest.mark.parametrize("shape", [(1000, 900), (1001, 1000), (384, 384), (500, 1200), (770, 385)])
def test_roundtrip_recovers_original_scene(shape):
    rng = np.random.default_rng(0)
    scene = rng.random(shape, dtype=np.float32) + 0.1  # strictly non-zero
    patches = cut_into_patches(scene)
    assert np.array_equal(reconstruct_scene(patches, SCENE, shape), scene)


def test_roundtrip_with_tif_suffix_keys():
    scene = np.random.default_rng(1).random((700, 800), dtype=np.float32)
    patches = cut_into_patches(scene, suffix=".TIF")
    assert np.array_equal(reconstruct_scene(patches, SCENE, scene.shape), scene)


def test_patch_order_does_not_matter():
    scene = np.random.default_rng(2).random((800, 800), dtype=np.float32)
    patches = cut_into_patches(scene)
    shuffled = dict(reversed(list(patches.items())))
    assert np.array_equal(reconstruct_scene(shuffled, SCENE, scene.shape), scene)


def test_only_requested_scene_is_used():
    a = np.ones((400, 400), np.float32)
    b = np.full((400, 400), 2.0, np.float32)
    other = "LC08_L1TP_009999_20160101_20170101_01_T1"
    patches = {**cut_into_patches(a), **cut_into_patches(b, other)}
    assert np.array_equal(reconstruct_scene(patches, SCENE, a.shape), a)
    assert np.array_equal(reconstruct_scene(patches, other, b.shape), b)


def test_tile_position_is_one_indexed():
    patches = {
        f"patch_1_1_by_1_{SCENE}": np.full((P, P), 1.0, np.float32),
        f"patch_2_1_by_2_{SCENE}": np.full((P, P), 2.0, np.float32),
        f"patch_3_2_by_1_{SCENE}": np.full((P, P), 3.0, np.float32),
        f"patch_4_2_by_2_{SCENE}": np.full((P, P), 4.0, np.float32),
    }
    canvas = stitch_patches(patches, SCENE)
    assert canvas.shape == (2 * P, 2 * P)
    assert canvas[0, 0] == 1 and canvas[0, P] == 2 and canvas[P, 0] == 3 and canvas[P, P] == 4


def test_missing_tile_raises_unless_allowed():
    patches = {
        f"patch_1_1_by_1_{SCENE}": np.ones((P, P), np.float32),
        f"patch_4_2_by_2_{SCENE}": np.ones((P, P), np.float32),
    }
    with pytest.raises(DatasetError, match="missing"):
        stitch_patches(patches, SCENE)
    canvas = stitch_patches(patches, SCENE, allow_missing=True)
    assert canvas[0, P] == 0  # gap left as zero, like the official script


def test_bad_inputs_raise():
    with pytest.raises(DatasetError):
        stitch_patches({f"patch_1_1_by_1_{SCENE}": np.ones((10, 10), np.float32)}, SCENE)
    with pytest.raises(DatasetError):
        stitch_patches({f"patch_1_1_by_1_{SCENE}": np.ones((P, P))}, "LC08_unknown_scene")
    with pytest.raises(DatasetError):
        unpad_center(np.zeros((10, 10)), (11, 10))


def test_unpad_uses_floor_offsets():
    padded = np.arange(10 * 10).reshape(10, 10)
    out = unpad_center(padded, (7, 4))  # offsets floor(3/2)=1, floor(6/2)=3
    assert np.array_equal(out, padded[1:8, 3:7])


def test_group_by_scene_sorts_by_grid_position():
    names = [f"patch_2_2_by_1_{SCENE}", f"patch_1_1_by_2_{SCENE}", f"patch_9_1_by_1_{SCENE}"]
    grouped = group_by_scene(names)
    assert [(p.row, p.col) for p in grouped[SCENE]] == [(1, 1), (1, 2), (2, 1)]
