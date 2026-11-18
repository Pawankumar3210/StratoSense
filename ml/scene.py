"""Rebuild complete scene maps from 384x384 test patches (official 38-Cloud protocol).

The official ``evaluation.m`` does the following, and this module mirrors it:

1. Patch location comes from the filename: ``patch_{index}_{row}_by_{col}_{scene_id}``.
   ``row``/``col`` are 1-indexed, so a patch covers rows ``(row-1)*384 : row*384`` and
   columns ``(col-1)*384 : col*384`` of the padded scene.
2. Patches are **non-overlapping** tiles (there is nothing to blend), placed on a zero canvas.
3. The scene was zero-padded before cropping; the padding is removed with a *centred*
   crop of the reference mask size: ``offset = floor((padded - gt) / 2)`` per axis.

Predictions are stitched as probabilities and binarized afterwards. For non-overlapping
tiles this is identical to the official binarize-then-stitch order.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np

from ml.config import PATCH_SIZE
from ml.dataset import DatasetError, PatchName


def group_by_scene(names: Iterable[str]) -> dict[str, list[PatchName]]:
    """Group patch names by scene id (sorted for determinism)."""
    groups: dict[str, list[PatchName]] = defaultdict(list)
    for name in names:
        parsed = PatchName.parse(name)
        groups[parsed.scene_id].append(parsed)
    return {
        scene: sorted(items, key=lambda p: (p.row, p.col))
        for scene, items in sorted(groups.items())
    }


def stitch_patches(
    patches: Mapping[str, np.ndarray],
    scene_id: str,
    *,
    patch_size: int = PATCH_SIZE,
    allow_missing: bool = False,
) -> np.ndarray:
    """Place every patch of ``scene_id`` on the padded-scene canvas.

    ``patches`` maps patch name -> 2-D array of shape ``(patch_size, patch_size)``.
    Raises if a tile of the rectangular grid is missing, unless ``allow_missing`` (missing
    tiles are then left as zeros, which matches the official script's behaviour).
    """
    selected = [PatchName.parse(n) for n in patches if PatchName.parse(n).scene_id == scene_id]
    if not selected:
        raise DatasetError(f"No patches found for scene {scene_id!r}")

    n_rows = max(p.row for p in selected)
    n_cols = max(p.col for p in selected)
    seen: set[tuple[int, int]] = set()
    dtype = next(iter(patches.values())).dtype
    canvas = np.zeros((n_rows * patch_size, n_cols * patch_size), dtype=dtype)

    # Patch dict keys may or may not carry a ".TIF" suffix; resolve through parsed names.
    by_parsed = {PatchName.parse(k).name: v for k, v in patches.items()}
    for patch in selected:
        tile = by_parsed[patch.name]
        if tile.shape != (patch_size, patch_size):
            raise DatasetError(
                f"{patch.name}: expected {(patch_size, patch_size)}, got {tile.shape}"
            )
        key = (patch.row, patch.col)
        if key in seen:
            raise DatasetError(f"Duplicate grid position {key} in scene {scene_id}")
        seen.add(key)
        r0, c0 = (patch.row - 1) * patch_size, (patch.col - 1) * patch_size
        canvas[r0 : r0 + patch_size, c0 : c0 + patch_size] = tile

    missing = n_rows * n_cols - len(seen)
    if missing and not allow_missing:
        raise DatasetError(
            f"Scene {scene_id}: {missing} of {n_rows * n_cols} grid tiles are missing."
        )
    return canvas


def unpad_center(padded: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Centre-crop ``padded`` to ``target_shape`` (``floor`` offsets, as in the official code)."""
    (ny, nx), (nys, nxs) = padded.shape, target_shape
    if nys > ny or nxs > nx:
        raise DatasetError(
            f"Reference mask {target_shape} is larger than reconstructed scene {padded.shape}."
        )
    top, left = (ny - nys) // 2, (nx - nxs) // 2
    return padded[top : top + nys, left : left + nxs]


def reconstruct_scene(
    patches: Mapping[str, np.ndarray],
    scene_id: str,
    target_shape: tuple[int, int],
    *,
    patch_size: int = PATCH_SIZE,
    allow_missing: bool = False,
) -> np.ndarray:
    """Stitch a scene's patches and crop away the padding to match the reference mask."""
    padded = stitch_patches(patches, scene_id, patch_size=patch_size, allow_missing=allow_missing)
    return unpad_center(padded, target_shape)
