"""PyTorch ``Dataset`` over 38-Cloud patches. Importing this module requires PyTorch."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.dataset import PatchRecord, load_patch
from ml.preprocessing import NormalizationStats, normalize


def apply_dihedral(k: int, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Apply one of the 8 flip/rotate transforms identically to every array.

    Arrays are ``(..., H, W)``. ``k`` in 0..7: ``k % 4`` quarter-turns, flipped if ``k >= 4``.
    These transforms preserve the geometry of satellite scenes, so image, mask and validity
    mask stay perfectly aligned.
    """
    out = []
    for arr in arrays:
        t = np.rot90(arr, k % 4, axes=(-2, -1))
        if k >= 4:
            t = np.flip(t, axis=-1)
        out.append(np.ascontiguousarray(t))
    return tuple(out)


class CloudPatchDataset(Dataset):
    """Yields ``(x, y, valid)``: normalized image, 0/1 mask and validity mask tensors.

    * ``x``: float32 ``(4, 384, 384)``
    * ``y``: float32 ``(1, 384, 384)``
    * ``valid``: float32 ``(1, 384, 384)``; 0 on nodata margin pixels (exclude from loss/metrics)
    """

    def __init__(
        self,
        records: Sequence[PatchRecord],
        stats: NormalizationStats,
        *,
        augment: bool = False,
    ) -> None:
        if any(r.gt_path is None for r in records):
            raise ValueError("CloudPatchDataset needs ground truth for every record.")
        self.records = list(records)
        self.stats = stats
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw, mask = load_patch(self.records[idx])
        assert mask is not None
        x, valid = normalize(raw, self.stats)
        y = mask.astype(np.float32)
        v = valid.astype(np.float32)
        if self.augment:
            # torch RNG is seeded per DataLoader worker, unlike numpy's global RNG.
            k = int(torch.randint(0, 8, ()).item())
            x, y, v = apply_dihedral(k, x, y, v)
        return torch.from_numpy(x), torch.from_numpy(y)[None], torch.from_numpy(v)[None]
