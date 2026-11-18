import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml.dataset import DatasetLayout, build_patch_index  # noqa: E402
from ml.preprocessing import NormalizationStats  # noqa: E402
from ml.torch_dataset import CloudPatchDataset, apply_dihedral  # noqa: E402

STATS = NormalizationStats(mean=(0.2, 0.3, 0.4, 0.5), std=(0.1, 0.1, 0.1, 0.1))


def test_dihedral_has_8_distinct_transforms_and_keeps_alignment():
    img = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    mask = img[0].copy()  # mask is a copy of channel 0 -> must remain equal after any transform
    seen = set()
    for k in range(8):
        a, m = apply_dihedral(k, img, mask)
        assert np.array_equal(a[0], m)
        seen.add(a.tobytes())
    assert len(seen) == 8


def test_dataset_item_shapes_and_validity(fake_root):
    records = build_patch_index(DatasetLayout.discover(fake_root, "train"))
    ds = CloudPatchDataset(records, STATS)
    x, y, v = ds[0]  # margin patch
    assert x.shape == (4, 384, 384) and x.dtype == torch.float32
    assert y.shape == (1, 384, 384) and v.shape == (1, 384, 384)
    assert v[0, :100].sum() == 0 and v[0, 100:].min() == 1
    assert (x[:, :100] == 0).all()  # nodata is neutral in the input
    assert set(y.unique().tolist()) <= {0.0, 1.0}


def test_augmentation_keeps_image_mask_valid_aligned(fake_root):
    records = build_patch_index(DatasetLayout.discover(fake_root, "train"))
    plain, aug = CloudPatchDataset(records, STATS), CloudPatchDataset(records, STATS, augment=True)
    _, y0, v0 = plain[0]
    for seed in range(8):
        torch.manual_seed(seed)
        x, y, v = aug[0]
        assert y.sum() == y0.sum() and v.sum() == v0.sum()
        assert (
            (x[:, :, :] == 0).all(dim=0, keepdim=True) | (v == 1)
        ).all()  # zeros only where invalid or exact-zero


def test_dataset_requires_ground_truth(fake_root):
    records = build_patch_index(DatasetLayout.discover(fake_root, "train"), with_gt=False)
    with pytest.raises(ValueError):
        CloudPatchDataset(records, STATS)
