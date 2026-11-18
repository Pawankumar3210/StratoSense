import numpy as np
import pytest

from ml.compute_stats import main
from ml.config import SplitConfig
from ml.dataset import split_scenes
from ml.preprocessing import NormalizationStats
from ml.tests.conftest import BAND_BASE, SCENES


def test_cli_writes_stats_from_training_scenes_only(fake_root, tmp_path):
    out = tmp_path / "norm.json"
    assert main(["--data-root", str(fake_root), "--out", str(out)]) == 0
    stats = NormalizationStats.load(out)

    train, val = split_scenes(SCENES, SplitConfig())
    prov = stats.provenance
    assert prov["train_scenes"] == train
    assert prov["excluded_validation_scenes"] == val
    assert not set(prov["train_scenes"]) & set(prov["excluded_validation_scenes"])
    assert prov["nonempty_patches_only"] is True and prov["n_patches"] == len(train)

    # Fixture band values are BASE + index*10 + noise, so means must land near BASE/65535.
    for mean, base in zip(stats.mean, BAND_BASE.values(), strict=True):
        assert abs(mean - base / 65535) < 0.01
    assert all(s > 0 for s in stats.std)
    assert np.isfinite(stats.mean).all()


def test_cli_requires_nonempty_list_unless_overridden(tmp_path):
    from ml.tests.conftest import make_fake_dataset

    root = make_fake_dataset(tmp_path)
    (root / "38-Cloud_training" / "training_patches_38-Cloud_nonempty.csv").unlink()
    with pytest.raises(SystemExit, match="nonempty"):
        main(["--data-root", str(root), "--out", str(tmp_path / "n.json")])
    assert main(["--data-root", str(root), "--out", str(tmp_path / "n.json"), "--all-patches"]) == 0
