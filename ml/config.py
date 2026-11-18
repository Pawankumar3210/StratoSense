"""Central, dependency-free configuration for the StratoSense ML pipeline.

Everything that defines the *contract* between data, model and backend lives here:
band order, patch size, raw value range and the default split / training settings.
Nothing in this module imports torch, so the backend can read it cheaply.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Input contract (do not reorder: the trained model depends on this order) -------------
BAND_ORDER: tuple[str, ...] = ("red", "green", "blue", "nir")
LANDSAT8_BAND_NUMBERS: dict[str, int] = {"red": 4, "green": 3, "blue": 2, "nir": 5}
NUM_BANDS = len(BAND_ORDER)
PATCH_SIZE = 384

# 38-Cloud images are 16-bit; the dataset authors scale by the 16-bit maximum.
RAW_DTYPE = "uint16"
RAW_MAX_VALUE = 65535.0
# Landsat 8 Level-1 fill value: the black margin around each scene is all zeros.
NODATA_VALUE = 0

# Ground-truth masks are stored as 0 (clear) / 255 (cloud).
MASK_CLEAR_VALUE = 0
MASK_CLOUD_VALUE = 255

# Official dataset facts (from the dataset README); used for sanity checks only.
OFFICIAL_TRAIN_PATCHES = 8400
OFFICIAL_TEST_PATCHES = 9201
OFFICIAL_TEST_SCENES = 20

DATA_ROOT_ENV = "STRATOSENSE_DATA_ROOT"
DEFAULT_THRESHOLD = 0.5  # replaced by the validation-selected value after training


def data_root_from_env() -> Path:
    """Return the dataset root from ``STRATOSENSE_DATA_ROOT`` or raise a clear error."""
    value = os.environ.get(DATA_ROOT_ENV)
    if not value:
        raise RuntimeError(
            f"Set {DATA_ROOT_ENV} to the folder that contains the extracted 38-Cloud "
            "training and test directories (see docs/dataset.md)."
        )
    return Path(value).expanduser()


@dataclass(frozen=True)
class SplitConfig:
    """Deterministic scene-level train/validation split (never split by patch)."""

    val_fraction: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class TrainConfig:
    """Initial training settings from the project spec; tuned later against validation."""

    in_channels: int = NUM_BANDS
    out_channels: int = 1
    encoder_filters: tuple[int, ...] = (32, 64, 128, 256)
    bottleneck_filters: int = 512
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 20
    early_stopping_patience: int = 5
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    use_mixed_precision: bool = True
    # Whether to drop patches that are mostly zero-valued margin. To be confirmed after
    # data exploration (the dataset README suggests >80% informative pixels).
    use_nonempty_patches_only: bool = True
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)
