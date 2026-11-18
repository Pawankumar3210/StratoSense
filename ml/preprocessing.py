"""Shared preprocessing for StratoSense (training, evaluation and production inference).

Design rules
------------
* NumPy only. The backend imports this module without needing PyTorch.
* Band order is a fixed contract: ``config.BAND_ORDER`` (red, green, blue, nir).
* Normalization statistics come from *training scenes only*, are persisted to JSON, and
  are the only normalization ever applied afterwards. Uploaded images are never
  normalized with their own statistics.
* Invalid (nodata) pixels are tracked in an explicit validity mask and are neutral
  (zero) in the network input after normalization.
* Unsupported inputs raise ``PreprocessingError`` instead of being silently coerced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ml.config import BAND_ORDER, NODATA_VALUE, NUM_BANDS, RAW_DTYPE, RAW_MAX_VALUE


class PreprocessingError(ValueError):
    """Raised when an input array cannot be preprocessed faithfully."""


# --------------------------------------------------------------------------------------
# Raw array validation and validity mask
# --------------------------------------------------------------------------------------
def validate_raw_array(raw: np.ndarray) -> None:
    """Check that ``raw`` is a ``(4, H, W)`` uint16 array in the model's band order."""
    if raw.ndim != 3 or raw.shape[0] != NUM_BANDS:
        raise PreprocessingError(
            f"Expected a (bands={NUM_BANDS}, H, W) array in order {BAND_ORDER}, got shape "
            f"{tuple(raw.shape)}."
        )
    if raw.shape[1] < 1 or raw.shape[2] < 1:
        raise PreprocessingError(f"Empty spatial dimensions: {tuple(raw.shape)}.")
    if raw.dtype != np.dtype(RAW_DTYPE):
        raise PreprocessingError(
            f"Expected {RAW_DTYPE} digital numbers (Landsat 8 Level-1 style), got "
            f"{raw.dtype}. The model was trained on 16-bit DN values and other value "
            "ranges would give misleading predictions."
        )


def compute_valid_mask(raw: np.ndarray, nodata: int | float = NODATA_VALUE) -> np.ndarray:
    """Return a boolean ``(H, W)`` mask that is True where the pixel carries data.

    A pixel is *invalid* only if **all** bands equal the nodata value (the black scene
    margin). A single zero in one band does not invalidate a pixel.
    """
    if raw.ndim != 3:
        raise PreprocessingError(f"Expected (bands, H, W), got shape {tuple(raw.shape)}.")
    return ~np.all(raw == nodata, axis=0)


# --------------------------------------------------------------------------------------
# Normalization statistics
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class NormalizationStats:
    """Per-band mean/std of ``raw / raw_max_value`` computed on valid training pixels."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    band_order: tuple[str, ...] = BAND_ORDER
    raw_max_value: float = RAW_MAX_VALUE
    nodata_value: float = NODATA_VALUE
    # Free-form provenance (dataset, scene ids used, pixel count, date, ...).
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if tuple(self.band_order) != BAND_ORDER:
            raise PreprocessingError(
                f"Stats band order {tuple(self.band_order)} != required {BAND_ORDER}."
            )
        if len(self.mean) != NUM_BANDS or len(self.std) != NUM_BANDS:
            raise PreprocessingError("mean and std must each have one value per band.")
        if not all(np.isfinite(self.mean)) or not all(np.isfinite(self.std)):
            raise PreprocessingError("mean/std must be finite.")
        if any(s <= 0 for s in self.std):
            raise PreprocessingError("std values must be strictly positive.")
        if self.raw_max_value <= 0:
            raise PreprocessingError("raw_max_value must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": list(self.mean),
            "std": list(self.std),
            "band_order": list(self.band_order),
            "raw_max_value": self.raw_max_value,
            "nodata_value": self.nodata_value,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizationStats:
        try:
            return cls(
                mean=tuple(float(v) for v in data["mean"]),
                std=tuple(float(v) for v in data["std"]),
                band_order=tuple(data["band_order"]),
                raw_max_value=float(data["raw_max_value"]),
                nodata_value=float(data["nodata_value"]),
                provenance=dict(data.get("provenance", {})),
            )
        except KeyError as exc:
            raise PreprocessingError(f"Normalization file is missing key {exc}.") from exc

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> NormalizationStats:
        path = Path(path)
        if not path.is_file():
            raise PreprocessingError(
                f"Normalization file not found: {path}. Compute it from the training "
                "split with `python -m ml.compute_stats` (see docs/dataset.md)."
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class RunningStats:
    """Streaming per-band mean/std over valid pixels (float64 accumulators).

    Feed it unit-scaled data (``raw / raw_max_value``) one patch at a time so the full
    training set never has to fit in memory.
    """

    def __init__(self, num_bands: int = NUM_BANDS) -> None:
        self.num_bands = num_bands
        self.count = 0
        self._sum = np.zeros(num_bands, dtype=np.float64)
        self._sumsq = np.zeros(num_bands, dtype=np.float64)

    def update(self, unit_scaled: np.ndarray, valid: np.ndarray) -> None:
        if unit_scaled.shape[0] != self.num_bands:
            raise PreprocessingError("Band count mismatch in RunningStats.update.")
        if valid.shape != unit_scaled.shape[1:]:
            raise PreprocessingError("Validity mask shape does not match image shape.")
        pixels = unit_scaled[:, valid].astype(np.float64)  # (bands, n_valid)
        self.count += pixels.shape[1]
        self._sum += pixels.sum(axis=1)
        self._sumsq += np.square(pixels).sum(axis=1)

    def finalize(self, provenance: dict[str, Any] | None = None) -> NormalizationStats:
        if self.count == 0:
            raise PreprocessingError("No valid pixels were accumulated; cannot compute stats.")
        mean = self._sum / self.count
        var = np.maximum(self._sumsq / self.count - np.square(mean), 0.0)  # population var
        prov = dict(provenance or {})
        prov["valid_pixel_count"] = int(self.count)
        return NormalizationStats(
            mean=tuple(float(m) for m in mean),
            std=tuple(float(s) for s in np.sqrt(var)),
            provenance=prov,
        )


# --------------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------------
def scale_to_unit(raw: np.ndarray, raw_max_value: float = RAW_MAX_VALUE) -> np.ndarray:
    """Convert validated uint16 DN values to float32 in ``[0, 1]``."""
    validate_raw_array(raw)
    return (raw.astype(np.float32) / np.float32(raw_max_value)).astype(np.float32, copy=False)


def normalize(raw: np.ndarray, stats: NormalizationStats) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a raw ``(4, H, W)`` uint16 array for the network.

    Returns
    -------
    x : float32 ``(4, H, W)``
        ``(raw/max - mean) / std`` with invalid pixels set to 0.
    valid : bool ``(H, W)``
        True where the pixel carries data.
    """
    validate_raw_array(raw)
    valid = compute_valid_mask(raw, stats.nodata_value)
    unit = raw.astype(np.float32) / np.float32(stats.raw_max_value)
    mean = np.asarray(stats.mean, dtype=np.float32)[:, None, None]
    std = np.asarray(stats.std, dtype=np.float32)[:, None, None]
    x = (unit - mean) / std
    x[:, ~valid] = 0.0
    return x.astype(np.float32, copy=False), valid
