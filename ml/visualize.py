"""Display-only helpers (NumPy). Nothing here feeds the model."""

from __future__ import annotations

import numpy as np

from ml.config import BAND_ORDER


def rgb_composite(
    raw: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    low: float = 2.0,
    high: float = 98.0,
) -> np.ndarray:
    """Return a uint8 ``(H, W, 3)`` natural-colour preview of a ``(4, H, W)`` array.

    A per-image percentile stretch makes dark 16-bit data viewable. This is for *display
    only*; model inputs use the persisted training normalization instead. Invalid pixels
    are rendered black and excluded from the stretch.
    """
    rgb = raw[[BAND_ORDER.index("red"), BAND_ORDER.index("green"), BAND_ORDER.index("blue")]]
    rgb = rgb.astype(np.float32)
    if valid is None:
        valid = np.ones(rgb.shape[1:], dtype=bool)
    out = np.zeros((*rgb.shape[1:], 3), dtype=np.uint8)
    if not valid.any():
        return out
    for i in range(3):
        band = rgb[i]
        lo, hi = np.percentile(band[valid], [low, high])
        scaled = np.zeros_like(band) if hi <= lo else np.clip((band - lo) / (hi - lo), 0.0, 1.0)
        out[..., i] = (scaled * 255.0).round().astype(np.uint8)
    out[~valid] = 0
    return out
