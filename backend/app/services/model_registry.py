"""Tracks whether a trained model is available for inference.

This build only *reports* model status. Checkpoint loading and inference arrive with the
trained model; until then the service is honestly reported as not ready. There is no
fallback or placeholder model.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings


@dataclass(frozen=True)
class ModelStatus:
    ready: bool
    model_loaded: bool
    reason: str | None = None
    model_version: str | None = None
    device: str | None = None


def inspect_model(settings: Settings) -> ModelStatus:
    path = settings.resolved_model_path
    if not path.is_file():
        return ModelStatus(
            ready=False,
            model_loaded=False,
            reason=(
                f"No model checkpoint found at '{settings.model_path}'. Train the model and "
                "provide the artifact (see docs/deployment.md)."
            ),
        )
    return ModelStatus(
        ready=False,
        model_loaded=False,
        reason=(
            "A checkpoint file exists, but the inference loader is not implemented in this build."
        ),
    )
