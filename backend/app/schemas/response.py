"""Pydantic response models for the public API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


class ReadyResponse(BaseModel):
    """Readiness means a *real trained model is loaded*; nothing is ever simulated."""

    model_config = ConfigDict(protected_namespaces=())

    ready: bool
    model_loaded: bool
    model_version: str | None = None
    device: str | None = None
    reason: str | None = None


class ErrorResponse(BaseModel):
    detail: str
