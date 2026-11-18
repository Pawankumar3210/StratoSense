"""Runtime configuration, read from environment variables (see ``.env.example``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml.config import REPO_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    app_env: Literal["development", "production", "test"] = "development"
    model_path: Path = Path("artifacts/checkpoints/stratosense_unet.pth")
    model_version: str = "1.0.0"
    model_device: Literal["auto", "cpu", "cuda"] = "auto"
    model_artifact_url: str = ""
    model_sha256: str = ""
    max_upload_mb: int = 50
    cors_origins: str = "http://localhost:5173"  # comma-separated exact origins

    @field_validator("max_upload_mb")
    @classmethod
    def _positive_upload_limit(cls, value: int) -> int:
        if value < 1:
            raise ValueError("MAX_UPLOAD_MB must be at least 1")
        return value

    @model_validator(mode="after")
    def _validate_cors_for_production(self) -> Settings:
        origins = self.cors_origin_list
        if self.app_env == "production":
            if not origins:
                raise ValueError("CORS_ORIGINS must list the frontend origin in production")
            if "*" in origins:
                raise ValueError("CORS_ORIGINS must not contain '*' in production")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.cors_origins.split(",") if o.strip()]

    @property
    def resolved_model_path(self) -> Path:
        """Relative ``MODEL_PATH`` values are resolved against the repository root."""
        return self.model_path if self.model_path.is_absolute() else REPO_ROOT / self.model_path

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
