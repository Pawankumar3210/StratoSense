from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app import __version__
from app.config import Settings, get_settings
from app.schemas.response import HealthResponse, ReadyResponse
from app.services.model_registry import inspect_model

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok", service="stratosense-api", version=__version__, environment=settings.app_env
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"model": ReadyResponse, "description": "Model not loaded"}},
    summary="Readiness check (true only when a real trained model is loaded)",
)
def ready(settings: Settings = Depends(get_settings)):
    status = inspect_model(settings)
    body = ReadyResponse(
        ready=status.ready,
        model_loaded=status.model_loaded,
        model_version=status.model_version,
        device=status.device,
        reason=status.reason,
    )
    if status.ready:
        return body
    return JSONResponse(status_code=503, content=body.model_dump())
