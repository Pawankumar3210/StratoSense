"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import api_router
from app.config import Settings, get_settings

logger = logging.getLogger("stratosense")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="StratoSense API",
        version=__version__,
        description="Deep-learning cloud segmentation for four-band Landsat 8 imagery.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Log details server-side; never expose tracebacks to clients.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    # Routes resolve settings through this dependency, so the settings passed to the
    # factory are the ones every route sees (also makes tests straightforward).
    app.dependency_overrides[get_settings] = lambda: settings
    app.include_router(api_router)
    return app


app = create_app()
