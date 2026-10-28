"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from sqlalchemy import text

from wafermap import __version__
from wafermap.api.deps import SessionDep
from wafermap.api.routers import lots, predict, stats, wafers
from wafermap.api.schemas import Health
from wafermap.config import Settings, get_settings
from wafermap.ml.predict import ModelRegistry

log = logging.getLogger(__name__)

DESCRIPTION = """
Wafer-map defect analytics on the public WM-811K dataset.

* **Lots / wafers** - browse production lots and drill down to die-level wafer maps.
* **Analytics** - yield trend along the lot sequence and defect-pattern distribution.
* **Prediction** - classify a new wafer map (CNN via ONNX Runtime, or the LightGBM baseline).
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.registry = ModelRegistry(settings.model_dir)
        log.info("models loaded: %s", app.state.registry.names())
        yield

    app = FastAPI(
        title="Wafer-map Defect Analytics API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    api = APIRouter(prefix="/api/v1")

    @api.get("/health", response_model=Health, tags=["meta"], summary="Liveness + dependencies")
    def health(session: SessionDep) -> Health:
        try:
            session.execute(text("SELECT 1"))
            db_ok = True
        except Exception:
            db_ok = False
        return Health(
            status="ok", database=db_ok, models=app.state.registry.names(), version=__version__
        )

    for r in (lots.router, wafers.router, stats.router, predict.router):
        api.include_router(r)
    app.include_router(api)
    return app


app = create_app()
