"""ASGI application factory. Importing this module never fetches UN data."""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings
from .provider import build_service
from .routes import router
from .service import InvalidSelection

logger = logging.getLogger("policy_pulse.api")


def create_app(*, service=None, settings=None, service_factory=build_service):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        async def initialize():
            try:
                app.state.service = await asyncio.to_thread(service_factory, settings)
                app.state.data_status = "ready"
            except Exception:
                logger.exception("Dataset initialization failed")
                app.state.data_status = "error"

        task = asyncio.create_task(initialize()) if app.state.service is None else None
        yield
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(
        title="Policy Pulse API",
        version="1.0.0",
        description="Read-only access to adopted UN General Assembly resolutions and the existing Policy Pulse methodology. Scores are 0–1; unavailable observations are null.",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.data_status = "ready" if service is not None else "loading"
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET"],
            allow_headers=["Accept", "Content-Type"],
            expose_headers=["X-Request-ID"],
        )

    @app.middleware("http")
    async def request_context(request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.monotonic()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        logger.info(
            "%s %s %s %.3fs request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            time.monotonic() - started,
            request_id,
        )
        return response

    @app.exception_handler(InvalidSelection)
    async def selection_error(request, exc):
        return JSONResponse(
            status_code=422, content={"detail": {"code": "invalid_selection", "message": str(exc)}}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "invalid_request",
                    "message": "; ".join(e["msg"] for e in exc.errors()),
                }
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        logger.exception("Unhandled request error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            headers={"X-Request-ID": getattr(request.state, "request_id", "unknown")},
            content={
                "detail": {
                    "code": "internal_error",
                    "message": "The request could not be completed.",
                }
            },
        )

    @app.get("/health/live", tags=["Health"])
    def live():
        return {"status": "ok"}

    @app.get("/health/ready", tags=["Health"])
    def ready(request: Request):
        return JSONResponse(
            status_code=200 if request.app.state.service is not None else 503,
            content={"status": request.app.state.data_status},
        )

    app.include_router(router)
    return app


app = create_app()
