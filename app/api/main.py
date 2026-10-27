"""The Policy Pulse HTTP API (FastAPI), serving the same data and analyses as the Dash app.

Importing this module loads the data from Postgres (through `app.data`), exactly like the Dash
app: under gunicorn's `--preload` that happens once in the master and the workers share it.

    uv run start-api                                          # dev server on :8000
    gunicorn -k uvicorn_worker.UvicornWorker --preload app.api.main:app   # production

Interactive docs at /api/docs, the OpenAPI schema at /api/openapi.json.
"""

import hashlib
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response

from .. import data
from .routes import analysis, reference, resolutions

API_PREFIX = "/api/v1"

# Origins allowed to call the API from a browser: the Vite dev server by default.
_DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"

# Part of every ETag, so a restart (e.g. a deploy with changed analyses) invalidates what browsers
# cached even when the data did not change. Drawn at import, i.e. once in the gunicorn master
# under `--preload`, so all workers agree on it.
_BUILD_TOKEN = uuid.uuid4().hex


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Started per worker, after the fork: a poller started at import would live in the gunicorn
    # master, which serves no requests. See `app.__main__._ensure_data_reloader`.
    data.start_reloader()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Policy Pulse API",
        summary="Voting data of the UN General Assembly, and analyses over it.",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    origins = os.getenv("POLICY_PULSE_CORS_ORIGINS", _DEFAULT_CORS_ORIGINS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins.split(",") if o.strip()],
        allow_methods=["GET"],
        allow_headers=["*"],
        expose_headers=["ETag", "X-Data-Version"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.middleware("http")
    async def data_version_etag(request: Request, call_next):
        """Answer repeat requests with 304 until the data changes.

        Every response is a pure function of the URL, the loaded data and the running code, so the
        ETag can be derived from those alone — before doing any work, which is what makes a 304
        cheap. The code is identified by `_BUILD_TOKEN`, which changes with every server start.
        """
        if request.method != "GET" or not request.url.path.startswith(API_PREFIX):
            return await call_next(request)

        version = reference.data_version()
        key = f"{_BUILD_TOKEN}|{version}|{request.url.path}?{request.url.query}"
        etag = f'W/"{hashlib.sha1(key.encode()).hexdigest()}"'
        headers = {"ETag": etag, "X-Data-Version": version, "Cache-Control": "no-cache"}

        if etag in request.headers.get("if-none-match", ""):
            return Response(status_code=304, headers=headers)
        response = await call_next(request)
        if response.status_code == 200:
            response.headers.update(headers)
        return response

    @app.get("/api/health", tags=["meta"])
    def health() -> dict[str, str]:
        """Liveness: the process is up and has data loaded."""
        return {"status": "ok"}

    for router in (reference.router, resolutions.router, analysis.router):
        app.include_router(router, prefix=API_PREFIX)
    return app


app = create_app()


def main() -> None:
    """Run the local development server (`uv run start-api`)."""
    import uvicorn

    port = int(os.getenv("POLICY_PULSE_API_PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port)
