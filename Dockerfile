# Build: docker build -t policy-pulse .
# Run:   docker run --rm -p 8050:8050 --env-file .env policy-pulse
#
# Connection settings (PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD) are injected at *run* time via
# --env-file or the orchestrator's environment, never baked into the image. Note this image only
# copies `app/` and `config/` — not the whole build context — so a local `.env`, the notebooks and
# the reference datasets cannot end up in a published layer.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app
# UV_NO_CACHE: without it uv keeps every downloaded wheel in its cache *inside the layer*, which
# cost ~370 MB of image for files nothing reads at runtime.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1

# To be set by the CI
ARG BUILD_COMMIT=unknown
ARG BUILD_DATE=unknown
ENV BUILD_COMMIT=${BUILD_COMMIT} \
    BUILD_DATE=${BUILD_DATE}

# Set to True at runtime if we want to show that it's experimental
ENV WARN_EXPERIMENTAL=False

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# The unprivileged user is created *before* the dependencies are installed, and everything is
# copied with --chown, so the virtualenv is owned correctly as it is written. A `chown -R /app`
# after the fact would rewrite every file in the venv into a new layer — ~480 MB of duplication.
#
# logs/ has to exist and be writable: the logger writes there relative to the project root (see
# config/data_sources.yaml).
RUN useradd --create-home appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

# Dependencies in their own layer: changes under app/ don't invalidate it on rebuild.
#
# --no-install-project: the app is imported from /app via PYTHONPATH, not installed as a package
# (src/policy_pulse is a stub), so there is nothing to build from this project itself.
# psycopg's `binary` extra carries its own libpq, so no apt packages are needed here.
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

ENV PATH="/app/.venv/bin:${PATH}"

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser config ./config

EXPOSE 8050

# --preload loads the data once in the master process before forking workers, which then share it
# copy-on-write. The data load opens and closes its own Postgres connection inside the master, so
# no connection is inherited across the fork.
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "2", "--timeout", "120", "--preload", "app.__main__:server"]
