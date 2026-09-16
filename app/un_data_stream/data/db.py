"""
Postgres access for the app's data load.

The app is a **read-only** consumer of the storage layer: the `undl-ingest` repo's jobs own
fetching from UNDL and writing these tables, the app only ever selects from them.
"""

import logging
import os

import pandas as pd
import sqlalchemy as sa
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import Connection, Engine
from sqlalchemy.pool import NullPool

REQUIRED_ENV_VARS = ("PGHOST", "PGPORT", "PGDATABASE", "PGUSER")


def load_env() -> None:
    """
    Load `.env` from the repo root, regardless of the process's cwd, then check that the
    connection settings are actually present.
    """
    load_dotenv(find_dotenv(usecwd=True))

    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if "PGPASSWORD" not in os.environ:
        missing.append("PGPASSWORD")
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(sorted(missing))}. "
            "Set them in .env (see .env.example) or pass them in with `docker run --env-file`."
        )

    port = os.environ["PGPORT"]
    if not port.isdigit():
        raise RuntimeError(f"PGPORT must be a number, got {port!r}.")


def _database_url() -> sa.URL:
    return sa.URL.create(
        "postgresql+psycopg",
        username=os.environ.get("PGUSER"),
        password=os.environ.get("PGPASSWORD"),
        host=os.environ.get("PGHOST"),
        port=int(os.environ["PGPORT"]) if os.environ.get("PGPORT") else None,
        database=os.environ.get("PGDATABASE"),
    )


def create_engine() -> Engine:
    """
    Engine for a one-shot bulk read.

    `NullPool` (no connection pooling) is deliberate: the app loads its data at import time, which
    under gunicorn's `--preload` happens in the master process *before* it forks its workers. A
    pooled connection left open at fork time would be inherited — the same socket used from
    several processes at once, which is undefined behaviour. With `NullPool` every `connect()`
    opens a fresh connection and closing it really closes it.
    """
    return sa.create_engine(_database_url(), poolclass=NullPool)


def connect_or_explain(engine: Engine, logger: logging.Logger | None = None):
    """Connect, turning an unreachable database into a message that says what to fix.

    The app loads its data at import time, so a connection failure here is fatal — and under
    gunicorn it surfaces in the container log with the worker dying, where SQLAlchemy's own
    traceback buries the useful part (which host, which database, which user) under driver
    internals. This states it plainly instead, without the password.
    """
    try:
        return engine.connect()
    except Exception as exc:
        url = engine.url
        detail = (
            f"Could not connect to Postgres at {url.host}:{url.port} "
            f"database={url.database!r} user={url.username!r}: {type(exc).__name__}: {exc}"
        )
        hint = (
            "Check the PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD variables (locally: .env, see "
            ".env.example; in Docker: --env-file or the orchestrator's environment) and that the "
            "database is reachable from this host."
        )
        if logger is not None:
            logger.error(detail)
        raise RuntimeError(f"{detail}\n{hint}") from exc


def read_table(conn: Connection, table_name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Read one table into a DataFrame, optionally only `columns`.
    """
    return pd.read_sql_table(table_name, conn, columns=columns)
