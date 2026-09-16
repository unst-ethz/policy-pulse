"""
Postgres access for the app's data load.

The app is a **read-only** consumer of the storage layer: the `undl-ingest` repo's jobs own
fetching from UNDL and writing these tables, the app only ever selects from them.
"""

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


def read_table(
    conn: Connection, table_name: str, columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Read one table into a DataFrame, optionally only `columns`.
    """
    return pd.read_sql_table(table_name, conn, columns=columns)
