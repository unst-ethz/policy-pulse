"""
MongoDB access for the app's data load.

The app is a **read-only** consumer of the storage layer: the `undl-ingest` repo's jobs own
fetching from UNDL and writing these collections, the app only ever reads from them.
"""

import logging
import os
from datetime import datetime

from dotenv import find_dotenv, load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

REQUIRED_ENV_VARS = ("MONGODB_URI", "MONGODB_DATABASE")

# Collection names, as undl-ingest's `common/collections.py` defines them.
RESOLUTIONS = "resolutions"
SUBJECTS = "subjects"
MEMBER_STATES = "member_states"
INGESTION_RUNS = "ingestion_runs"


def load_env() -> None:
    """
    Load `.env` from the repo root, regardless of the process's cwd, then check that the
    connection settings are actually present.
    """
    load_dotenv(find_dotenv(usecwd=True))

    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(sorted(missing))}. "
            "Set them in .env (see .env.example) or pass them in with `docker run --env-file`."
        )

    # MongoClient takes a bare string as a hostname, so a pasted password alone would be sent to
    # DNS and end up in the traceback; never print the value.
    if not os.environ["MONGODB_URI"].startswith(("mongodb://", "mongodb+srv://")):
        raise RuntimeError("MONGODB_URI must be a full mongodb:// or mongodb+srv:// URI.")


def host() -> str:
    """The host part of `MONGODB_URI`, without credentials, for messages."""
    return os.environ["MONGODB_URI"].split("://", 1)[1].split("/", 1)[0].rsplit("@", 1)[-1]


def connect() -> MongoClient:
    """
    A new client for one load; the caller closes it (`with db.connect() as client:`).

    Never a shared, long-lived client: the app loads its data at import time, which under
    gunicorn's `--preload` happens in the master process *before* it forks its workers, and a
    PyMongo client must not cross a fork (its sockets and monitor threads would be shared with,
    or missing from, the children). A client per load, closed afterwards, leaves nothing to
    inherit.
    """
    return MongoClient(os.environ["MONGODB_URI"], tz_aware=True, appname="policy-pulse")


def database(client: MongoClient) -> Database:
    return client[os.environ["MONGODB_DATABASE"]]


def connect_or_explain(logger: logging.Logger | None = None) -> MongoClient:
    """Connect, turning an unreachable database into a message that says what to fix.

    The app loads its data at import time, so a connection failure here is fatal — and under
    gunicorn it surfaces in the container log with the worker dying, where PyMongo's own
    traceback buries the useful part (which host, which database) under driver internals. This
    states it plainly instead, without the credentials.
    """
    client = connect()
    try:
        # MongoClient connects lazily; ping so a failure surfaces here and not mid-load.
        client.admin.command("ping")
    except Exception as exc:
        client.close()
        detail = (
            f"Could not connect to MongoDB at {host()} "
            f"database={os.environ.get('MONGODB_DATABASE')!r}: {type(exc).__name__}: {exc}"
        )
        hint = (
            "Check the MONGODB_URI/MONGODB_DATABASE variables (locally: .env, see .env.example; "
            "in Docker: --env-file or the orchestrator's environment) and that the database is "
            "reachable from this host."
        )
        if logger is not None:
            logger.error(detail)
        raise RuntimeError(f"{detail}\n{hint}") from exc
    return client


def read_success_marker(db: Database) -> datetime | None:
    """Timestamp of the most recent successful ingestion run, or None if there has been none.

    The app's freshness marker. Deliberately not derived from document-level timestamps: those
    can reflect a run that failed halfway, whereas this only advances once a full batch completed.
    """
    run = db[INGESTION_RUNS].find_one(
        {"status": "success"}, {"completed_at": 1}, sort=[("completed_at", -1)]
    )
    return run.get("completed_at") if run else None
