"""Periodic in-process reload of the app's data.

The ingestion jobs refresh Postgres on their own schedule. Rather than waiting for a deploy to
pick that up, each app process polls a cheap marker — `MAX(completed_at)` over successful
`ingestion_runs` — and rebuilds its in-memory copy when the marker moves.

Two things about the mechanics are easy to get wrong:

**Threads do not survive `fork`.** Under gunicorn's `--preload` the app is imported once in the
master, which then forks workers, and only the forking thread exists in the child. A poller
started at import time would run in the master (which serves nothing) and be absent from every
worker — the loop would appear to be configured and simply never fire. `start()` is therefore
called lazily, per worker, on the first request, and records the pid it started under so an
inherited flag can't fool it.

**Each worker reloads independently.** They don't coordinate, so `--preload`'s copy-on-write
sharing ends at the first reload and each worker holds a private copy from then on. That is the
accepted trade-off for not needing a coordinator; the jitter below keeps them from rebuilding in
lockstep.
"""

import logging
import os
import random
import threading
from datetime import datetime
from typing import Callable, Optional

from . import db

DEFAULT_INTERVAL_SECONDS = 1800  # 30 min; the ingestion jobs run twice a day
JITTER_FRACTION = 0.1

logger = logging.getLogger("UNResolutionAnalyzer")

_started_in_pid: Optional[int] = None
_lock = threading.Lock()


def interval_from_env() -> int:
    """Poll interval in seconds from `RELOAD_INTERVAL_SECONDS`; 0 disables reloading."""
    raw = os.environ.get("RELOAD_INTERVAL_SECONDS")
    if raw is None or not raw.strip():
        return DEFAULT_INTERVAL_SECONDS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "RELOAD_INTERVAL_SECONDS=%r is not a number; falling back to %ss",
            raw,
            DEFAULT_INTERVAL_SECONDS,
        )
        return DEFAULT_INTERVAL_SECONDS
    return max(value, 0)


def read_marker() -> Optional[datetime]:
    """Read the live freshness marker on a connection of its own."""
    engine = db.create_engine()
    try:
        with engine.connect() as conn:
            return db.read_success_marker(conn)
    finally:
        engine.dispose()


def is_stale(loaded_marker: Optional[datetime], live_marker: Optional[datetime]) -> bool:
    """Whether a rebuild is due.

    Only a strictly newer marker triggers one. An equal marker means nothing has been ingested
    since this data was read. A live marker of None means nothing has *ever* been ingested
    successfully, which is not a reason to throw away data we already hold.
    """
    if live_marker is None:
        return False
    if loaded_marker is None:
        return True
    return live_marker > loaded_marker

def start(reload_now: Callable[[], None], interval_s: Optional[int] = None) -> bool:
    """Start this process's poller if it isn't already running. Returns whether it started.

    `reload_now` is expected to check staleness itself and to be safe to call repeatedly; it is
    invoked on a daemon thread, so it must not touch anything that requires the main thread.
    """
    global _started_in_pid

    if interval_s is None:
        interval_s = interval_from_env()

    if interval_s == 0:
        logger.info("Periodic reload disabled (RELOAD_INTERVAL_SECONDS=0)")
        return False

    with _lock:
        pid = os.getpid()
        if _started_in_pid == pid:
            return False
        _started_in_pid = pid

    def loop() -> None:
        while True:
            # Jitter so that N workers, all started by the same burst of traffic, don't rebuild
            # simultaneously — each rebuild costs ~25s of CPU and a full read of the tables.
            delay = interval_s * (1 + random.uniform(-JITTER_FRACTION, JITTER_FRACTION))
            threading.Event().wait(delay)
            try:
                reload_now()
            except Exception:
                # Keep serving the data we have and try again next tick. A reload failing is not
                # a reason to take the process down.
                logger.exception("Periodic reload failed; keeping the current data")

    threading.Thread(target=loop, name="data-reloader", daemon=True).start()
    logger.info("Periodic reload every ~%ss (pid %s)", interval_s, os.getpid())
    return True
