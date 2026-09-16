import functools

import numpy as np
import pandas as pd
import pytest


@functools.lru_cache(maxsize=1)
def postgres_unavailable() -> str | None:
    """Return why Postgres can't be reached, or None if it can.

    The app reads every table from Postgres at import time, so anything touching `app.data` needs
    a live database. Checked once per session, with a real connection rather than a port probe —
    wrong credentials or a missing database should skip these tests just as cleanly as a stopped
    server, and say which it was.
    """
    try:
        import sqlalchemy as sa

        from app.un_data_stream.data import db

        db.load_env()
        engine = db.create_engine()
        try:
            with engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
        finally:
            engine.dispose()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_postgres: test requires a reachable Postgres (auto-skipped without one)"
    )


def pytest_collection_modifyitems(config, items):
    reason = postgres_unavailable()
    if reason is None:
        return

    skip_postgres = pytest.mark.skip(reason=f"Postgres not reachable ({reason})")
    for item in items:
        if "needs_postgres" in item.keywords:
            item.add_marker(skip_postgres)


@pytest.fixture(scope="module")
def random_un_votes_dataframe():
    """Generates a synthetic, pseudo-random DataFrame mimicking UN voting records."""
    np.random.seed(1946)

    n_countries = 193
    n_resolutions = 1_000

    countries = [f"Member_State_{i}" for i in range(n_countries)]
    undl_ids = [f"A/RES/78/{i}" for i in range(n_resolutions)]

    data = {
        'undl_id': undl_ids,
        'date': pd.date_range(start='2024-01-01', periods=n_resolutions),
        'title': [f"Resolution Topic {i}" for i in range(n_resolutions)]
    }

    # Fill country columns with random Y, N, A, or NaN (Missing)
    for country in countries:
        data[country] = np.random.choice(['Y', 'N', 'A', np.nan], n_resolutions, p=[0.4, 0.2, 0.3, 0.1])

    return pd.DataFrame(data)


@pytest.fixture
def data_processor(caplog):
    """
    Provides an initialized DataProcessor instance.
    The 'caplog' fixture is included to allow testing of log output.
    """
    import logging
    from app.un_data_stream.data.processor import DataProcessor

    # Using a dedicated test logger to avoid polluting main logs
    logger = logging.getLogger("un_data_test")
    config = {
        "env": "test",
        "threshold": 0.5
    }
    return DataProcessor(config, logger)
