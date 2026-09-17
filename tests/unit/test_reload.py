"""Unit tests for the periodic-reload machinery.

No database and no app import: the staleness decision, the interval config and the atomicity of
the snapshot swap are all pure logic.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.un_data_stream.analysis.snapshot import DataSnapshot, empty_snapshot
from app.un_data_stream.data import reloader

NOW = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
EARLIER = NOW - timedelta(hours=12)


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


def test_newer_marker_is_stale():
    assert reloader.is_stale(EARLIER, NOW)


def test_equal_marker_is_not_stale():
    """Nothing has been ingested since this data was read — rebuilding would be pure cost."""
    assert not reloader.is_stale(NOW, NOW)


def test_older_marker_is_not_stale():
    """Shouldn't happen, but a clock skew or a restored backup must not cause a reload loop."""
    assert not reloader.is_stale(NOW, EARLIER)


def test_no_live_marker_is_not_stale():
    """No successful ingestion run has *ever* completed.

    That is not a reason to throw away data we already hold — most likely the marker query ran
    against an empty or freshly-created `ingestion_runs`.
    """
    assert not reloader.is_stale(NOW, None)
    assert not reloader.is_stale(None, None)


def test_data_loaded_without_a_marker_is_stale_once_one_exists():
    assert reloader.is_stale(None, NOW)


# ---------------------------------------------------------------------------
# Interval configuration
# ---------------------------------------------------------------------------


def test_interval_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("RELOAD_INTERVAL_SECONDS", raising=False)
    assert reloader.interval_from_env() == reloader.DEFAULT_INTERVAL_SECONDS


@pytest.mark.parametrize("raw, expected", [("60", 60), ("0", 0), ("  120 ", 120), ("-5", 0)])
def test_interval_parses_env(monkeypatch, raw, expected):
    monkeypatch.setenv("RELOAD_INTERVAL_SECONDS", raw)
    assert reloader.interval_from_env() == expected


def test_interval_falls_back_on_garbage(monkeypatch):
    monkeypatch.setenv("RELOAD_INTERVAL_SECONDS", "half an hour")
    assert reloader.interval_from_env() == reloader.DEFAULT_INTERVAL_SECONDS


def test_zero_interval_does_not_start_a_thread(monkeypatch):
    monkeypatch.setattr(reloader, "_started_in_pid", None)
    called = []
    assert reloader.start(lambda: called.append(1), interval_s=0) is False
    assert not called


def test_start_is_once_per_process(monkeypatch):
    """A second call must not add a second poller — before_request calls this on every request."""
    monkeypatch.setattr(reloader, "_started_in_pid", None)
    threads = []
    monkeypatch.setattr(
        reloader.threading, "Thread", lambda **kw: threads.append(kw) or _NullThread()
    )
    assert reloader.start(lambda: None, interval_s=900) is True
    assert reloader.start(lambda: None, interval_s=900) is False
    assert len(threads) == 1
    assert threads[0]["daemon"] is True, "must not hold the process open at shutdown"


class _NullThread:
    def start(self):
        pass


# ---------------------------------------------------------------------------
# Snapshot swap
# ---------------------------------------------------------------------------


def _snapshot(n_resolutions: int, marker: datetime) -> DataSnapshot:
    ids = [str(i) for i in range(n_resolutions)]
    countries = ["AAA", "BBB"]
    shape = (n_resolutions, len(countries))
    return DataSnapshot(
        resolution_table=pd.DataFrame({"undl_id": ids, "date": ["2020-01-01"] * n_resolutions}),
        resolution_subject_table=pd.DataFrame(columns=["undl_id", "subject_id"]),
        subject_table=pd.DataFrame(columns=["subject_id", "label_en", "node_type"]),
        closure_table=pd.DataFrame(columns=["ancestor_id", "descendant_id", "depth"]),
        country_columns=countries,
        multilateral_scores=np.zeros(shape, dtype=np.float32),
        yes=np.ones(shape, dtype=bool),
        no=np.zeros(shape, dtype=bool),
        abstained=np.zeros(shape, dtype=bool),
        voted=np.ones(shape, dtype=bool),
        row_index={rid: i for i, rid in enumerate(ids)},
        loaded_at=marker,
        source_marker=marker,
    )


class _FakeRepo:
    """Duck-types what the engine needs: get_data(), logger, source_marker."""

    def __init__(self, snapshot, logger):
        self._snapshot = snapshot
        self.logger = logger
        self.source_marker = snapshot.source_marker

    def get_data(self):
        s = self._snapshot
        return {
            "resolution": s.resolution_table,
            "resolution_subject": s.resolution_subject_table,
            "subject": s.subject_table,
            "closure": s.closure_table,
            "broader": pd.DataFrame(),
            "country_columns": s.country_columns,
            "member_states": pd.DataFrame(),
            "multilateral_scores": s.multilateral_scores,
            "vote_bool_arrays": (s.yes, s.no, s.abstained, s.voted),
        }


@pytest.fixture
def engine(caplog):
    import logging

    from app.un_data_stream.analysis.query_engine import ResolutionQueryEngine

    logger = logging.getLogger("reload_test")
    logger.addHandler(logging.NullHandler())
    return ResolutionQueryEngine(repo=_FakeRepo(_snapshot(3, EARLIER), logger))


def test_swap_replaces_every_field_together(engine):
    """The whole point of the snapshot: no reader can see new tables with old arrays."""
    before = engine.snapshot
    assert len(engine.resolution_table) == 3

    previous = engine.swap(_snapshot(5, NOW))

    assert previous is before
    assert len(engine.resolution_table) == 5
    assert engine.snapshot.source_marker == NOW
    assert engine.snapshot.multilateral_scores.shape[0] == 5
    assert len(engine.snapshot.row_index) == 5


def test_a_query_in_flight_keeps_its_snapshot(engine):
    """A method takes one snapshot reference at entry, so a swap mid-query can't split it."""
    snap = engine.snapshot
    engine.swap(_snapshot(5, NOW))

    # the reference the in-flight query holds is unchanged and still self-consistent
    assert len(snap.resolution_table) == 3
    assert snap.multilateral_scores.shape == (3, 2)
    assert len(snap.row_index) == 3


def test_public_properties_follow_the_swap(engine):
    """Features read engine.resolution_table etc. directly; they must see current data."""
    engine.swap(_snapshot(5, NOW))

    assert len(engine.resolution_table) == 5
    assert engine.country_columns == ["AAA", "BBB"]
    assert engine.get_available_countries() == ["AAA", "BBB"]


def test_queries_work_against_an_empty_snapshot(engine):
    """A degenerate load must return empty results, not raise."""
    engine.swap(empty_snapshot())

    assert engine.query_resolutions().empty
    assert engine.query_agreement_between_countries("AAA").empty
    assert engine.query_multilateral_stats().empty
