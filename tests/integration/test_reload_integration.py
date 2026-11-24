"""The periodic reload, end to end against MongoDB.

Exercises the decision path (`reload_if_stale`) and the marker read against the real
`ingestion_runs` collection. Deliberately does *not* leave a synthetic run behind: the runs it
inserts are deleted again, and none is a 'success', because a stray 'success' run would make
every other worker — and the next developer's app — think there is new data.
"""

from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.needs_db


@pytest.fixture(scope="module")
def data():
    from app import data as app_data

    return app_data


def test_snapshot_records_the_marker_it_loaded_with(data):
    """Without this, staleness can't be judged and the app would either never reload or always."""
    snapshot = data.query_engine.snapshot

    assert snapshot.source_marker is not None, "there should be a successful ingestion run"
    assert snapshot.loaded_at >= snapshot.source_marker, "data cannot predate its own marker"
    assert snapshot.describe()


def test_live_marker_matches_what_was_loaded(data):
    """Nothing has been ingested since the app started, so no reload should be due."""
    from app.un_data_stream.data import reloader

    live = reloader.read_marker()

    assert live == data.query_engine.snapshot.source_marker
    assert data.reload_if_stale() is False, "must not rebuild when the marker hasn't moved"


def test_marker_read_uses_only_successful_runs(data):
    """A failed or still-running ingestion must not advance the marker.

    The inserted runs are removed in `finally`, so the collection is left as it was.
    """
    from app.un_data_stream.data import db, reloader

    with db.connect() as client:
        database = db.database(client)
        runs = database[db.INGESTION_RUNS]
        before = db.read_success_marker(database)
        future = datetime.now(timezone.utc) + timedelta(days=365)
        try:
            # Written directly on purpose: the app's db module has no write helpers (it is a
            # read-only consumer), and this test has no business borrowing the ingest repo's.
            for status in ("failed", "running", "partial"):
                runs.insert_one(
                    {
                        "source_dataset": "test_reload",
                        "started_at": future,
                        "completed_at": future,
                        "status": status,
                    }
                )
                assert db.read_success_marker(database) == before, (
                    f"a '{status}' run must not advance the marker"
                )
        finally:
            runs.delete_many({"source_dataset": "test_reload"})

        # and the cleanup really left nothing behind
        assert runs.count_documents({"source_dataset": "test_reload"}) == 0
        assert db.read_success_marker(database) == before

    assert reloader.read_marker() == before


def test_reload_leaves_data_in_place_when_the_rebuild_fails(data, monkeypatch):
    """A reload that blows up must keep serving the previous snapshot, not a partial one."""
    from app.un_data_stream.data import reloader

    snapshot_before = data.query_engine.snapshot

    # pretend the world moved on, then make the rebuild fail
    monkeypatch.setattr(
        reloader, "read_marker", lambda: datetime.now(timezone.utc) + timedelta(days=1)
    )

    def exploding_repo(*args, **kwargs):
        raise RuntimeError("simulated failure mid-rebuild")

    monkeypatch.setattr(data, "DataRepository", exploding_repo)

    with pytest.raises(RuntimeError, match="simulated failure"):
        data.reload_if_stale()

    assert data.query_engine.snapshot is snapshot_before, "old data must still be served"
    assert not data.query_engine.query_resolutions().empty


def test_reload_keeps_the_same_engine_instance(data, monkeypatch):
    """The invariant the whole snapshot design exists for.

    Anything holding `data.query_engine` (e.g. `from app.data import query_engine`) would serve
    the pre-reload data forever if a reload replaced the engine *object* — same app, different
    numbers, no error anywhere. So a reload must swap the snapshot inside the existing instance
    and never rebind the instance itself.
    """
    from app.un_data_stream.data import reloader

    engine_before = data.query_engine
    snapshot_before = engine_before.snapshot

    # Pretend an ingestion run just finished; the rebuild below is real.
    monkeypatch.setattr(
        reloader, "read_marker", lambda: datetime.now(timezone.utc) + timedelta(seconds=1)
    )

    assert data.reload_if_stale() is True

    assert data.query_engine is engine_before, "the engine instance must survive a reload"
    assert data.query_engine.snapshot is not snapshot_before, "its data must have been replaced"
    assert not data.query_engine.query_resolutions().empty
    # derived module state was rebuilt alongside it
    assert data.available_countries
    assert data.TOP_LEVEL_SUBJECTS
