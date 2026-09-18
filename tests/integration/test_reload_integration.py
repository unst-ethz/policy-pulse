"""The periodic reload, end to end against Postgres.

Exercises the decision path (`reload_if_stale`) and the marker read against the real
`ingestion_runs` table. Deliberately does *not* leave a synthetic run row behind: the marker it
inserts is rolled back, because a stray 'success' row would make every other worker — and the
next developer's app — think there is new data.
"""

from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa

pytestmark = pytest.mark.needs_postgres


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

    Inserted inside a transaction that is rolled back, so the table is left untouched.
    """
    from app.un_data_stream.data import db, reloader

    engine = db.create_engine()
    try:
        with engine.connect() as conn:
            before = db.read_success_marker(conn)
            future = datetime.now(timezone.utc) + timedelta(days=365)
            # Raw SQL on purpose: the app's db module has no write helpers (it is a read-only
            # consumer), and this test has no business borrowing the ingest repo's.
            insert_run = sa.text(
                "INSERT INTO ingestion_runs (source_dataset, started_at, completed_at, status) "
                "VALUES ('test_reload', :ts, :ts, :status)"
            )
            for status in ("failed", "running", "partial"):
                conn.execute(insert_run, {"ts": future, "status": status})
                assert db.read_success_marker(conn) == before, (
                    f"a '{status}' run must not advance the marker"
                )
            conn.rollback()

        # and the rollback really left nothing behind
        with engine.connect() as conn:
            assert db.read_success_marker(conn) == before
    finally:
        engine.dispose()

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

    Three features capture `data.query_engine` when their callbacks are registered
    (trends_page.py -> agreement_choropleth / agreement_by_subject / multilateral_scatter). If a
    reload replaced the engine *object*, those three would serve the pre-reload data forever
    while every other tab moved on — same app, different numbers, no error anywhere. So a reload
    must swap the snapshot inside the existing instance and never rebind the instance itself.
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


def test_reload_clears_every_cache_derived_from_the_repository(data, monkeypatch):
    """Rebuilding the snapshot is not enough if features memoised the old one.

    Each of these caches its own view of the repository, and none is keyed on anything that
    changes when data is reloaded — so without an explicit clear they serve pre-reload numbers
    for the life of the process, on the landing page and the trends tab where new resolutions are
    exactly what a reader is looking for.
    """
    import sys

    # `app.__main__` builds the Dash instance and imports every page/feature module — the state a
    # reload actually happens in. Only modules already in sys.modules get invalidated.
    import app.__main__  # noqa: F401
    from app.un_data_stream.data import reloader

    wordcloud = sys.modules["app.features.wordcloud_interactive"]
    recent = sys.modules["app.features.recent_resolutions_panel"]
    stats = sys.modules["app.features.general_stats_panel"]
    trends = sys.modules["app.pages.trends_page"]

    # Populate every cache.
    wordcloud._init_wc_data()
    recent._get_recent_resolutions_cached()
    stats._get_stats_component_cached()
    trends._calculate_data_uncached("CHE", ("USA",))
    assert wordcloud._initialized
    assert recent._get_recent_resolutions_cached.cache_info().currsize == 1
    assert stats._get_stats_component_cached.cache_info().currsize == 1
    assert trends._calculate_data_uncached.cache_info().currsize == 1

    monkeypatch.setattr(
        reloader, "read_marker", lambda: datetime.now(timezone.utc) + timedelta(seconds=1)
    )
    assert data.reload_if_stale() is True

    assert not wordcloud._initialized
    assert recent._get_recent_resolutions_cached.cache_info().currsize == 0
    assert stats._get_stats_component_cached.cache_info().currsize == 0
    assert trends._calculate_data_uncached.cache_info().currsize == 0

    # Clearing the flag is only half of it: every read of the word-cloud indices has to route
    # through the accessors, or it keeps returning the stale dict the flag no longer describes.
    assert wordcloud._word_undlid_map("default")
    assert wordcloud._initialized, "reading the index must rebuild it"


def test_invalidation_skips_modules_the_process_never_imported(data):
    """A worker that only ever served the landing page has no trends_page in sys.modules.

    Importing one here to invalidate it would run `register_page()` as an import side effect,
    which raises outside a live Dash app — so absent modules must simply be skipped.
    """
    import sys

    monkeypatched = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.startswith("app.pages.") or name.startswith("app.features.")
    }
    try:
        data._invalidate_feature_caches()  # must not raise
    finally:
        sys.modules.update(monkeypatched)


def test_features_track_rebound_globals_rather_than_pinning_them(data):
    """`_rebuild_derived_state()` rebinds these names; it does not mutate the objects in place.

    So a feature module that did `from ..data import TOP_LEVEL_SUBJECTS` would hold the pre-reload
    set for the life of the process — its tab would keep the old subject list and old labels while
    every other tab moved on, with no error anywhere. Modules must read them off `data`.
    """
    import sys

    import app.__main__  # noqa: F401

    sentinel_subjects = {"__sentinel__"}
    sentinel_labels = {"__sentinel__": "sentinel"}
    real_subjects = data.TOP_LEVEL_SUBJECTS
    real_labels = data.SUBJECT_ID_TO_LABEL_MAP

    data.TOP_LEVEL_SUBJECTS = sentinel_subjects
    data.SUBJECT_ID_TO_LABEL_MAP = sentinel_labels
    try:
        for name, module in list(sys.modules.items()):
            if not name.startswith("app.features.") and not name.startswith("app.pages."):
                continue
            for attr, sentinel in (
                ("TOP_LEVEL_SUBJECTS", sentinel_subjects),
                ("SUBJECT_ID_TO_LABEL_MAP", sentinel_labels),
            ):
                pinned = getattr(module, attr, None)
                assert pinned is None or pinned is sentinel, (
                    f"{name} imported {attr} by value and is pinned to the pre-reload object; "
                    f"use `from .. import data` and read `data.{attr}` at call time"
                )
    finally:
        data.TOP_LEVEL_SUBJECTS = real_subjects
        data.SUBJECT_ID_TO_LABEL_MAP = real_labels
