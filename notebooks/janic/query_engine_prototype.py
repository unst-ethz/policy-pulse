import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Notebook 2: read back from Postgres, build a real `ResolutionQueryEngine`

    Prototype for `plans/intermediate_storage_layer_plan.md`. Reads `resolution_outcomes` /
    `resolution_votes` back out of the local Postgres populated by
    `notebooks/janic/ingestion_prototype.py`, pivots them into the wide in-memory shape
    `app/un_data_stream` expects, and constructs a genuine, unmodified
    `app.un_data_stream.analysis.query_engine.ResolutionQueryEngine` from it -- then runs real
    queries against it. This is the actual "does this whole approach work" bar for the
    voting-data path: not just "the DataFrame looks right", but "the app's own query engine
    produces sane results from Postgres-sourced data".

    **Unlike notebook 1, this can't use an isolated `uv run --sandbox` PEP 723 environment.**
    It imports `app.un_data_stream` directly (the real production code, deliberately not
    reimplemented here) so it needs the project's own ambient environment, since `app` isn't a
    pip-installable package. Run with:

    ```
    uv run notebooks/janic/query_engine_prototype.py
    ```

    from the repo root (no `--sandbox`) -- same as notebook 1, cwd matters here too (for both
    `.env` discovery and resolving the `app` import).
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # Must run from the repo root (see markdown above) -- this is what makes `app` importable
    # below, since it isn't installed as a package. repo_root is threaded into the next cell as a
    # parameter purely to force execution order (sys.path must be patched before `import app...`
    # runs), even though its value isn't otherwise used there.
    repo_root = Path.cwd()
    sys.path.insert(0, str(repo_root))
    return


@app.cell
def _():
    from app.un_data_stream.data.processor import DataProcessor
    from app.un_data_stream.analysis.query_engine import ResolutionQueryEngine

    return DataProcessor, ResolutionQueryEngine


@app.cell
def _():
    import logging
    import warnings

    import pandas as pd
    import psycopg
    from dotenv import load_dotenv, find_dotenv

    # Same find_dotenv(usecwd=True) rationale as notebook 1 -- load_dotenv()'s default frame-based
    # search looks in the wrong place if this file is ever imported/run from outside the repo tree.
    load_dotenv(find_dotenv(usecwd=True))
    return logging, pd, psycopg, warnings


@app.cell
def _(mo, psycopg):
    conn = psycopg.connect()
    mo.md(
        f"Connected to `{conn.info.dbname}` at `{conn.info.host}:{conn.info.port}` "
        f"as `{conn.info.user}`."
    )
    return (conn,)


@app.cell
def _(mo):
    mo.md("""
    ## Read back from Postgres
    """)
    return


@app.cell
def _(conn, mo, pd, warnings):
    # pandas warns that a plain psycopg connection isn't a SQLAlchemy connectable -- harmless
    # here (its DBAPI2 fallback path works fine for a plain SELECT), just noisy for two reads.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        resolution_outcomes_df = pd.read_sql_query("SELECT * FROM resolution_outcomes", conn)
        resolution_votes_df = pd.read_sql_query("SELECT * FROM resolution_votes", conn)

    mo.md(
        f"Read **{len(resolution_outcomes_df)}** outcomes and **{len(resolution_votes_df)}** "
        "votes from Postgres."
    )
    return resolution_outcomes_df, resolution_votes_df


@app.cell
def _(mo):
    mo.md("""
    ## Reshape into the wide `resolution_table` shape

    `resolution_votes` is normalized long (one row per resolution x country) -- storage-friendly,
    but not what the query engine consumes. `app/un_data_stream/data/processor.py`'s
    `DataProcessor.calculate_agreement_data()` expects one row per resolution with a *column per
    country* holding that country's vote code, and derives its `country_columns` list as "every
    column not in a fixed metadata set". That means anything in `resolution_outcomes` outside
    that metadata set gets silently miscounted as a country column -- so before merging, we keep
    only the metadata columns that function actually expects.
    """)
    return


@app.cell
def _():
    # Must match calculate_agreement_data()'s own `metadata_columns` set exactly (see
    # app/un_data_stream/data/processor.py) -- this is that same set, kept here rather than
    # imported since it's a local variable inside that method, not an exported constant.
    METADATA_COLUMNS = [
        "undl_id", "date", "session", "resolution", "draft",
        "committee_report", "meeting", "title", "agenda_title",
        "subjects", "total_yes", "total_no", "total_abstentions",
        "total_non_voting", "total_ms", "undl_link", "subject_id",
        "description", "agenda", "modality", "source_dataset",
    ]
    return (METADATA_COLUMNS,)


@app.cell
def _(METADATA_COLUMNS, mo, resolution_outcomes_df, resolution_votes_df):
    present_meta_cols = [c for c in METADATA_COLUMNS if c in resolution_outcomes_df.columns]
    _dropped_cols = [c for c in resolution_outcomes_df.columns if c not in present_meta_cols]

    votes_wide = resolution_votes_df.pivot(index="undl_id", columns="country_code", values="vote")

    resolution_table = resolution_outcomes_df[present_meta_cols].merge(
        votes_wide, on="undl_id", how="left"
    )
    _country_cols = [c for c in resolution_table.columns if c not in present_meta_cols]
    # A resolution absent entirely from resolution_votes (adopted without a vote -- no 967 field
    # at all) gets NaN for every country here after the left-join; 'X' (non-voting) is correct.
    resolution_table[_country_cols] = resolution_table[_country_cols].fillna("X")

    mo.md(f"""
    Built a `resolution_table`-shaped DataFrame: **{len(resolution_table)}** rows x
    **{len(_country_cols)}** country columns.

    Columns dropped (not in `calculate_agreement_data`'s expected shape yet -- `vote_note`,
    `amended_draft`, `related_documents`, `source_updated_at`, `inserted_at` aren't consumed by
    the query engine today; not a bug, just a gap for whenever the app wants to surface them):
    {", ".join(_dropped_cols) or "none"}.
    """)
    return resolution_table, votes_wide


@app.cell
def _(mo, votes_wide):
    mo.ui.table(votes_wide)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Compute agreement data with the app's own code

    Reusing `DataProcessor.calculate_agreement_data()` unmodified -- the exact function
    `DataRepository` calls in production -- rather than reimplementing the vectorised
    consensus/multilateral-score math here. `DataProcessor` only needs a `config` dict and a
    `logger` to construct; neither is touched by this particular method, so an empty config is
    fine (no data_sources.yaml needed for this).
    """)
    return


@app.cell
def _(logging):
    logger = logging.getLogger("query_engine_prototype")
    logger.addHandler(logging.NullHandler())
    return (logger,)


@app.cell
def _(DataProcessor, logger, mo, resolution_table):
    processor = DataProcessor(config={}, logger=logger)
    consensus_scores, country_columns, multilateral_scores, vote_bool_arrays = (
        processor.calculate_agreement_data(resolution_table)
    )

    mo.md(
        f"`country_columns`: **{len(country_columns)}**. "
        f"`multilateral_scores` shape: **{multilateral_scores.shape}**."
    )
    return (
        consensus_scores,
        country_columns,
        multilateral_scores,
        vote_bool_arrays,
    )


@app.cell
def _(consensus_scores, resolution_table):
    resolution_table_scored = resolution_table.copy()
    resolution_table_scored["consensus_score"] = resolution_table_scored["undl_id"].map(
        consensus_scores
    )
    return (resolution_table_scored,)


@app.cell
def _(mo):
    mo.md("""
    ## Wire up a real `ResolutionQueryEngine`

    `ResolutionQueryEngine.__init__` only ever calls `repo.get_data()` and reads `repo.logger` --
    it doesn't care how `repo` was built. `DataRepository`'s own constructor, though, only knows
    how to build from raw UNDL fetches or load a fixed CSV/pkl cache directory -- neither fits
    "I already have these DataFrames in memory from Postgres". Rather than fighting that
    constructor, this defines a minimal stand-in satisfying the same contract -- feeding the
    genuine, unmodified `ResolutionQueryEngine` class, just built our own way.

    Subject/thesaurus/member-states tables are empty for now (that ingestion hasn't been
    prototyped yet -- see the "Next steps" note at the bottom). Fine for every query exercised
    below except subject-filtered `query_resolutions`, which isn't tested here.
    """)
    return


@app.cell
def _(pd):
    class StandInRepo:
        """Duck-types DataRepository's contract with ResolutionQueryEngine: get_data() + logger."""

        def __init__(
            self, resolution_table, country_columns, multilateral_scores, vote_bool_arrays, logger
        ):
            self._data = {
                "resolution": resolution_table,
                "resolution_subject": pd.DataFrame(columns=["undl_id", "subject_id"]),
                "subject": pd.DataFrame(),
                "closure": pd.DataFrame(columns=["ancestor_id", "descendant_id"]),
                "broader": pd.DataFrame(),
                "country_columns": country_columns,
                "member_states": pd.DataFrame(),
                "multilateral_scores": multilateral_scores,
                "vote_bool_arrays": vote_bool_arrays,
            }
            self.logger = logger

        def get_data(self):
            return self._data

    return (StandInRepo,)


@app.cell
def _(
    ResolutionQueryEngine,
    StandInRepo,
    country_columns,
    logger,
    mo,
    multilateral_scores,
    resolution_table_scored,
    vote_bool_arrays,
):
    repo = StandInRepo(
        resolution_table_scored, country_columns, multilateral_scores, vote_bool_arrays, logger
    )
    engine = ResolutionQueryEngine(repo)
    mo.md("Constructed a real `ResolutionQueryEngine` from the Postgres-sourced data.")
    return (engine,)


@app.cell
def _(mo):
    mo.md("""
    ## Run real queries
    """)
    return


@app.cell
def _(engine, mo):
    resolutions_result = engine.query_resolutions()
    mo.md(f"`query_resolutions()`: **{len(resolutions_result)}** resolutions returned.")
    return (resolutions_result,)


@app.cell
def _(resolutions_result):
    resolutions_result[["undl_id", "resolution", "date", "modality", "consensus_score"]]
    return


@app.cell
def _(engine, mo):
    multilateral_stats = engine.query_multilateral_stats()
    mo.md(f"`query_multilateral_stats()`: **{len(multilateral_stats)}** countries.")
    return (multilateral_stats,)


@app.cell
def _(multilateral_stats):
    multilateral_stats.sort_values("participation_count", ascending=False).head(10)
    return


@app.cell
def _(mo, multilateral_stats):
    # Pick whichever country actually voted the most in this sample for the demo below, rather
    # than a hardcoded code that might happen to have zero participation in a small sample (all
    # its bilateral scores would be NaN, and the demo would look broken when it isn't).
    _voted = multilateral_stats[multilateral_stats["participation_count"] > 0].sort_values(
        "participation_count", ascending=False
    )
    demo_country = _voted.iloc[0]["country"]
    demo_country_votes = int(_voted.iloc[0]["participation_count"])
    mo.md(
        f"Using **{demo_country}** for the bilateral-agreement demo below "
        f"({demo_country_votes} votes in this sample)."
    )
    return (demo_country,)


@app.cell
def _(demo_country, engine, mo):
    agreement_avg = engine.query_agreement_between_countries(demo_country, average=True)
    mo.md(f"`query_agreement_between_countries({demo_country!r}, average=True)`:")
    return (agreement_avg,)


@app.cell
def _(agreement_avg):
    agreement_avg
    return


@app.cell
def _(mo):
    mo.md("""
    ## Cross-check against the raw Postgres data

    Independent verification that the pivot + `calculate_agreement_data()` reconstruction didn't
    lose or misattribute anything: recompute each country's participation count directly from
    `resolution_votes` via a plain groupby, and compare against what the query engine itself
    reports. This is a different code path from `calculate_agreement_data()`'s own vote-counting,
    so agreement here is real evidence, not just "it ran without an exception".
    """)
    return


@app.cell
def _(mo, multilateral_stats, resolution_votes_df):
    _direct = (
        resolution_votes_df[resolution_votes_df["vote"].isin(["Y", "N", "A"])]
        .groupby("country_code")
        .size()
        .rename("direct_participation")
    )
    _engine = multilateral_stats.set_index("country")["participation_count"].rename(
        "engine_participation"
    )
    participation_cross_check = _direct.to_frame().join(_engine, how="outer").fillna(0).astype(int)
    participation_mismatches = participation_cross_check[
        participation_cross_check["direct_participation"]
        != participation_cross_check["engine_participation"]
    ]

    mo.md(
        f"**{len(participation_mismatches)}** / {len(participation_cross_check)} countries have a "
        "participation-count mismatch between the direct Postgres groupby and the query engine's "
        "own computation."
    )
    return (participation_mismatches,)


@app.cell
def _(participation_mismatches):
    participation_mismatches
    return


@app.cell
def _(mo):
    mo.md("""
    ## Next steps

    - This closes out the voting-data prototyping thread: fetch → parse → QA → load → read back →
      real `ResolutionQueryEngine` queries, all validated against live data end-to-end.
    - Subject-filtered `query_resolutions` and the member-states-backed country name QA
      (`app/data.py`'s authority list) both need the thesaurus/member-states ingestion that was
      deliberately deferred earlier -- natural next thread, same fetch/parse/QA/load shape as
      notebook 1.
    - The five dropped columns noted above (`vote_note`, `amended_draft`, `related_documents`,
      `source_updated_at`, `inserted_at`) are a real, if small, gap between what's stored and what
      the query engine's `calculate_agreement_data()` currently expects -- worth a note in the
      plan if/when the app wants to surface any of them.
    - Everything here still runs against the local prototype Postgres with today's small sample
      (20 outcomes / 2702 votes) -- results are directionally sane but not meaningful at that
      scale; that's expected, this notebook is validating the *mechanism*, not the data.
    """)
    return


if __name__ == "__main__":
    app.run()
