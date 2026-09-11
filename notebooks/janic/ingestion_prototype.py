# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pandas==2.3.3",
#     "requests==2.32.5",
#     "psycopg[binary]==3.3.5",
#     "python-dotenv==1.2.3",
# ]
# ///

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
    # Ingestion prototype: GA voting data from the UNDL per-record API

    Prototype for `plans/intermediate_storage_layer_plan.md`. Fetches a sample of GA
    resolutions directly from the UNDL API (not the bulk CSV export), parses them into
    DataFrames shaped like the planned `resolution_outcomes` / `resolution_votes` Postgres
    tables, and runs the QA checks the plan calls for (vote-count cross-check, country
    name consistency).

    No database yet — this notebook only validates fetch + parse + QA. Writing to Postgres
    and reloading into `ResolutionQueryEngine` is a follow-up notebook.

    Run standalone with `uv run notebooks/janic/ingestion_prototype.py`, or interactively
    with `uv run marimo edit --sandbox notebooks/janic/ingestion_prototype.py` (the
    `--sandbox` flag uses the inline dependency block above, no need to install anything
    into the project's own environment).
    """)
    return


@app.cell
def _():
    import requests
    import pandas as pd
    from collections import Counter

    return Counter, pd, requests


@app.cell
def _(mo):
    mo.md("""
    ## Config
    """)
    return


@app.cell
def _():
    BASE_URL = "https://metadata.un.org/editor/api/marc/bibs/records"
    # Recent-history sample, not a full backfill — the existing bulk-CSV pipeline already owns
    # historical data per the plan. Large enough to hit abstentions, no-vote resolutions, and
    # multi-subject resolutions (all seen in a 20-record test run).
    SAMPLE_SIZE = 20
    PAGE_LIMIT = 100
    return BASE_URL, PAGE_LIMIT, SAMPLE_SIZE


@app.cell
def _(mo):
    mo.md("""
    ## Fetch

    Two endpoints (confirmed live, see the plan's "Querying the UNDL API" section):

    1. Search/list — paginated, sorted by `updated`. Gives us `_id` (→ `undl_id`) per resolution.
    2. Per-record detail — full MARC-ish record for one resolution, keyed by that `_id`.

    No server-side date-range filter exists (a `updated:[... TO ...]` search clause 500s), so
    incremental fetching in the real ingestion job will page `sort=updated&direction=desc`
    until a record's `updated` drops to/below the last watermark, rather than filtering
    server-side. This prototype just takes a flat recent sample.
    """)
    return


@app.cell
def _(BASE_URL, requests):
    # Could decorate this for vcrpy

    def list_recent_record_ids(sample_size: int, page_limit: int = 100) -> list[dict]:
        """Page the search endpoint (sorted by most-recently-updated) and collect brief records."""
        out: list[dict] = []
        start = 1
        while len(out) < sample_size:
            resp = requests.get(
                BASE_URL,
                params={
                    # 989:Voting Data alone also matches SC/HRC voting-data records; the source
                    # notebook's own query combines c='Voting Data' with p='981:"General Assembly"'
                    # (see compile_votes_ga_run.py 2.1) — added here since parse_outcome() below
                    # hardcodes source_dataset="GA" and assumes GA's field shape.
                    "search": '989:Voting Data AND 981:"General Assembly"',
                    "subtype": "all",
                    "format": "brief",
                    "sort": "updated",
                    "direction": "desc",
                    "start": start,
                    "limit": min(page_limit, sample_size - len(out)),
                },
                timeout=15,
            )
            resp.raise_for_status()
            batch = resp.json()["data"]
            if not batch:
                break
            out.extend(batch)
            start += len(batch)
        # Belt-and-suspenders: the per-page `limit` above should already land exactly on
        # sample_size, but slicing here guards against the API ever returning more per page
        # than requested (seen with some Invenio-style APIs' `limit` being a soft cap).
        return out[:sample_size]

    return (list_recent_record_ids,)


@app.cell
def _(BASE_URL, requests):
    # Could decorate this for vcrpy

    def fetch_record_detail(record_id) -> dict:
        """Full raw MARC-ish record for one resolution."""
        resp = requests.get(f"{BASE_URL}/{record_id}", timeout=15)
        resp.raise_for_status()
        return resp.json()["data"]

    return (fetch_record_detail,)


@app.cell
def _(PAGE_LIMIT, SAMPLE_SIZE, list_recent_record_ids, mo):
    brief_records = list_recent_record_ids(SAMPLE_SIZE, PAGE_LIMIT)
    mo.md(f"Fetched **{len(brief_records)}** brief records.")
    return (brief_records,)


@app.cell
def _(Counter, brief_records, fetch_record_detail, mo):
    with mo.status.progress_bar(total=len(brief_records)) as _bar:
        raw_records: dict = {}
        fetch_errors: list = []
        tag_counter = Counter()
        for _row in brief_records:
            try:
                _raw = fetch_record_detail(_row["_id"])
            except Exception as exc:  # network hiccup on one record shouldn't kill the batch
                fetch_errors.append({"undl_id": _row["_id"], "error": str(exc)})
            else:
                raw_records[_row["_id"]] = _raw
                tag_counter.update(k for k in _raw if k.isdigit())
            _bar.update()

    mo.md(f"Fetched detail for **{len(raw_records)}** records, **{len(fetch_errors)}** errors.")
    return raw_records, tag_counter


@app.cell
def _(mo, raw_records: dict, tag_counter):
    mo.md(f"""
    ### MARC tag frequency across the sample

    Not every field is present on every record (e.g. `967`/`996` — the vote roll-call and
    tally — are absent entirely for resolutions adopted without a vote). Useful for spotting
    fields the current mapping hasn't confirmed yet.

    {chr(10).join(f"- `{tag}`: {count}/{len(raw_records)}" for tag, count in tag_counter.most_common())}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Parse

    Field mapping cross-checked against the source team's own tooling —
    `notebooks/voting-preprocessing/python_processing/datasets/votes_ga/map.json` (the MARC-tag
    map for this exact dataset) and `src/unlibmd.py` (their extraction helpers) — which corrected
    a few guesses and confirmed which fields actually make it into the published CSVs
    (`compile_votes_ga_run.py`, sections 4.1/4.3):

    - `modality` is **not** a raw MARC field — it's derived (2.6 in the source notebook) by
      cross-checking three raw signals: whether `967` (per-country votes) is present, whether
      `590.a` == `"Vote"`, and whether `591.a` contains `"NON-RECORDED"`. Replicated below as
      `modality`/`_modality_check`, same three-way output (`"Vote, recorded"` /
      `"Vote, non-recorded"` / `"Without a vote"`) and same `"Issue"` fallback for combinations
      that don't fit, flagged as a QA finding rather than silently guessed.
    - `vote_note` is `996.a` ("additional voting information, e.g. peculiarities").
    - `993.a` is reused for four *different* concepts, disambiguated by the first indicator:
      blank → `related_documents`, `ind1="2"` → `draft`, `ind1="3"` → `committee_report`,
      `ind1="6"` → `amended_draft`.

    Per your "minimally copy what the CSV contains" steer: `committee_report` and `amended_draft`
    are now real fields (both are in the published `ga_outcomes`/`ga_voting` CSVs). `related_documents`
    is also now a real field — it's extracted by the source tooling too but actually dropped
    before either published CSV; kept here anyway since it's plausibly useful for a future
    "related resolutions" feature and costs one nullable column. Raw `recording_modality`
    (`591.a`) and `dhl_id` (`035.a`, and not even present in this API's responses) stay
    intermediate/discovery-only — the former is fully absorbed into the `modality` derivation,
    the latter isn't populated at all here. The agenda item *symbol* (`991.a`) isn't in
    `map.json` at all, so it stays out too.
    """)
    return


@app.function
def subfield_values(entries, code, ind1=None):
    """Subfield values for a code across every repeated occurrence of a MARC field.

    If `ind1` is given, only entries whose first indicator matches are considered — needed for
    tags like 993 where the same subfield code means different things depending on indicator.
    """
    if not entries:
        return []
    return [
        sf["value"]
        for entry in entries
        if ind1 is None or entry.get("indicators", [None, None])[0] == ind1
        for sf in entry.get("subfields", [])
        if sf["code"] == code
    ]


@app.function
def first_subfield(entries, code, ind1=None, default=None):
    vals = subfield_values(entries, code, ind1=ind1)
    return vals[0] if vals else default


@app.function
def derive_modality(has_votes: bool, raw_modality, recording_modality) -> tuple[str, str]:
    """Reproduce compile_votes_ga_run.py 2.6's three-way modality classification.

    Returns (modality, modality_check) — modality_check is the source notebook's own QA label
    ('Recorded Votes' / 'Non-Recorded Votes' / 'Without Vote' / 'Issue' for anything that
    doesn't fit the expected combinations).
    """
    with_votes = raw_modality == "Vote"
    not_recorded = bool(recording_modality) and "NON-RECORDED" in recording_modality

    if has_votes and with_votes:
        return "Vote, recorded", "Recorded Votes"
    if not has_votes and with_votes and not_recorded:
        return "Vote, non-recorded", "Non-Recorded Votes"
    if not has_votes and not with_votes and not not_recorded:
        return "Without a vote", "Without Vote"
    # Same fallback as the source notebook: an unclassifiable combination still needs *some*
    # modality value, so it's lumped with non-recorded — but flagged via modality_check so it
    # surfaces as a QA finding instead of silently passing as a normal non-recorded vote.
    return "Vote, non-recorded", "Issue"


@app.function
def parse_outcome(record_id, raw: dict) -> dict:
    """One resolution_outcomes-shaped row."""

    def _int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    tally_entries = raw.get("996") or [{}]
    tally = {sf["code"]: sf["value"] for sf in tally_entries[0].get("subfields", [])}

    title_entries = raw.get("245") or [{}]
    title_sub = {sf["code"]: sf["value"] for sf in title_entries[0].get("subfields", [])}
    title = " ".join(p for p in [title_sub.get("a"), title_sub.get("b"), title_sub.get("c")] if p)

    subject_strings = subfield_values(raw.get("991"), "d")
    agenda_titles = subfield_values(raw.get("991"), "c")
    agenda_symbols = subfield_values(raw.get("991"), "a")

    raw_modality = first_subfield(raw.get("590"), "a")
    recording_modality = first_subfield(raw.get("591"), "a")
    modality, modality_check = derive_modality(bool(raw.get("967")), raw_modality, recording_modality)

    return {
        "undl_id": str(record_id),
        "source_dataset": "GA",
        "resolution": first_subfield(raw.get("791"), "a"),
        "date": first_subfield(raw.get("992"), "a"),
        "modality": modality,
        "vote_note": tally.get("a"),
        "draft": first_subfield(raw.get("993"), "a", ind1="2"),
        "meeting": first_subfield(raw.get("952"), "a"),
        "subjects": " | ".join(subject_strings) if subject_strings else None,
        "total_yes": _int(tally.get("b")),
        "total_no": _int(tally.get("c")),
        "total_abstentions": _int(tally.get("d")),
        "total_non_voting": _int(tally.get("e")),
        "total_ms": _int(tally.get("f")),
        # undl_link intentionally not stored -- https://digitallibrary.un.org/record/{undl_id},
        # reconstructed on the fly, see plan doc's "Derived URL columns" decision
        "title": title or None,
        "session": _int(first_subfield(raw.get("791"), "c")),
        "committee_report": first_subfield(raw.get("993"), "a", ind1="3"),
        "amended_draft": first_subfield(raw.get("993"), "a", ind1="6"),
        # map.json's "ind1: null" means MARC's blank indicator, i.e. "_" in this API's encoding.
        # Not in the source team's own published CSV (dropped there), but kept here — plausibly
        # useful for a future "related resolutions" feature, and one nullable TEXT column is cheap.
        "related_documents": "; ".join(dict.fromkeys(subfield_values(raw.get("993"), "a", ind1="_"))) or None,
        "agenda_title": "; ".join(dict.fromkeys(t for t in agenda_titles if t)) or None,
        "description": None,  # SC-specific, always null for GA
        "agenda": None,  # SC-specific, always null for GA
        "source_updated_at": raw.get("updated"),
        # discovery-only, not in the CSV / DDL (see markdown note above):
        "_modality_check": modality_check,
        "_raw_modality": raw_modality,
        "_recording_modality": recording_modality,
        "_agenda_symbols": "; ".join(dict.fromkeys(agenda_symbols)) or None,
    }


@app.function
def parse_votes(record_id, raw: dict) -> list[dict]:
    """resolution_votes-shaped rows, plus ms_name kept alongside for the QA check below."""
    votes = []
    for entry in raw.get("967", []):
        sd = {sf["code"]: sf["value"] for sf in entry.get("subfields", [])}
        country_code = sd.get("c")
        if not country_code:
            continue
        votes.append(
            {
                "undl_id": str(record_id),
                "country_code": country_code,
                "vote": sd.get("d", "X"),  # subfield absent entirely -> non-voting/absent
                "ms_name": sd.get("e"),
            }
        )
    return votes


@app.cell
def _(pd, raw_records: dict):
    resolution_outcomes_df = pd.DataFrame(
        parse_outcome(record_id, raw) for record_id, raw in raw_records.items()
    )
    resolution_outcomes_df["date"] = pd.to_datetime(resolution_outcomes_df["date"]).dt.date
    # utc=True matters: the API's `updated` field is RFC 1123 ("Fri, 04 Sep 2026 18:43:38 GMT") --
    # pandas parses that as a *naive* timestamp by default (silently dropping the GMT/UTC offset),
    # which would let a TIMESTAMPTZ column below reinterpret it in the DB session's local zone.
    resolution_outcomes_df["source_updated_at"] = pd.to_datetime(
        resolution_outcomes_df["source_updated_at"], utc=True
    )

    _votes_raw = [
        vote for record_id, raw in raw_records.items() for vote in parse_votes(record_id, raw)
    ]
    resolution_votes_qa_df = pd.DataFrame(_votes_raw)  # includes ms_name, for QA only
    resolution_votes_df = resolution_votes_qa_df.drop(columns=["ms_name"])  # matches the planned DDL

    resolution_outcomes_df.head()
    return resolution_outcomes_df, resolution_votes_df, resolution_votes_qa_df


@app.cell
def _(mo):
    mo.md("""
    ## QA
    """)
    return


@app.cell
def _(mo, resolution_votes_qa_df):
    _unexpected = set(resolution_votes_qa_df["vote"].unique()) - {"Y", "N", "A", "X"}
    mo.md(
        f"**Vote code check**: {resolution_votes_qa_df['vote'].value_counts().to_dict()} — "
        + ("no unexpected codes." if not _unexpected else f"⚠️ unexpected codes: {_unexpected}")
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Vote-count cross-check

    Compares the reported totals (`996`) against counts derived from the individual votes
    (`967`), per the plan's QA decision. Resolutions with no recorded vote at all (adopted
    without a vote) have no `967`/`996` fields and are excluded rather than counted as
    mismatches.
    """)
    return


@app.cell
def _(mo, resolution_outcomes_df, resolution_votes_df):
    _derived = resolution_votes_df.pivot_table(
        index="undl_id", columns="vote", values="country_code", aggfunc="count", fill_value=0
    )
    for _col in ["Y", "N", "A", "X"]:
        if _col not in _derived.columns:
            _derived[_col] = 0

    _merged = resolution_outcomes_df.set_index("undl_id").join(_derived)
    _had_vote = _merged["total_yes"].notna()
    vote_count_mismatches = _merged[
        _had_vote
        & (
            (_merged["total_yes"] != _merged["Y"])
            | (_merged["total_no"] != _merged["N"])
            | (_merged["total_abstentions"] != _merged["A"])
            | (_merged["total_non_voting"] != _merged["X"])
        )
    ][["resolution", "total_yes", "Y", "total_no", "N", "total_abstentions", "A", "total_non_voting", "X"]]

    mo.md(
        f"{_had_vote.sum()} / {len(_merged)} resolutions had a recorded vote; "
        f"**{len(vote_count_mismatches)}** of those have a vote-count mismatch."
    )
    return (vote_count_mismatches,)


@app.cell
def _(vote_count_mismatches):
    vote_count_mismatches
    return


@app.cell
def _(mo):
    mo.md("""
    ### Modality derivation check

    Per `compile_votes_ga_run.py` 2.6: `modality` isn't a raw field, it's derived from three raw
    signals (see `derive_modality` above). Rows where those signals don't fit one of the three
    expected combinations get flagged `_modality_check == "Issue"` rather than silently guessed.
    """)
    return


@app.cell
def _(mo, resolution_outcomes_df):
    modality_issues = resolution_outcomes_df[resolution_outcomes_df["_modality_check"] == "Issue"]
    mo.md(f"**{len(modality_issues)}** / {len(resolution_outcomes_df)} resolutions have an unclassifiable modality combination.")
    return (modality_issues,)


@app.cell
def _(modality_issues):
    modality_issues
    return


@app.cell
def _(mo):
    mo.md("""
    ### Country name consistency check

    Per the plan's decision, `ms_name` isn't stored — it's used transiently to sanity-check
    our own name resolution. Full cross-checking against `app/data.py`'s authority list is
    deferred until the member_states API work happens; for now this only checks *internal*
    consistency (does the same `country_code` ever come back with different names across the
    sample?), which is still a useful preview of what that reconciliation will need to handle.
    """)
    return


@app.cell
def _(mo, resolution_votes_qa_df):
    _name_variants = resolution_votes_qa_df.groupby("country_code")["ms_name"].unique()
    name_inconsistencies = _name_variants[_name_variants.map(len) > 1]
    mo.md(
        f"**{len(name_inconsistencies)}** country code(s) have more than one distinct `ms_name` "
        "in the sample."
    )
    return (name_inconsistencies,)


@app.cell
def _(name_inconsistencies):
    # Figure out what to do with the name incosistencies -> should we reach out to joelle such that they can be fixed in the voting data? Ideally we can keep track on the actual resolution that has the incosistent naming
    name_inconsistencies
    return


@app.cell
def _(mo):
    mo.md("""
    ### Low-count code/name combinations

    Per `compile_votes_ga_run.py` 3.4: group votes by `(country_code, ms_name)` and look at the
    rarest combinations — a lens on likely transcription errors or rare/retired-code variants
    that the exact-duplicate check above wouldn't catch on its own. The source notebook uses a
    hard threshold of 30 occurrences, tuned for its ~20k-row full-history dataset; at this
    prototype's sample size that threshold doesn't transfer, so this just lists the rarest
    combinations directly rather than a fixed cutoff — recalibrate once this runs against full
    history.
    """)
    return


@app.cell
def _(mo, resolution_votes_qa_df):
    code_name_counts = (
        resolution_votes_qa_df.groupby(["country_code", "ms_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count")
    )
    mo.md(f"{len(code_name_counts)} distinct (country_code, ms_name) combinations in the sample; rarest shown below.")
    return (code_name_counts,)


@app.cell
def _(code_name_counts):
    code_name_counts.head(20)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Schema sanity checks

    Cheap checks that would also matter once this is a real Postgres load (PK uniqueness,
    `vote` CHECK constraint, required-not-null fields).
    """)
    return


@app.cell
def _(mo, resolution_outcomes_df, resolution_votes_df):
    _dupe_undl_ids = resolution_outcomes_df["undl_id"][resolution_outcomes_df["undl_id"].duplicated()]
    _bad_votes = resolution_votes_df[~resolution_votes_df["vote"].isin(["Y", "N", "A", "X"])]
    _null_required = resolution_outcomes_df[
        resolution_outcomes_df[["undl_id", "date", "modality"]].isna().any(axis=1)
    ]
    mo.md(
        f"""
        - Duplicate `undl_id` in outcomes: **{len(_dupe_undl_ids)}**
        - Votes with an out-of-CHECK-constraint value: **{len(_bad_votes)}**
        - Outcomes rows missing a required (`NOT NULL`) field: **{len(_null_required)}**
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Load into Postgres

    Writes `resolution_outcomes_df` / `resolution_votes_df` into the local Postgres set up per
    `notebooks/janic/postgres/` (`Dockerfile` + `schema.sql`). Connection info comes from a
    gitignored `.env` at the repo root (`PGHOST`/`PGPORT`/`PGDATABASE`/`PGUSER`/`PGPASSWORD`) --
    `psycopg.connect()` picks those up with no explicit conninfo needed, same as plain `psql`/libpq.

    **Reload strategy for this prototype: wipe both tables and re-insert on every run**, rather
    than upserting. This notebook is currently the only writer to this local DB, and each run's
    sample may not overlap with the last one's `undl_id`s, so a partial `ON CONFLICT DO UPDATE`
    wouldn't actually make this idempotent by itself. Real upsert logic (`ON CONFLICT DO UPDATE`,
    keyed off `source_updated_at` per the plan's "Row bookkeeping" decision) is deferred to the
    actual ingestion job, same as previously agreed -- this is just enough to prove the write path
    end-to-end and make the notebook safely re-runnable while iterating.
    """)
    return


@app.cell
def _():
    import psycopg
    from dotenv import load_dotenv, find_dotenv

    # load_dotenv()'s default search walks up from the *calling file's own directory* (stack-frame
    # based), not the process's cwd -- usecwd=True makes it search from cwd instead, matching how
    # this notebook is actually invoked (`uv run notebooks/janic/ingestion_prototype.py` from the
    # repo root, where .env lives). Confirmed via a standalone repro: the frame-based default
    # silently found no .env at all when the calling script lived outside the repo tree.
    load_dotenv(find_dotenv(usecwd=True))
    return (psycopg,)


@app.cell
def _(mo, psycopg):
    conn = psycopg.connect()
    mo.md(
        f"Connected to `{conn.info.dbname}` at `{conn.info.host}:{conn.info.port}` "
        f"as `{conn.info.user}`."
    )
    return (conn,)


@app.cell
def _():
    # Matches resolution_outcomes' column order in notebooks/janic/postgres/schema.sql exactly
    # (inserted_at is DB-generated via DEFAULT now(), not provided here).
    OUTCOME_COLUMNS = [
        "undl_id", "source_dataset", "resolution", "date", "modality", "draft", "meeting",
        "subjects", "vote_note", "total_yes", "total_no", "total_abstentions", "total_non_voting",
        "total_ms", "title", "session", "committee_report", "amended_draft",
        "related_documents", "agenda_title", "description", "agenda", "source_updated_at",
    ]
    return (OUTCOME_COLUMNS,)


@app.function
def to_pg_value(v):
    """Convert one DataFrame cell into a plain value psycopg can adapt directly.

    Needed because DataFrame cells built from parse_outcome()/parse_votes() come back as numpy
    scalars (int64/float64) or pandas Timestamps, not builtin Python types -- psycopg has no
    adapter for numpy scalar types by default and would raise on them untouched.
    """
    if v is None or v != v:  # v != v catches float NaN and pandas NaT (both fail self-equality)
        return None
    if hasattr(v, "to_pydatetime"):  # pandas Timestamp -> datetime.datetime
        return v.to_pydatetime()
    if hasattr(v, "item"):  # numpy scalar (int64, float64, ...) -> native Python int/float
        return v.item()
    return v


@app.function
def wipe_tables(conn):
    """Clear resolution_outcomes/resolution_votes so this notebook is safely re-runnable.

    Also handy standalone -- e.g. from another script/REPL sharing this connection -- when you
    just want a clean slate without re-fetching/re-parsing anything. For a quick clean slate from
    the shell with no Python involved at all, see notebooks/janic/postgres/reset.sh instead.
    """
    with conn.cursor() as cur:
        # FK from resolution_votes -> resolution_outcomes: delete children first.
        cur.execute("DELETE FROM resolution_votes")
        cur.execute("DELETE FROM resolution_outcomes")
    conn.commit()


@app.cell
def _(mo):
    mo.md("""
    Writing is intentionally the only thing this cell does. Verifying it worked is a separate
    step below, not folded in here -- and validating the write is different from validating the
    *extraction* (does `ResolutionQueryEngine` produce correct query results from this data),
    which is out of scope for this notebook entirely and belongs in the follow-up
    read-back-from-Postgres notebook.
    """)
    return


@app.cell
def _(OUTCOME_COLUMNS, conn, mo, resolution_outcomes_df, resolution_votes_df):
    outcome_rows = [
        tuple(to_pg_value(v) for v in row)
        for row in resolution_outcomes_df[OUTCOME_COLUMNS].itertuples(index=False, name=None)
    ]
    vote_rows = [
        tuple(to_pg_value(v) for v in row)
        for row in resolution_votes_df[["undl_id", "country_code", "vote"]].itertuples(
            index=False, name=None
        )
    ]

    # Not `with conn:` -- psycopg3's connection context manager commits/rolls back *and closes*
    # the connection on exit (unlike psycopg2), which would break the verification cells below
    # that reuse this same `conn`. Explicit commit/rollback instead, connection stays open.
    try:
        wipe_tables(conn)
        with conn.cursor() as _cur:
            _cols_sql = ", ".join(OUTCOME_COLUMNS)
            _placeholders = ", ".join(["%s"] * len(OUTCOME_COLUMNS))
            _cur.executemany(
                f"INSERT INTO resolution_outcomes ({_cols_sql}) VALUES ({_placeholders})",
                outcome_rows,
            )
            _cur.executemany(
                "INSERT INTO resolution_votes (undl_id, country_code, vote) VALUES (%s, %s, %s)",
                vote_rows,
            )
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()

    mo.md(f"Inserted **{len(outcome_rows)}** outcomes and **{len(vote_rows)}** votes.")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Verify the load

    Just "did the write work as intended" -- row counts match what was sent, plus a spot check on
    one row to catch column-order/type bugs a count alone wouldn't. Not a check of whether the
    *data itself* is analytically correct (that's what the QA section above is for, on the
    DataFrames, before they ever reach Postgres).
    """)
    return


@app.cell
def _(conn, mo, resolution_outcomes_df, resolution_votes_df):
    with conn.cursor() as _cur:
        _cur.execute("SELECT COUNT(*) FROM resolution_outcomes")
        db_outcome_count = _cur.fetchone()[0]
        _cur.execute("SELECT COUNT(*) FROM resolution_votes")
        db_vote_count = _cur.fetchone()[0]

    _outcome_ok = db_outcome_count == len(resolution_outcomes_df)
    _vote_ok = db_vote_count == len(resolution_votes_df)

    mo.md(f"""
    - `resolution_outcomes`: **{db_outcome_count}** in DB vs **{len(resolution_outcomes_df)}** in the DataFrame -- {"OK" if _outcome_ok else "MISMATCH"}
    - `resolution_votes`: **{db_vote_count}** in DB vs **{len(resolution_votes_df)}** in the DataFrame -- {"OK" if _vote_ok else "MISMATCH"}
    """)
    return


@app.cell
def _(conn, mo, resolution_outcomes_df):
    _sample_id = resolution_outcomes_df.iloc[0]["undl_id"]
    with conn.cursor() as _cur:
        _cur.execute(
            "SELECT resolution, date, total_yes, source_updated_at "
            "FROM resolution_outcomes WHERE undl_id = %s",
            (_sample_id,),
        )
        db_row = _cur.fetchone()

    _df_row = resolution_outcomes_df.loc[
        resolution_outcomes_df["undl_id"] == _sample_id,
        ["resolution", "date", "total_yes", "source_updated_at"],
    ].iloc[0]

    mo.md(f"""
    Spot check for `{_sample_id}`:

    - DB row: `{db_row}`
    - DataFrame row: `{tuple(_df_row)}`
    """)
    return


@app.cell
def _():
    ## Quick check
    #conn = psycopgs.connect()
    return


@app.cell(hide_code=True)
def _(conn, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM resolution_outcomes
        """,
        engine=conn
    )
    return


@app.cell(hide_code=True)
def _(conn, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM resolution_votes
        """,
        engine=conn
    )
    return


@app.cell(hide_code=True)
def _(conn, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM ingestion_runs -- missing not yet implemented
        """,
        engine=conn
    )
    return


@app.cell
def _(mo):
    mo.md("""
    - Local Postgres is up (`notebooks/janic/postgres/`, `docker build` + `docker run` — see that
      folder's `Dockerfile`) and this notebook now writes `resolution_outcomes` /
      `resolution_votes` into it end-to-end, with a row-count + spot-check verification step
      right above (plain wipe-and-reinsert — real upsert/idempotency is still a separate,
      deferred step, per the plan).
    - Second notebook (not started yet): read back from Postgres, pivot into the wide in-memory
      shape `ResolutionQueryEngine` expects, and actually run `query_resolutions` /
      `query_multilateral_stats` against it — that's the real "does this work" bar, not just
      "the DataFrame looks right".
    - Plan updated with the corrected mapping, the derived-modality QA step, and three new
      columns (`committee_report`, `amended_draft`, `related_documents` — the last kept even
      though the source CSVs drop it, since it's plausibly useful later). `_recording_modality`/
      `_agenda_symbols` stay unstored.
    - Fold in real name-mismatch / low-count-combination examples once member_states work starts.
    """)
    return


if __name__ == "__main__":
    app.run()
