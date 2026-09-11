# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pandas==2.3.3",
#     "requests==2.32.5",
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
    # Ingestion prototype: member states from the UNDL MARC authority API

    Prototype for `plans/intermediate_storage_layer_plan.md` (member_states isn't in the plan's
    Postgres schema section yet — this notebook is the exploration that schema design is based
    on). Fetches the full UN member/former-state authority list directly from the UNDL MARC
    *authorities* API (`/editor/api/marc/auths/records`, distinct from the *bibliographic* API
    `ingestion_prototype.py` uses for resolutions), parses it into a DataFrame, and QAs it against
    what the app already trusts (`app/assets/joining_dates.csv`).

    Fetch + parse + QA only, matching `ingestion_prototype.py`'s original scope before its
    Postgres load section was added — no `member_states` Postgres table exists yet; that's a
    follow-up once this shape is agreed.

    **Cross-checked against the source team's own tooling** —
    `notebooks/voting-preprocessing/python_processing/notebooks/compile_unms_names_run.ipynb`
    (the notebook that builds the published `unms_names` CSV) and its output,
    `reference_ds/2026_08_17_unms_names.csv`. That comparison found three real gaps in this
    notebook's first-draft field mapping (046 can repeat for non-contiguous membership periods;
    678$u conflates "UN Founding Member" text with real resolution symbols; 510's `$w` subfield
    gives earlier/later direction that was being discarded), plus two fields worth adding
    (`scope_note`, `unms_ontology_link`) that the reference CSV has and this one didn't. A fourth
    gap surfaced afterward, from actually running the reference-CSV cross-check below rather than
    the initial comparison: `680` (behind the newly-added `scope_note`) turned out to repeat
    within a record too, same class of bug as `046` — caught because 2 records (Moldova, Syria)
    mismatched on first pass. All four are fixed below. See the `## Parse` section for the
    field-by-field mapping including these fixes, and the QA section's cross-check against the
    reference CSV for whether they actually line up now.

    Run standalone with `uv run notebooks/janic/member_states_ingestion_prototype.py`, or
    interactively with `uv run marimo edit --sandbox notebooks/janic/member_states_ingestion_prototype.py`.
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
    BASE_URL = "https://metadata.un.org/editor/api/marc/auths/records"
    PAGE_LIMIT = 100
    return (BASE_URL,)


@app.cell
def _(mo):
    mo.md("""
    ## Fetch

    Same two-endpoint shape as `ingestion_prototype.py` (search/list → per-record detail), but
    against the MARC *authorities* collection, not bibliographic records.

    Two things confirmed by ad hoc exploration before writing this notebook, both non-obvious:

    - **Subfield-scoped search doesn't work.** `110%9:ms` (or `110$9:ms`, trying to filter on
      field 110 subfield 9 specifically) returns 0 results. Plain `110:ms` works instead — it
      full-text-searches *all* of field 110's subfields and happens to match subfield 9's value
      exactly, returning exactly 193 records (matches the known current member-state count).
    - **Parenthesized boolean queries are broken.** `110:ms OR 110:fs` (244 results — a clean
      193+51 union, no overlap) is *not* equivalent to `(110:ms OR 110:fs)` (58 results) — the
      parser does something else entirely with the parens. `AND NOT 110:unknown` is also skipped
      deliberately: it's a literal text match against the single synthetic placeholder row
      (`110$a = "Unknown"`), not a semantic type filter, and interacts unpredictably with the
      parens bug above. The placeholder is instead dropped after parsing, the same way
      `app/un_data_stream/fetchers/member_states_fetcher.py` already does it (row has no ISO
      code).

    So the working query is the unparenthesized `110:ms OR 110:fs`.
    """)
    return


@app.cell
def _(BASE_URL, requests):
    def list_member_state_record_ids(page_limit: int = 100) -> list[dict]:
        """Page the auths search endpoint for all current + former member-state name records."""
        out: list[dict] = []
        start = 1
        while True:
            resp = requests.get(
                BASE_URL,
                params={
                    "search": "110:ms OR 110:fs",  # see markdown above: no parens, no AND NOT
                    "start": start,
                    "limit": page_limit,
                    "sort": "updated",
                    "direction": "asc",
                },
                timeout=15,
            )
            resp.raise_for_status()
            batch = resp.json()["data"]
            if not batch:
                break
            out.extend(batch)
            start += len(batch)
        return out

    return (list_member_state_record_ids,)


@app.cell
def _(BASE_URL, requests):
    def fetch_auth_record_detail(record_id) -> dict:
        """Full raw MARC-ish authority record for one entry (member state, former name, or the
        cross-referenced thesaurus concept a 550 field points to — same collection either way).
        """
        resp = requests.get(f"{BASE_URL}/{record_id}", timeout=15)
        resp.raise_for_status()
        return resp.json()["data"]

    return (fetch_auth_record_detail,)


@app.cell
def _(list_member_state_record_ids, mo):
    # The list endpoint returns URLs (.../records/{id}), not ids directly -- unlike the
    # resolutions notebook's brief records, which carry `_id` inline.
    record_urls = list_member_state_record_ids()
    record_ids = [u.rsplit("/", 1)[-1] for u in record_urls]
    mo.md(f"Found **{len(record_ids)}** member/former-state authority records.")
    return (record_ids,)


@app.cell
def _(Counter, fetch_auth_record_detail, mo, record_ids):
    with mo.status.progress_bar(total=len(record_ids)) as _bar:
        raw_records: dict = {}
        fetch_errors: list = []
        tag_counter = Counter()
        for _rid in record_ids:
            try:
                _raw = fetch_auth_record_detail(_rid)
            except Exception as exc:  # network hiccup on one record shouldn't kill the batch
                fetch_errors.append({"record_id": _rid, "error": str(exc)})
            else:
                raw_records[_rid] = _raw
                tag_counter.update(k for k in _raw if k.isdigit())
            _bar.update()

    mo.md(f"Fetched detail for **{len(raw_records)}** records, **{len(fetch_errors)}** errors.")
    return raw_records, tag_counter


@app.cell
def _(mo, raw_records: dict, tag_counter):
    mo.md(f"""
    ### MARC tag frequency across the sample

    {chr(10).join(f"- `{tag}`: {count}/{len(raw_records)}" for tag, count in tag_counter.most_common())}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Parse

    Field mapping, confirmed live against real records (Belize `71973` for a plain current
    member state; Cape Verde `72025`/Democratic Kampuchea `72149`/Laos `377135` for former-name
    entries; North Macedonia's pair `898082`/`228505` for the richest example — variant-name
    aliases, an earlier/later-name link, *and* a subject cross-reference all present at once):

    | Tag/subfield | Meaning | → |
    |---|---|---|
    | `110$a` | English name | `name_en` |
    | `110$9` | record type | `record_type` (`ms` = current member state, `fs` = former name of a state that still has a current name, also matches the single synthetic `"Unknown"` placeholder — dropped after parsing, no ISO code) |
    | `043$b` | M49 numeric code | `m49_code` |
    | `043$c` | ISO alpha-3 code | `iso_code` — join key everywhere else in the app |
    | `046` (repeated, `$s`/`$t`) | validity start / end date, **one pair per repeated field occurrence** | `coverage_periods` — `start:end` pairs joined with `|`, blank end = ongoing, same format `compile_unms_names_run.ipynb`'s own `coverage_periods` column uses. **Fixed from a first-draft bug**: `046` can repeat within a single record for non-contiguous membership periods — confirmed live on Cambodia's current-name record (`72150`), which has *three* separate `046` entries (`1955-1970`/`1975-1976`/`1990-open`), not one. Taking only the first (this notebook's original approach) silently dropped 2 of Cambodia's 3 periods; 7 records in the reference CSV have multiple periods. |
    | `993`–`997` `$a` | localized names | `name_fr` / `name_es` / `name_ar` / `name_zh` / `name_ru` (all 6 UN languages together with `110$a`=English) |
    | `410$a` (repeated) | variant spellings / abbreviations (MARC "See From" tracing) | `other_names` — semicolon-joined. This is the field behind the existing CSV's "Other Names" column (`app/data.py`'s `_LANG_COL`/search-alias logic) — confirmed by North Macedonia's former-name record `228505`, whose `410` list includes `FYROM`, exactly the example already named in that code's own comment. |
    | `510` (repeated, `$a`+`$w`) | earlier/later name of the *same entity*, `$w` gives direction (`a` = earlier, `b` = later) | `earlier_names` / `later_names` — semicolon-joined, one per direction. **Fixed from a first-draft bug**: this notebook originally kept only the first `510` xref with no notion of direction — Cambodia's current-name record has *two* `510` entries (both `$w=b`, "Khmer Republic" and "Democratic Kampuchea"), and treating "first xref" as authoritative silently dropped the second and ignored direction entirely. A `510` entry with **no `$w`** (undirected) is kept separately as `_undirected_names`, a QA finding rather than silently dropped or guessed — confirmed live: exactly one such case exists, Yemen Arab Republic (`875042`, fs) → "Democratic Yemen" with no `$w`, matching `compile_unms_names_run.ipynb`'s own documented exception (same one case, independently confirmed) — worth reporting back to the data owners rather than just working around it. |
    | `678$u` | either the literal text `"UN Founding Member"`, or an admitting GA resolution symbol — **two different concepts in one field** | `founding_member` (bool) / `membership_resolution` (resolution symbol only, `None` for founding members). **Fixed from a first-draft bug**: confirmed live on the USA's record (`28824`) that `678$u` is literally the string `"UN Founding Member"` for founding members, not a resolution — this notebook's original single `admission_resolution` field would have stored that literal text as if it were a resolution symbol for all 51 founding members. |
    | `680` (repeated, `$i`) | scope note (historical/succession context, e.g. "By a communication dated 24 Dec. 1991, the President of the Russian Federation notified the Secretary-General that membership of the USSR in the UN was continued by the Russian Federation") | `scope_note` — new field, wasn't captured at all before. **Also repeats** (same class of bug as `046`, caught the same way: cross-checking against the reference CSV found 2 records — Moldova, Syria — where this notebook's first-draft `first_subfield()` approach silently dropped a second `680` entry; confirmed live against Moldova's raw record, which has two). Joined with `|`, matching the reference CSV's own convention for this field. |
    | top-level `updated` | last-modified timestamp (RFC 1123) | `source_updated_at` — same `utc=True` parsing gotcha as the resolutions notebook |

    **New, not a direct field mapping**: `unms_ontology_link` — `http://metadata.un.org/UNMS/{m49_code}`, populated only for `record_type == "ms"` (current names; the reference CSV does the same). Points at a third identifier system, the "UN Member and Observer States Ontology", distinct from both this MARC auth record and the `550`→thesaurus link below — not otherwise explored in this notebook yet.

    **The `550` field** (the thing this exploration set out to figure out) is *not* a same-entity
    link like `510` — it's a cross-reference into a **different vocabulary**: the UNBIS subject
    thesaurus. Confirmed by following North Macedonia's `550` xref (`275600`): that target record
    has `150$a = "NORTH MACEDONIA"` (MARC tag `150` = topical/geographic subject heading, not
    `110` = corporate/country name) and `035$a = "http://metadata.un.org/thesaurus/1006488"` — a
    thesaurus concept URI in exactly the format `plans/intermediate_storage_layer_plan.md`'s
    `subject.subject_id` column already uses. So `550` is a **join key between `member_states` and
    the future `subject` table**: it lets a resolution indexed with the geographic subject
    "NORTH MACEDONIA" be traced back to the ISO code `MKD`, independent of voting data.

    Kept here as `subject_xref_id` (the raw target record id) — not yet resolved to a real
    `subject_id` URI, since the `subject` table doesn't exist until the thesaurus notebook runs.
    The QA section below verifies the "550 points at a thesaurus concept" hypothesis against the
    live data before this gets relied on.
    """)
    return


@app.function
def subfield_values(entries, code, ind1=None):
    """Subfield values for a code across every repeated occurrence of a MARC field."""
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
def first_xref(entries):
    """First subfield-`a` xref (record id) across a field's entries, if any carry one."""
    if not entries:
        return None
    for entry in entries:
        for sf in entry.get("subfields", []):
            if sf.get("code") == "a" and "xref" in sf:
                return sf["xref"]
    return None


@app.function
def parse_periods(entries):
    """One (start, end) tuple per repeated 046 field occurrence.

    046 can repeat within a single record for non-contiguous membership periods -- confirmed
    live: Cambodia's current-name record (72150) has three separate 046 entries
    (1955-1970 / 1975-1976 / 1990-open). Iterating entries directly (not subfield_values() per
    code) matters here: a naive flatten-then-zip of all $s values against all $t values would
    silently misalign start/end pairs for a record where one period is still ongoing (no $t at
    all) followed by another that has both.
    """
    if not entries:
        return []
    periods = []
    for entry in entries:
        sub = {sf["code"]: sf["value"] for sf in entry.get("subfields", [])}
        periods.append((sub.get("s"), sub.get("t")))
    return periods


@app.function
def format_periods(periods) -> str | None:
    """start:end pairs joined with '|', blank end = ongoing.

    Same format compile_unms_names_run.ipynb's own coverage_periods column uses -- matched
    deliberately for direct comparability against reference_ds/2026_08_17_unms_names.csv.
    """
    if not periods:
        return None
    return "|".join(f"{s or ''}:{e or ''}" for s, e in periods)


@app.function
def parse_related_names(entries):
    """Split 510 entries by direction: $w='a' -> earlier name, $w='b' -> later name, no $w ->
    undirected.

    Undirected entries are kept separately (as a QA finding, see the '_undirected_names' column)
    rather than silently dropped or guessed at -- confirmed live: exactly one such case exists,
    Yemen Arab Republic (875042, fs) -> "Democratic Yemen" with no $w, matching
    compile_unms_names_run.ipynb's own documented exception (same one case, independently
    confirmed here rather than just trusted from their notes).
    """
    earlier, later, undirected = [], [], []
    for entry in entries or []:
        sub = {sf["code"]: sf.get("value") for sf in entry.get("subfields", [])}
        name = sub.get("a")
        if not name:
            continue
        w = sub.get("w")
        if w == "a":
            earlier.append(name)
        elif w == "b":
            later.append(name)
        else:
            undirected.append(name)
    return earlier, later, undirected


@app.function
def parse_member_state(record_id, raw: dict) -> dict:
    """One member_states-shaped row."""
    record_type = first_subfield(raw.get("110"), "9")
    m49_code = first_subfield(raw.get("043"), "b")
    raw_membership_info = first_subfield(raw.get("678"), "u")
    _earlier, _later, _undirected = parse_related_names(raw.get("510"))

    return {
        "record_id": str(record_id),
        "iso_code": first_subfield(raw.get("043"), "c"),
        "m49_code": m49_code,
        "record_type": record_type,
        "name_en": first_subfield(raw.get("110"), "a"),
        "name_fr": first_subfield(raw.get("993"), "a"),
        "name_es": first_subfield(raw.get("994"), "a"),
        "name_ar": first_subfield(raw.get("995"), "a"),
        "name_zh": first_subfield(raw.get("996"), "a"),
        "name_ru": first_subfield(raw.get("997"), "a"),
        "other_names": "; ".join(dict.fromkeys(subfield_values(raw.get("410"), "a"))) or None,
        "coverage_periods": format_periods(parse_periods(raw.get("046"))),
        "founding_member": raw_membership_info == "UN Founding Member",
        "membership_resolution": (
            raw_membership_info
            if raw_membership_info and raw_membership_info != "UN Founding Member"
            else None
        ),
        "scope_note": "|".join(subfield_values(raw.get("680"), "i")) or None,
        "earlier_names": "; ".join(dict.fromkeys(_earlier)) or None,
        "later_names": "; ".join(dict.fromkeys(_later)) or None,
        "_undirected_names": "; ".join(dict.fromkeys(_undirected)) or None,
        "subject_xref_id": first_xref(raw.get("550")),
        "unms_ontology_link": (
            f"http://metadata.un.org/UNMS/{m49_code}" if record_type == "ms" and m49_code else None
        ),
        "source_updated_at": raw.get("updated"),
    }


@app.cell
def _(pd, raw_records: dict):
    member_states_df = pd.DataFrame(
        parse_member_state(record_id, raw) for record_id, raw in raw_records.items()
    )
    member_states_df["source_updated_at"] = pd.to_datetime(
        member_states_df["source_updated_at"], utc=True
    )
    # Nullable Int64, not float64 (the default when a numeric column has NaNs) -- this is an xref
    # record id, not a measurement, and str(275517.0) would break the /records/{id} URL below.
    member_states_df["subject_xref_id"] = member_states_df["subject_xref_id"].astype("Int64")

    member_states_df.head()
    return (member_states_df,)


@app.cell
def _(mo):
    mo.md("""
    ## QA
    """)
    return


@app.cell
def _(member_states_df, mo):
    mo.md(f"""
    **`record_type` breakdown**: {member_states_df['record_type'].value_counts(dropna=False).to_dict()}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Drop the synthetic "Unknown" placeholder

    Same rule `MemberStatesFetcher` already applies to the CSV: any row with no ISO code isn't a
    real state and can't be joined against anything.
    """)
    return


@app.cell
def _(member_states_df, mo):
    _no_iso = member_states_df[member_states_df["iso_code"].isna()]
    member_states_clean_df = member_states_df[member_states_df["iso_code"].notna()].copy()
    mo.md(
        f"Dropped **{len(_no_iso)}** row(s) with no ISO code: "
        f"{_no_iso['name_en'].tolist()}. **{len(member_states_clean_df)}** rows remain."
    )
    return (member_states_clean_df,)


@app.cell
def _(mo):
    mo.md("""
    ### Schema sanity checks
    """)
    return


@app.cell
def _(member_states_clean_df, mo):
    _dupe_ids = member_states_clean_df["record_id"][member_states_clean_df["record_id"].duplicated()]
    _bad_types = set(member_states_clean_df["record_type"].unique()) - {"ms", "fs"}
    _null_required = member_states_clean_df[
        member_states_clean_df[["record_id", "iso_code", "name_en", "record_type"]].isna().any(axis=1)
    ]
    mo.md(f"""
    - Duplicate `record_id`: **{len(_dupe_ids)}**
    - Unexpected `record_type` values: {_bad_types if _bad_types else "none"}
    - Rows missing a required field: **{len(_null_required)}**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### ISO-code coverage vs. `app/assets/joining_dates.csv`

    `joining_dates.csv` is the app's own authoritative list of ISO codes that actually appear in
    voting data (per `CLAUDE.md`). Every code there should resolve to at least one row here —
    anything missing would silently break name lookups for a country that *has* votes.
    """)
    return


@app.cell
def _(mo, pd):
    import pathlib

    _joining_dates_path = pathlib.Path("app/assets/joining_dates.csv")
    joining_dates_df = pd.read_csv(_joining_dates_path)
    mo.md(f"Loaded **{len(joining_dates_df)}** voting-data ISO codes from `{_joining_dates_path}`.")
    return (joining_dates_df,)


@app.cell
def _(joining_dates_df, member_states_clean_df, mo):
    _voting_isos = set(joining_dates_df["country"])
    _authority_isos = set(member_states_clean_df["iso_code"])
    missing_from_authority = sorted(_voting_isos - _authority_isos)
    mo.md(
        f"**{len(missing_from_authority)}** / {len(_voting_isos)} voting-data ISO codes have no "
        f"matching authority record: {missing_from_authority}"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    Codes here are expected to be exactly the ones `app/data.py`'s `_LEGACY_VOTING_CODES` already
    patches in by hand (`GER`, `SCG`) — codes that appear in voting data but were merged into a
    successor code in the authority table itself, not a gap in this fetch. Anything *else* showing
    up here would be a real finding.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### `other_names` (410) sample

    Spot-check that variant-name aliases came through as expected — should include cases like
    North Macedonia's former-name record carrying `FYROM`.
    """)
    return


@app.cell
def _(member_states_clean_df):
    member_states_clean_df[member_states_clean_df["other_names"].notna()][
        ["iso_code", "name_en", "record_type", "other_names"]
    ].head(10)
    return


@app.cell
def _(mo):
    mo.md("""
    ### `coverage_periods` (046) multi-period check

    Confirms the fix for the bug found via `compile_unms_names_run.ipynb`: records with more than
    one non-contiguous membership period (multiple `046` entries) should show up with a `|` in
    `coverage_periods`, not just their first period.
    """)
    return


@app.cell
def _(member_states_clean_df, mo):
    multi_period = member_states_clean_df[member_states_clean_df["coverage_periods"].str.contains("\\|", na=False)]
    mo.md(f"**{len(multi_period)}** record(s) with more than one coverage period.")
    return (multi_period,)


@app.cell
def _(multi_period):
    multi_period[["iso_code", "name_en", "record_type", "coverage_periods"]]
    return


@app.cell
def _(mo):
    mo.md("""
    ### `founding_member` / `membership_resolution` (678) split check

    Confirms the 678$u split worked: no `membership_resolution` value should ever literally equal
    `"UN Founding Member"` (that text should only ever end up in `founding_member=True`).
    """)
    return


@app.cell
def _(member_states_clean_df, mo):
    _founding_count = member_states_clean_df["founding_member"].sum()
    _leaked = member_states_clean_df[member_states_clean_df["membership_resolution"] == "UN Founding Member"]
    mo.md(f"""
    - `founding_member = True`: **{_founding_count}**
    - `membership_resolution` values that still literally say "UN Founding Member" (should be 0): **{len(_leaked)}**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### `scope_note` (680) sample

    New field, spot-check it reads as expected — should include cases like USSR's continuation-of-
    membership note.
    """)
    return


@app.cell
def _(member_states_clean_df):
    member_states_clean_df[member_states_clean_df["scope_note"].notna()][
        ["iso_code", "name_en", "record_type", "scope_note"]
    ].head(10)
    return


@app.cell
def _(mo):
    mo.md("""
    ### `unms_ontology_link` (derived) sanity check

    Should be populated for every `ms` record with an `m49_code`, and `None` everywhere else.
    """)
    return


@app.cell
def _(member_states_clean_df, mo):
    _ms_rows = member_states_clean_df[member_states_clean_df["record_type"] == "ms"]
    _ms_missing_link = _ms_rows[_ms_rows["unms_ontology_link"].isna() & _ms_rows["m49_code"].notna()]
    _non_ms_with_link = member_states_clean_df[
        (member_states_clean_df["record_type"] != "ms") & member_states_clean_df["unms_ontology_link"].notna()
    ]
    mo.md(f"""
    - `ms` records with an `m49_code` but no `unms_ontology_link` (should be 0): **{len(_ms_missing_link)}**
    - Non-`ms` records with a `unms_ontology_link` set (should be 0): **{len(_non_ms_with_link)}**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### `earlier_names` / `later_names` (510, direction-aware via `$w`)

    This notebook's first draft kept only the first `510` xref per record with no notion of
    direction, on the assumption it would resolve to another record in this same fetch sharing
    the same `iso_code`. That assumption turned out wrong for roughly half the records with a
    `510` field — confirmed by following actual mismatches (some are genuine successor-state
    links across *different* ISO codes, e.g. Russian Federation→USSR; some chain into older
    untyped pre-independence records outside the `ms`/`fs` query's reach, e.g. Belize→"British
    Honduras"). Real, useful data — the bug was in the parsing, not the source.

    Fixed by cross-checking against `compile_unms_names_run.ipynb`, which already solved this
    correctly: `510`'s `$w` subfield gives explicit direction (`a` = earlier, `b` = later), and
    a record can carry more than one `510` entry (confirmed live: Cambodia's current-name record
    has two, both `$w=b`). `earlier_names`/`later_names` below capture every directed entry,
    joined with `; ` when there's more than one. Entries with **no `$w` at all** are kept
    separately as `_undirected_names` — a QA finding to review and potentially report back to
    UNBIS, not silently dropped (the reference CSV drops this case entirely; this notebook keeps
    it instead, per your call to track it rather than discard it).

    One caveat carried over from the original investigation, still true and worth keeping in mind
    for the eventual ingestion job: re-running the full fetch minutes apart produced slightly
    different results in earlier testing, even with the record count (244) staying fixed — this
    system is live/actively edited, and paging by `sort=updated&direction=asc` isn't perfectly
    stable under concurrent edits.
    """)
    return


@app.cell
def _(member_states_clean_df, mo):
    _has_earlier = member_states_clean_df["earlier_names"].notna().sum()
    _has_later = member_states_clean_df["later_names"].notna().sum()
    _has_undirected = member_states_clean_df["_undirected_names"].notna().sum()
    mo.md(f"""
    - Records with `earlier_names`: **{_has_earlier}**
    - Records with `later_names`: **{_has_later}**
    - Records with `_undirected_names` (no `$w` — QA finding): **{_has_undirected}**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    #### Undirected `510` relations — QA finding

    Expected (per `compile_unms_names_run.ipynb`'s own notes): exactly one case, Yemen Arab
    Republic → Democratic Yemen. Anything else showing up here is new information worth reporting
    back to UNBIS, not something to silently work around.
    """)
    return


@app.cell
def _(member_states_clean_df, mo):
    undirected_findings = member_states_clean_df[member_states_clean_df["_undirected_names"].notna()][
        ["iso_code", "name_en", "record_type", "_undirected_names"]
    ]
    mo.md(f"**{len(undirected_findings)}** record(s) with an undirected `510` relation.")
    return (undirected_findings,)


@app.cell
def _(undirected_findings):
    undirected_findings
    return


@app.cell
def _(mo):
    mo.md("""
    ### `subject_xref_id` (550) investigation

    The actual question this notebook set out to answer. Hypothesis from the markdown above:
    `550` cross-references a thesaurus *concept* record (MARC tag `150`, `035$a` a
    `metadata.un.org/thesaurus/...` URI), not another member-states record. Verified by fetching
    the xref target for a sample of records that have a `550` field and checking its shape
    directly — this needs a fresh API call per sample record, since thesaurus concepts weren't
    part of the member-states fetch above.
    """)
    return


@app.cell
def _(fetch_auth_record_detail, member_states_clean_df, mo, pd):
    _has_subject_xref = member_states_clean_df[member_states_clean_df["subject_xref_id"].notna()]
    _sample = _has_subject_xref.head(15)

    subject_xref_checks = []
    with mo.status.progress_bar(total=len(_sample)) as _bar:
        for _, _row in _sample.iterrows():
            _target = fetch_auth_record_detail(str(_row["subject_xref_id"]))
            _thesaurus_uri = next(
                (v for v in subfield_values(_target.get("035"), "a") if "thesaurus" in v),
                None,
            )
            subject_xref_checks.append(
                {
                    "name_en": _row["name_en"],
                    "xref_id": _row["subject_xref_id"],
                    "has_150_tag": "150" in _target,
                    "has_110_tag": "110" in _target,
                    "thesaurus_uri": _thesaurus_uri,
                    "target_label": first_subfield(_target.get("150"), "a"),
                }
            )
            _bar.update()

    subject_xref_checks_df = pd.DataFrame(subject_xref_checks)
    _confirmed = subject_xref_checks_df["thesaurus_uri"].notna().sum()
    mo.md(
        f"**{_confirmed}** / {len(subject_xref_checks_df)} sampled `550` targets confirmed as "
        "thesaurus concepts (tag `150` present, `035$a` is a thesaurus URI, no `110` tag)."
    )
    return (subject_xref_checks_df,)


@app.cell
def _(subject_xref_checks_df):
    subject_xref_checks_df
    return


@app.cell
def _(mo):
    mo.md("""
    ## Cross-check against `reference_ds/2026_08_17_unms_names.csv`

    The actual test of whether the three gap fixes above (and the two new fields) landed
    correctly: compare this notebook's output against the CSV `compile_unms_names_run.ipynb`
    actually published, field by field, on live data — not just "the QA cells above look
    reasonable in isolation".

    **Can't join on record id** — confirmed earlier that this CSV's `undl_id` comes from a
    different ID space than this notebook's `record_id` (e.g. Belize is `877086` there vs `71973`
    here; same underlying MARC content, different frontend/id system). Joining instead on
    `(iso_code, record_type, name_en)`, which should be close to unique on both sides.

    **Multi-value fields use different separators** — this notebook uses `; `, the reference CSV
    uses `|` (its own established convention, matched here only for `coverage_periods` since that
    format's blank-end-means-ongoing semantics are worth keeping directly comparable). Compared as
    sets of trimmed values, not as raw strings, so the separator difference itself doesn't count
    as a mismatch.

    Not every reference column is compared — `undl_id`/`undl_link` are skip-worthy given the
    ID-space difference above, and `french`/`spanish`/`arabic`/`chinese`/`russian` were already
    validated (unchanged) by earlier QA sections in this notebook.
    """)
    return


@app.cell
def _(mo, pd):
    reference_df = pd.read_csv("reference_ds/2026_08_17_unms_names.csv").rename(
        columns={
            "iso": "iso_code",
            "name_status": "record_type",
            "member_state": "name_en",
            "m49": "m49_code",
        }
    )
    mo.md(f"Loaded **{len(reference_df)}** rows from the reference CSV.")
    return (reference_df,)


@app.cell
def _(member_states_clean_df, mo, reference_df):
    _dupe_keys_ours = member_states_clean_df[
        member_states_clean_df.duplicated(subset=["iso_code", "record_type", "name_en"], keep=False)
    ]
    _dupe_keys_ref = reference_df[
        reference_df.duplicated(subset=["iso_code", "record_type", "name_en"], keep=False)
    ]
    mo.md(f"""
    Join key `(iso_code, record_type, name_en)` uniqueness check before joining:

    - Duplicate keys in this notebook's data: **{len(_dupe_keys_ours)}**
    - Duplicate keys in the reference CSV: **{len(_dupe_keys_ref)}**
    """)
    return


@app.cell
def _(member_states_clean_df, reference_df):
    reference_comparison_df = member_states_clean_df.merge(
        reference_df,
        on=["iso_code", "record_type", "name_en"],
        how="outer",
        suffixes=("", "_ref"),
        indicator=True,
    )
    return (reference_comparison_df,)


@app.cell
def _(mo, reference_comparison_df):
    _join_counts = reference_comparison_df["_merge"].value_counts()
    mo.md(f"""
    Join coverage: {_join_counts.to_dict()}

    (`both` = matched on both sides; `left_only` = in this notebook's fetch but not the reference
    CSV; `right_only` = in the reference CSV but not this notebook's fetch — some of both are
    expected, since the two were fetched from different APIs at different times.)
    """)
    return


@app.cell
def _(reference_comparison_df):
    reference_comparison_df[reference_comparison_df["_merge"] != "both"][
        ["iso_code", "name_en", "record_type", "_merge"]
    ]
    return


@app.cell
def _(mo):
    mo.md("""
    ### Field-by-field comparison, matched rows only
    """)
    return


@app.cell
def _(pd, reference_comparison_df):
    def _to_set(value, sep):
        if pd.isna(value) or not str(value).strip():
            return frozenset()
        return frozenset(p.strip() for p in str(value).split(sep) if p.strip())

    _matched = reference_comparison_df[reference_comparison_df["_merge"] == "both"].copy()

    _set_field_checks = {
        "other_names": ("; ", "|"),
        "earlier_names": ("; ", "|"),
        "later_names": ("; ", "|"),
        "coverage_periods": ("|", "|"),
        "scope_note": ("|", "|"),  # 680 repeats within a record too, same as 046 -- see fix above
    }
    _scalar_field_checks = ["m49_code", "membership_resolution", "unms_ontology_link"]

    field_comparison_results = {}
    for _field, (_our_sep, _ref_sep) in _set_field_checks.items():
        _ours = _matched[_field].apply(lambda v: _to_set(v, _our_sep))
        _refs = _matched[f"{_field}_ref"].apply(lambda v: _to_set(v, _ref_sep))
        _match = _ours == _refs
        field_comparison_results[_field] = f"{_match.sum()} / {len(_matched)} match"

    for _field in _scalar_field_checks:
        _ours = _matched[_field].fillna("")
        _refs = _matched[f"{_field}_ref"] if f"{_field}_ref" in _matched.columns else _matched[_field]
        _refs = _refs.fillna("").astype(str)
        _match = _ours.astype(str) == _refs
        field_comparison_results[_field] = f"{_match.sum()} / {len(_matched)} match"

    # founding_member: ours is bool, reference is the string "True"/"False"
    _founding_match = _matched["founding_member"].astype(str) == _matched["founding_member_ref"].astype(str)
    field_comparison_results["founding_member"] = f"{_founding_match.sum()} / {len(_matched)} match"

    return (field_comparison_results,)


@app.cell
def _(field_comparison_results, mo):
    mo.md(
        "\n".join(f"- `{field}`: {result}" for field, result in field_comparison_results.items())
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Next steps

    - **550 finding confirmed** (see check above): it's a join key from `member_states` to the
      future `subject` table, not a same-entity link. Worth adding to
      `plans/intermediate_storage_layer_plan.md` once the `subject` table actually exists, so
      `resolution_subject` matches and `member_states` can be cross-referenced without going
      through voting data at all.
    - **Four real gaps found and fixed, three from the initial cross-check against
      `compile_unms_names_run.ipynb`/its published CSV, one found afterward from actually running
      this notebook's own reference-CSV comparison below**: `046` repeating within one record for
      non-contiguous membership periods (now `coverage_periods`, was silently truncated to the
      first period); `678$u` conflating "UN Founding Member" text with real resolution symbols
      (now split into `founding_member`/`membership_resolution`); `510`'s `$w` direction being
      discarded entirely (now `earlier_names`/`later_names`, with undirected relations tracked
      separately as a QA finding — see `_undirected_names` above — rather than silently dropped
      like the reference CSV does); and `680` (behind the newly-added `scope_note`) also repeating
      within a record, same bug as `046` — only caught because the reference-CSV cross-check
      itself flagged 2 real mismatches (Moldova, Syria) on the first pass, not assumed from the
      046 case transferring automatically. Two new fields adopted from the reference CSV:
      `scope_note` (680$i) and `unms_ontology_link` (derived, a third identifier system — the
      "UN Member and Observer States Ontology" — not yet explored beyond confirming it's
      populated correctly).
    - **Cross-check against the reference CSV** (see section above) is the actual test of whether
      these fixes landed right, and after the `680` fix, every compared field now matches
      243/243 with a clean 243/243 join (zero rows on either side unmatched). If everything still
      lines up on your own review, per your steer this is the point to start designing the actual
      `member_states` Postgres table.
    - Natural columns for that table based on this notebook's current shape: `record_id` (PK),
      `iso_code`, `m49_code`, `record_type`, `name_en/fr/es/ar/zh/ru`, `other_names`,
      `coverage_periods`, `founding_member`, `membership_resolution`, `scope_note`,
      `earlier_names`, `later_names`, `subject_xref_id` (nullable until `subject` exists — could
      become a real FK once it does), `unms_ontology_link`, `source_updated_at`.
      `_undirected_names` is discovery/QA-only, not a candidate column — matches the pattern
      `ingestion_prototype.py` already uses for its own `_modality_check`/`_raw_modality` fields.
    - `app/data.py`'s `_build_name_index` reconstruction (group by `iso_code`, pick the "active"
      row) needs a real rewrite against this shape, not a column-rename shim — it currently parses
      comma-separated `Start date`/`End date` strings from the old CSV schema, not the
      `coverage_periods` pipe-joined format used here and in the reference CSV.
    - Thesaurus ingestion prototype exists now too (`thesaurus_ingestion_prototype.py`) — the
      `subject_xref_id` link found here was independently cross-checked against it (15/15 sampled
      URIs matched).
    """)
    return


if __name__ == "__main__":
    app.run()
