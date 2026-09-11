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
    # Ingestion prototype: UNBIS thesaurus from the UNDL MARC authority API

    **Second draft, replacing the original TTL/S3-based version of this notebook.** That version
    fetched the thesaurus as a bulk TTL file (working around a WAF block on the direct download
    path) and ran it through the app's real `ThesaurusProcessor`. Follow-up investigation — see
    `plans/intermediate_storage_layer_plan.md`'s "Querying the UNDL API for member states and
    thesaurus data" section — found that TTL export (and the alternative Skosmos REST API,
    `metadata.un.org/skosmos/rest/v1/thesaurus/`, also evaluated) both lag the live MARC editorial
    system: MARC has **8,523** thesaurus concept records (`150`-tagged) vs. **7,341** total
    subjects in the TTL export this notebook originally parsed. So this rewrite fetches directly
    from the MARC *authorities* API instead — same collection and per-record shape
    `member_states_ingestion_prototype.py` already uses (`/editor/api/marc/auths/records`), just a
    different tag (`150` for thesaurus concepts, vs. `110` for member/former states).

    A nice side effect: this no longer needs the real `app` package or `rdflib` at all — back to
    a `--sandbox`-compatible PEP 723 header like `ingestion_prototype.py`, not the ambient-env
    requirement the TTL version needed for `ThesaurusProcessor`.

    Run standalone with `uv run notebooks/janic/thesaurus_ingestion_prototype.py`, or interactively
    with `uv run marimo edit --sandbox notebooks/janic/thesaurus_ingestion_prototype.py`.
    """)
    return


@app.cell
def _():
    import requests
    import pandas as pd
    from collections import Counter, deque

    return Counter, deque, pd, requests


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
    # Full backfill needs all ~8,523 concepts (see intro markdown); this prototype validates the
    # mechanism against a substantial sample, same reasoning as ingestion_prototype.py's own
    # SAMPLE_SIZE for GA resolutions ("existing bulk pipeline already owns full history" doesn't
    # apply here the same way, but a live end-to-end run at full scale isn't needed to prove the
    # parsing/hierarchy-building logic works).
    SAMPLE_SIZE = 300
    # NORTH MACEDONIA / SOUTHERN EUROPE -- known live examples (found while evaluating the MARC vs.
    # Skosmos APIs) with a confirmed reciprocal 550/$w broader<->narrower relationship. Included
    # regardless of what the recent-updated sample happens to catch, so the reciprocity QA check
    # below has real data to work with rather than depending on both ends of some relationship
    # landing in a 300-of-8523 random sample by luck.
    FEATURED_RECORD_IDS = ["275600", "275568"]
    return BASE_URL, FEATURED_RECORD_IDS, SAMPLE_SIZE


@app.cell
def _(mo):
    mo.md("""
    ## Fetch

    Same two-endpoint shape as the other two notebooks (search/list → per-record detail).
    `search=150:*` matches every record with a `150` field (MARC tag for a thesaurus concept's
    preferred term) -- confirmed live to be a real, targeted filter (8,523 results) and not a
    parser quirk like some other queries against this API turned out to be: checked against the
    *unfiltered* collection total (362,883) and spot-checked 5 sample hits, all genuine
    `040$f=unbist`-sourced concepts with real `035$a` thesaurus URIs.
    """)
    return


@app.cell
def _(BASE_URL, requests):
    def list_thesaurus_concept_ids(sample_size: int, page_limit: int = 100) -> list[str]:
        """Page the search endpoint (most-recently-updated first) and collect a sample of ids."""
        out: list[str] = []
        start = 1
        while len(out) < sample_size:
            resp = requests.get(
                BASE_URL,
                params={
                    "search": "150:*",
                    "start": start,
                    "limit": min(page_limit, sample_size - len(out)),
                    "sort": "updated",
                    "direction": "desc",
                },
                timeout=15,
            )
            resp.raise_for_status()
            batch = resp.json()["data"]
            if not batch:
                break
            out.extend(u.rsplit("/", 1)[-1] for u in batch)
            start += len(batch)
        return out[:sample_size]

    return (list_thesaurus_concept_ids,)


@app.cell
def _(BASE_URL, requests):
    def fetch_auth_record_detail(record_id) -> dict:
        """Full raw MARC-ish authority record for one entry."""
        resp = requests.get(f"{BASE_URL}/{record_id}", timeout=15)
        resp.raise_for_status()
        return resp.json()["data"]

    return (fetch_auth_record_detail,)


@app.cell
def _(FEATURED_RECORD_IDS, SAMPLE_SIZE, list_thesaurus_concept_ids, mo):
    sampled_ids = list_thesaurus_concept_ids(SAMPLE_SIZE)
    # dict.fromkeys() to dedupe while preserving order, in case a featured id was already sampled.
    record_ids = list(dict.fromkeys(sampled_ids + FEATURED_RECORD_IDS))
    mo.md(f"**{len(record_ids)}** concept ids to fetch ({len(sampled_ids)} sampled + featured).")
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

    Field mapping, confirmed live against real records (NORTH MACEDONIA `275600` and SOUTHERN
    EUROPE `275568` — the same pair used to confirm the `550`/`$w` hierarchy mechanism while
    evaluating this approach):

    | Tag/subfield | Meaning | → |
    |---|---|---|
    | `035$a` (the `http://...` entry — this field also carries a parallel `T`-prefixed internal id, e.g. `T0010188`) | thesaurus concept URI | `subject_id` |
    | `150$a` | English preferred term | `label_en` |
    | `993`–`997` `$a` | localized preferred terms | `label_fr` / `label_es` / `label_ar` / `label_zh` / `label_ru` — same tag numbering convention confirmed for member states' `name_fr`..`name_ru` |
    | `450$a` (repeated) | non-preferred/variant terms (English) | `alt_labels_en`, semicolon-joined. **Known gap**: MARC's non-English alt-label field wasn't conclusively identified — one case (`497`, Russian) was spotted once during exploration but not confirmed as a systematic per-language pattern the way `993`-`997` are for preferred labels, so `alt_labels_fr/es/ar/zh/ru` aren't populated here (the TTL export's `ThesaurusProcessor` does have them, from SKOS `altLabel` triples with language tags) |
    | `072` (repeated, `$a`) | hierarchical domain code(s) (e.g. `17.04.00` — domain 17 = GEOGRAPHICAL DESCRIPTORS, `04` = its "EUROPE" micro-thesaurus) | `domain_code` — semicolon-joined; **repeats** (confirmed live on a 300-record sample: 28/300 carry more than one — a real polyhierarchy, not a parsing bug) |
    | `550` (repeated, `$a`+`$w`+`xref`) | related concept, direction via `$w` (`g`=broader/parent, `h`=narrower/child, absent=related/non-hierarchical) | used to build `broader_table` below (from `$w=h` entries only — each concept declares its own children), not a `subject_table` column itself |
    | top-level `updated` | last-modified timestamp (RFC 1123) | `source_updated_at` — same `utc=True` parsing gotcha as the other two notebooks |

    `node_type` is hardcoded `'concept'` for every row parsed here — correct for this fetch, since
    everything `150:*` matches genuinely is a concept. The ~19 top-level domains and their
    micro-thesauri (`node_type='scheme'`/`'micro_thesaurus'` in the combined table built below)
    aren't standalone `150`-tagged MARC records at all (confirmed via multiple negative searches —
    see `plans/intermediate_storage_layer_plan.md`) — they're fetched from the Skosmos REST API
    instead, once, as static bootstrap data. See "Bootstrap: domains and micro-thesauri" below.
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
def thesaurus_uri(entries_035):
    """Pick the http(s):// thesaurus URI out of 035 -- it also carries a parallel T-prefixed
    internal id (e.g. "T0010188") in a separate occurrence of the same field/subfield.
    """
    for v in subfield_values(entries_035, "a"):
        if v.startswith("http://") or v.startswith("https://"):
            return v
    return None


@app.function
def parse_related_terms(entries):
    """Split 550 entries by direction: $w='g' -> broader (parent), $w='h' -> narrower (child),
    no $w -> related (non-hierarchical). Returns three lists of (name, xref_record_id) tuples.
    """
    broader, narrower, related = [], [], []
    for entry in entries or []:
        name = None
        w = None
        xref = None
        for sf in entry.get("subfields", []):
            if sf.get("code") == "a":
                name = sf.get("value")
                xref = sf.get("xref")
            elif sf.get("code") == "w":
                w = sf.get("value")
        if not name:
            continue
        item = (name, xref)
        if w == "g":
            broader.append(item)
        elif w == "h":
            narrower.append(item)
        else:
            related.append(item)
    return broader, narrower, related


@app.function
def parse_thesaurus_concept(record_id, raw: dict) -> dict:
    """One subject_table-shaped row."""
    return {
        "record_id": str(record_id),
        "subject_id": thesaurus_uri(raw.get("035")),
        "label_en": first_subfield(raw.get("150"), "a"),
        "label_fr": first_subfield(raw.get("993"), "a"),
        "label_es": first_subfield(raw.get("994"), "a"),
        "label_ar": first_subfield(raw.get("995"), "a"),
        "label_zh": first_subfield(raw.get("996"), "a"),
        "label_ru": first_subfield(raw.get("997"), "a"),
        "alt_labels_en": "; ".join(dict.fromkeys(subfield_values(raw.get("450"), "a"))) or None,
        # 072 can repeat (a concept can belong to multiple domains/micro-thesauri at once --
        # confirmed live: 28/300 in a varied sample had more than one). first_subfield() would
        # silently drop everything after the first, same class of bug as 046/680 in the
        # member-states notebook.
        "domain_code": "; ".join(dict.fromkeys(subfield_values(raw.get("072"), "a"))) or None,
        "node_type": "concept",
        "source_updated_at": raw.get("updated"),
    }


@app.cell
def _(pd, raw_records: dict):
    subject_table_df = pd.DataFrame(
        parse_thesaurus_concept(record_id, raw) for record_id, raw in raw_records.items()
    )
    subject_table_df["source_updated_at"] = pd.to_datetime(
        subject_table_df["source_updated_at"], utc=True
    )

    subject_table_df.head()
    return (subject_table_df,)


@app.cell
def _(mo):
    mo.md("""
    ### Build `broader_table` and `related_count` from `550`

    Iterates the raw records directly (not the DataFrame above) since each record's `550` field
    needs the *other* records' `subject_id`s to resolve into `(parent_id, child_id)` pairs — built
    from each concept's own `$w='h'` (narrower/child) entries only, not both directions, since
    `550` stores the relationship reciprocally on both records (confirmed live) and using both
    would just double-count every edge.
    """)
    return


@app.cell
def _(pd, raw_records: dict):
    id_to_uri = {rid: thesaurus_uri(raw.get("035")) for rid, raw in raw_records.items()}

    broader_rows = []
    related_counts = []
    for _record_id, _raw in raw_records.items():
        _self_uri = id_to_uri[_record_id]
        _broader, _narrower, _related = parse_related_terms(_raw.get("550"))
        related_counts.append({"record_id": _record_id, "related_count": len(_related)})
        for _name, _xref in _narrower:
            _child_uri = id_to_uri.get(str(_xref)) if _xref is not None else None
            broader_rows.append(
                {
                    "parent_id": _self_uri,
                    "child_name": _name,
                    "child_record_id": str(_xref) if _xref is not None else None,
                    "child_id": _child_uri,
                }
            )

    broader_raw_df = pd.DataFrame(broader_rows)
    related_counts_df = pd.DataFrame(related_counts)
    return broader_raw_df, related_counts_df


@app.cell
def _(mo):
    mo.md("""
    ## QA
    """)
    return


@app.cell
def _(mo, subject_table_df):
    _dupe_subject_ids = subject_table_df["subject_id"][subject_table_df["subject_id"].duplicated()]
    _null_subject_id = subject_table_df[subject_table_df["subject_id"].isna()]
    _null_en_label = subject_table_df[subject_table_df["label_en"].isna()]
    mo.md(f"""
    - Duplicate `subject_id`: **{len(_dupe_subject_ids)}**
    - Rows with no resolvable `subject_id` (035 URI missing): **{len(_null_subject_id)}**
    - Rows with no English label: **{len(_null_en_label)}**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### `domain_code` (072) distribution

    Sanity check against the 18 known top-level thesaurus domains (`01`-`18`, per the Skosmos
    root listing found while evaluating this approach) — every code's leading 2-digit segment
    should fall in that range.
    """)
    return


@app.cell
def _(mo, subject_table_df):
    # domain_code is now potentially multi-valued ("14.02.02; 13.03.00; 01.07.01") -- split on the
    # join separator first, *then* take each individual code's domain prefix, rather than naively
    # splitting the whole joined string on "." (which would mangle multi-code rows).
    _all_codes = subject_table_df["domain_code"].dropna().str.split("; ").explode()
    _multi_domain = subject_table_df["domain_code"].dropna().str.contains("; ")
    _top_level = _all_codes.str.split(".").str[0]
    _out_of_range = _top_level[~_top_level.isin([f"{i:02d}" for i in range(1, 19)])]
    mo.md(f"""
    - Rows with a `domain_code`: **{subject_table_df['domain_code'].notna().sum()}** / {len(subject_table_df)}
    - Rows with *more than one* `domain_code` (real polyhierarchy, not a bug): **{_multi_domain.sum()}**
    - Distinct top-level domains seen (across all codes, not just the first per row): **{_top_level.nunique()}**
    - Top-level domain values outside `01`-`18` (should be 0): **{len(_out_of_range)}** {sorted(_out_of_range.unique()) if len(_out_of_range) else ""}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### `broader_table` resolution check

    Every `child_id` should resolve to a `record_id` already in this fetch -- with `SAMPLE_SIZE`
    far smaller than the full ~8,523 concepts, a meaningful fraction of edges pointing *outside*
    the sample is expected and fine (same shape as the "xref not in fetched set" cases in
    `member_states_ingestion_prototype.py`'s 510 check) — a full backfill run wouldn't have this
    gap at all, since it fetches every concept.
    """)
    return


@app.cell
def _(broader_raw_df, mo):
    _resolved = broader_raw_df["child_id"].notna()
    mo.md(f"""
    - Edges with a resolved `child_id`: **{_resolved.sum()}** / {len(broader_raw_df)}
    - Edges pointing outside this sample (expected, see above): **{(~_resolved).sum()}**
    """)
    return


@app.cell
def _(broader_raw_df):
    broader_table = broader_raw_df[broader_raw_df["child_id"].notna()][
        ["parent_id", "child_id"]
    ].drop_duplicates()
    broader_table
    return (broader_table,)


@app.cell
def _(mo):
    mo.md("""
    ### Reciprocity spot check

    Confirms `550`/`$w` really is stored on both ends, not just assumed from the earlier ad hoc
    check while evaluating this approach — uses the two `FEATURED_RECORD_IDS` (NORTH MACEDONIA /
    SOUTHERN EUROPE) guaranteed to be in this fetch regardless of sampling.
    """)
    return


@app.cell
def _(broader_table, mo, subject_table_df):
    _by_id = subject_table_df.set_index("record_id")
    _macedonia_uri = _by_id.loc["275600", "subject_id"]
    _southern_europe_uri = _by_id.loc["275568", "subject_id"]

    _macedonia_is_child_of_southern_europe = (
        (broader_table["parent_id"] == _southern_europe_uri)
        & (broader_table["child_id"] == _macedonia_uri)
    ).any()

    mo.md(f"""
    - NORTH MACEDONIA `subject_id`: `{_macedonia_uri}`
    - SOUTHERN EUROPE `subject_id`: `{_southern_europe_uri}`
    - `broader_table` contains the edge SOUTHERN EUROPE → NORTH MACEDONIA (parent → child): **{_macedonia_is_child_of_southern_europe}**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### `related_count` (550, no `$w`) — discovery only

    Non-hierarchical "related term" links aren't part of the currently-designed Postgres schema
    (no `subject_related` table) — counted here for visibility, not stored anywhere.
    """)
    return


@app.cell
def _(mo, related_counts_df):
    mo.md(f"""
    - Concepts with at least one related (non-hierarchical) term: **{(related_counts_df['related_count'] > 0).sum()}** / {len(related_counts_df)}
    - Total related-term links in this sample: **{related_counts_df['related_count'].sum()}**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Bootstrap: domains and micro-thesauri (Skosmos)

    The ~19 top-level domains and their micro-thesauri (e.g. domain `17` GEOGRAPHICAL DESCRIPTORS
    → micro-thesaurus `17.04` EUROPE) have no MARC record at all — confirmed via multiple
    independent negative searches (see `plans/intermediate_storage_layer_plan.md`'s "Domains and
    micro-thesauri" section for the full evidence). They're static, rarely-changing reference data
    (fundamentally unlike the ~8,500 individual concepts, which need the twice-daily polling
    cadence), so this fetches them **once**, from the Skosmos REST API, rather than trying to coax
    them out of MARC or re-polling Skosmos on a schedule.

    Checked the Skosmos API spec directly for a single call to enumerate every micro-thesaurus at
    once (`vocabularyStatistics`'s `subTypes` breakdown, `search?type=eurovoc:MicroThesaurus`) —
    neither works, Skosmos's own type index doesn't track this EuroVoc-borrowed type. So: one call
    for the vocabulary root (gives the 19 domain/root URIs + English labels), one `topConcepts`
    call per domain (18 calls, gives each domain's micro-thesauri), then one `data` call per
    domain *and* per micro-thesaurus for full 6-language labels.

    **Revised from this notebook's first draft, based on live evidence, not just reasoning about
    it.** The first draft deliberately kept concepts *un*connected to their micro-thesaurus in
    `broader_table` — reasoning that a micro-thesaurus's own `narrower` list in Skosmos is a flat
    classification-membership list (includes both intermediate regions and leaf countries at the
    same level), not the same relationship as the real concept-to-concept `broader`/`narrower`
    chain `550`/`$w` gives. That reasoning was sound on its own, but cross-checking a MARC-built
    sample against the reference TTL export (`reference_ds/2026_08_20_thesaurus_2.ttl`) found the
    **official published hierarchy adds this edge anyway**: NORTH MACEDONIA's TTL `skos:broader`
    set has *three* entries (SOUTHERN EUROPE, BALKAN REGION, **and** the EUROPE micro-thesaurus
    directly), while its raw MARC `550` field only ever gave two ($w=g to SOUTHERN EUROPE and
    BALKAN REGION) — same pattern confirmed on SOUTHERN EUROPE itself (TTL adds a direct edge to
    EUROPE the micro-thesaurus, alongside its real `550`-derived edge to EUROPE the regular
    concept). So the direct concept→micro-thesaurus edge is real production behavior, synthesized
    from `072` at TTL-generation time, not an artifact of a flatter Skosmos-only view — and
    filtering by micro-thesaurus is exactly the kind of thing the app's subject filter tree needs
    to support directly, not just via a `domain_code` attribute. **Decision: build this edge
    ourselves** (below, from each concept's `domain_code`, since MARC's `550` doesn't carry it) so
    the resulting `subject_broader`/`subject_closure` matches what's already in production, not a
    thinner version of it. See `plans/intermediate_storage_layer_plan.md` for the write-up.
    """)
    return


@app.cell
def _():
    SKOSMOS_BASE = "https://metadata.un.org/skosmos/rest/v1/thesaurus"
    return (SKOSMOS_BASE,)


@app.cell
def _(SKOSMOS_BASE, requests):
    def fetch_skosmos(path: str, **params) -> dict:
        resp = requests.get(f"{SKOSMOS_BASE}/{path}", params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    return (fetch_skosmos,)


@app.cell
def _(fetch_skosmos, mo):
    _root = fetch_skosmos("")
    # "00" (the UN Thesaurus root itself) plus domains "01"-"18" -- all 19 come back from one call,
    # each with an English prefLabel already (root gives English only; full 6-language labels are
    # fetched per-uri below, same as for micro-thesauri).
    domain_uris = [(s["uri"], s.get("prefLabel")) for s in _root["conceptschemes"]]
    mo.md(f"**{len(domain_uris)}** domain/root schemes found (expect 19: root `00` + domains `01`-`18`).")
    return (domain_uris,)


@app.cell
def _(domain_uris, fetch_skosmos, mo):
    with mo.status.progress_bar(total=len(domain_uris) - 1) as _bar:
        # topConcepts of each *domain* (not the root -- confirmed live the root's own topConcepts
        # call returns empty, the 18 domains aren't declared as its formal skos:hasTopConcept).
        microthesaurus_stubs: list[dict] = []
        for _uri, _label in domain_uris:
            if _uri.rsplit("/", 1)[-1] == "00":
                continue
            _tc = fetch_skosmos("topConcepts", scheme=_uri)
            for _mt in _tc.get("topconcepts", []):
                microthesaurus_stubs.append({"domain_uri": _uri, "mt_uri": _mt["uri"]})
            _bar.update()

    mo.md(f"**{len(microthesaurus_stubs)}** micro-thesauri found across 18 domains.")
    return (microthesaurus_stubs,)


@app.cell
def _(mo):
    mo.md("""
    ### Full multi-language data per domain and micro-thesaurus

    One `/data?uri=...` call each -- confirms each micro-thesaurus's own `eurovoc:domain` matches
    the domain it was found under (a consistency check on the two-step fetch above, not assumed).
    """)
    return


@app.cell
def _(domain_uris, fetch_skosmos, mo):
    def pref_labels_of(graph_node: dict) -> dict:
        vals = graph_node.get("prefLabel") or []
        if isinstance(vals, dict):
            vals = [vals]
        return {v["lang"]: v["value"] for v in vals if "lang" in v}

    def alt_labels_en_of(graph_node: dict) -> str | None:
        vals = graph_node.get("altLabel") or []
        if isinstance(vals, dict):
            vals = [vals]
        en = [v["value"] for v in vals if v.get("lang") == "en"]
        return "; ".join(dict.fromkeys(en)) or None

    domain_rows = []
    with mo.status.progress_bar(total=len(domain_uris)) as _bar:
        for _uri, _ in domain_uris:
            _data = fetch_skosmos("data", uri=_uri, format="application/json")
            _node = next(g for g in _data["graph"] if g["uri"] == _uri)
            _labels = pref_labels_of(_node)
            domain_rows.append(
                {
                    "record_id": None,
                    "subject_id": _uri,
                    "label_en": _labels.get("en"),
                    "label_fr": _labels.get("fr"),
                    "label_es": _labels.get("es"),
                    "label_ar": _labels.get("ar"),
                    "label_zh": _labels.get("zh"),
                    "label_ru": _labels.get("ru"),
                    "alt_labels_en": alt_labels_en_of(_node),
                    "domain_code": None,
                    "node_type": "root" if _uri.rsplit("/", 1)[-1] == "00" else "scheme",
                    "source_updated_at": None,
                }
            )
            _bar.update()
    return alt_labels_en_of, domain_rows, pref_labels_of


@app.cell
def _(mo):
    mo.md("""
    Not every `topConcepts` hit is actually a genuine micro-thesaurus, confirmed live: one entry
    ("CERTIFICATION (ENVIRONMENT)", found under domain `03`) is `topConceptOf` its domain but its
    own `type` is plain `skos:Concept` (no `eurovoc:MicroThesaurus`), and it uses the regular
    long-form concept id shape (with a `dct:identifier` T-number, same pattern as an ordinary
    concept) rather than the short domain-code-style id every real micro-thesaurus has. A domain
    can apparently have both proper micro-thesaurus subdivisions *and* a few standalone top-level
    concepts that don't fit into one -- filtered out by type below rather than trusted just because
    `topConcepts` returned it (this is also why an `eurovoc:domain` mismatch shows up otherwise:
    a plain concept doesn't carry that property at all).
    """)
    return


@app.cell
def _(
    alt_labels_en_of,
    fetch_skosmos,
    microthesaurus_stubs: list[dict],
    mo,
    pref_labels_of,
):
    microthesaurus_rows = []
    domain_mismatches = []
    not_actually_microthesaurus = []
    with mo.status.progress_bar(total=len(microthesaurus_stubs)) as _bar:
        for _stub in microthesaurus_stubs:
            _data = fetch_skosmos("data", uri=_stub["mt_uri"], format="application/json")
            _node = next(g for g in _data["graph"] if g["uri"] == _stub["mt_uri"])
            _types = _node.get("type") or []
            _types = [_types] if isinstance(_types, str) else _types
            if "http://eurovoc.europa.eu/schema#MicroThesaurus" not in _types:
                not_actually_microthesaurus.append(
                    {"uri": _stub["mt_uri"], "domain_uri": _stub["domain_uri"], "types": _types}
                )
                _bar.update()
                continue
            _labels = pref_labels_of(_node)
            _actual_domain = (_node.get("http://eurovoc.europa.eu/schema#domain") or {}).get("uri")
            if _actual_domain != _stub["domain_uri"]:
                domain_mismatches.append((_stub["mt_uri"], _stub["domain_uri"], _actual_domain))
            microthesaurus_rows.append(
                {
                    "record_id": None,
                    "subject_id": _stub["mt_uri"],
                    "label_en": _labels.get("en"),
                    "label_fr": _labels.get("fr"),
                    "label_es": _labels.get("es"),
                    "label_ar": _labels.get("ar"),
                    "label_zh": _labels.get("zh"),
                    "label_ru": _labels.get("ru"),
                    "alt_labels_en": alt_labels_en_of(_node),
                    "domain_code": _node.get("dc11:identifier"),
                    "node_type": "micro_thesaurus",
                    "source_updated_at": None,
                    "_parent_domain_uri": _stub["domain_uri"],
                }
            )
            _bar.update()
    mo.md(
        f"Fetched **{len(microthesaurus_rows)}** genuine micro-thesauri; "
        f"**{len(not_actually_microthesaurus)}** `topConcepts` hits dropped (not `eurovoc:MicroThesaurus`-typed); "
        f"`eurovoc:domain` mismatches among the genuine ones: **{len(domain_mismatches)}**."
    )
    return domain_mismatches, microthesaurus_rows, not_actually_microthesaurus


@app.cell
def _(mo, not_actually_microthesaurus):
    mo.md(f"""
    Dropped (not real micro-thesauri): {not_actually_microthesaurus}
    """)
    return


@app.cell
def _(domain_mismatches, mo):
    mo.md(f"""
    Mismatches (should be empty): {domain_mismatches}
    """)
    return


@app.cell
def _(domain_rows, mo):
    _by_id = {r["subject_id"].rsplit("/", 1)[-1]: r["subject_id"] for r in domain_rows}
    root_uri = _by_id["00"]
    bootstrap_broader_rows = [
        {"parent_id": root_uri, "child_id": r["subject_id"]}
        for r in domain_rows
        if r["subject_id"] != root_uri
    ]
    mo.md(f"Root → domain edges: **{len(bootstrap_broader_rows)}** (expect 18).")
    return (bootstrap_broader_rows,)


@app.cell
def _(bootstrap_broader_rows, microthesaurus_rows):
    bootstrap_broader_rows_all = bootstrap_broader_rows + [
        {"parent_id": r["_parent_domain_uri"], "child_id": r["subject_id"]} for r in microthesaurus_rows
    ]
    return (bootstrap_broader_rows_all,)


@app.cell
def _(domain_rows, microthesaurus_rows, mo, pd):
    bootstrap_subject_df = pd.DataFrame(
        domain_rows + [{k: v for k, v in r.items() if k != "_parent_domain_uri"} for r in microthesaurus_rows]
    )
    _node_type_counts = bootstrap_subject_df["node_type"].value_counts().to_dict()
    mo.md(f"**{len(bootstrap_subject_df)}** bootstrap subject rows: {_node_type_counts}")
    return (bootstrap_subject_df,)


@app.cell
def _(mo):
    mo.md("""
    ### Concept → micro-thesaurus edges (derived from `domain_code`)

    Reconstructs each concept's micro-thesaurus URI from its own `domain_code` (strip the dots,
    e.g. `"17.04.00"` → `170400`) and checks it against the micro-thesaurus set just fetched --
    not assumed from the single EUROPE example found while evaluating this approach. A concept
    with more than one `domain_code` (the ~9% polyhierarchy case, see the QA section above) gets
    one edge per resolved code -- genuinely belongs under more than one micro-thesaurus at once.
    """)
    return


@app.cell
def _(bootstrap_subject_df, mo, subject_table_df):
    _known_mt_ids = set(
        bootstrap_subject_df[bootstrap_subject_df["node_type"] == "micro_thesaurus"]["subject_id"]
    )
    _codes_by_concept = subject_table_df[["subject_id", "domain_code"]].dropna(subset=["domain_code"])

    concept_microthesaurus_edges = []
    unresolved_domain_codes = []
    for _, _row in _codes_by_concept.iterrows():
        for _code in _row["domain_code"].split("; "):
            _mt_uri = "http://metadata.un.org/thesaurus/" + _code.replace(".", "")
            if _mt_uri in _known_mt_ids:
                concept_microthesaurus_edges.append({"parent_id": _mt_uri, "child_id": _row["subject_id"]})
            else:
                unresolved_domain_codes.append({"subject_id": _row["subject_id"], "domain_code": _code})

    mo.md(f"""
    - Concept `domain_code` values checked: **{len(concept_microthesaurus_edges) + len(unresolved_domain_codes)}**
    - Resolved to a fetched micro-thesaurus (→ real edge built): **{len(concept_microthesaurus_edges)}**
    - Unresolved (should be near-zero -- see below): **{len(unresolved_domain_codes)}**
    """)
    return concept_microthesaurus_edges, unresolved_domain_codes


@app.cell
def _(mo, unresolved_domain_codes):
    mo.md(f"""
    Unresolved `domain_code` values (not built as an edge): {unresolved_domain_codes}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Combine into final `subject_table` / `broader_table`

    Union of the MARC-derived concepts and the Skosmos-derived domains/micro-thesauri, **now
    edge-connected** via the `domain_code`-derived edges above -- root → domain → micro-thesaurus
    → concept is one single connected hierarchy, matching what the reference TTL already has in
    production (see the revised decision above), not the two-disconnected-components shape this
    notebook's first draft produced.
    """)
    return


@app.cell
def _(bootstrap_subject_df, mo, pd, subject_table_df):
    subject_table_all = pd.concat([subject_table_df, bootstrap_subject_df], ignore_index=True)
    _dupes = subject_table_all["subject_id"][subject_table_all["subject_id"].duplicated()]
    mo.md(f"""
    - Combined `subject_table`: **{len(subject_table_all)}** rows ({subject_table_all['node_type'].value_counts().to_dict()})
    - Duplicate `subject_id` across the combined table (should be 0 -- concept URIs and domain/micro-thesaurus URIs use disjoint formats): **{len(_dupes)}**
    """)
    return (subject_table_all,)


@app.cell
def _(
    bootstrap_broader_rows_all,
    broader_table,
    concept_microthesaurus_edges,
    pd,
):
    broader_table_all = pd.concat(
        [
            broader_table,
            pd.DataFrame(bootstrap_broader_rows_all),
            pd.DataFrame(concept_microthesaurus_edges),
        ],
        ignore_index=True,
    ).drop_duplicates()
    return (broader_table_all,)


@app.cell
def _(mo):
    mo.md("""
    ### `closure_table` (BFS over `broader_table`)

    Same algorithm `ThesaurusProcessor._create_closure_table` already uses (BFS up the `broader`
    chain from each subject, recording ancestor + depth), hand-rolled here since this notebook no
    longer imports the real app code (see intro markdown) — not a re-validation of that algorithm,
    just reproducing its shape from this fetch mechanism's own `broader_table`.
    """)
    return


@app.cell
def _(broader_table_all, deque, pd, subject_table_all):
    children_to_parents: dict[str, set[str]] = {}
    for _, _edge in broader_table_all.iterrows():
        children_to_parents.setdefault(_edge["child_id"], set()).add(_edge["parent_id"])

    def _ancestors(subject_id: str) -> dict[str, int]:
        out = {}
        seen = set()
        queue = deque([(subject_id, 0)])
        while queue:
            current, depth = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            if current not in out or out[current] > depth:
                out[current] = depth
            for parent in children_to_parents.get(current, ()):
                if parent not in seen:
                    queue.append((parent, depth + 1))
        return out

    closure_rows = [
        {"ancestor_id": ancestor, "descendant_id": subject_id, "depth": depth}
        for subject_id in subject_table_all["subject_id"].dropna()
        for ancestor, depth in _ancestors(subject_id).items()
    ]
    closure_table = pd.DataFrame(closure_rows)
    return (closure_table,)


@app.cell
def _(
    bootstrap_subject_df,
    closure_table,
    mo,
    subject_table_all,
    subject_table_df,
):
    _self_refs = closure_table[closure_table["ancestor_id"] == closure_table["descendant_id"]]
    _bootstrap_ids = set(bootstrap_subject_df["subject_id"])
    _concept_ids = set(subject_table_df["subject_id"].dropna())
    _bootstrap_component = closure_table[closure_table["descendant_id"].isin(_bootstrap_ids)]
    _concept_component = closure_table[closure_table["descendant_id"].isin(_concept_ids)]
    mo.md(f"""
    - `closure_table` rows: **{len(closure_table)}**
    - Self-references (depth 0): **{len(_self_refs)}** / {subject_table_all['subject_id'].notna().sum()} subjects with a resolvable `subject_id`
    - Max depth overall: **{closure_table['depth'].max() if len(closure_table) else 'n/a'}**
    - Max depth among root/domain/micro-thesaurus descendants (expect 2 — root→domain→micro-thesaurus): **{_bootstrap_component['depth'].max() if len(_bootstrap_component) else 'n/a'}**
    - Max depth among concept descendants (now reaches up through micro-thesaurus → domain → root too, not just the `550` chain — this is the connectivity fix): **{_concept_component['depth'].max() if len(_concept_component) else 'n/a'}**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Next steps

    - This notebook's fetch mechanism (MARC `150:*` search + per-record detail, **plus** a static
      one-time Skosmos bootstrap for domains/micro-thesauri) is what the real ingestion job should
      use, per `plans/intermediate_storage_layer_plan.md`'s "Querying the UNDL API for member
      states and thesaurus data" section — not the original TTL/S3 approach this file used to
      have, and not Skosmos for the concepts themselves (both evaluated and rejected there, with
      live evidence).
    - **`node_type='scheme'` gap resolved** (was open in the first MARC-only draft): domains
      (`node_type='scheme'`) and micro-thesauri (`node_type='micro_thesaurus'`, a new value beyond
      the TTL-era binary concept/scheme split — genuinely a third kind of node, not a good fit for
      either) now come from Skosmos, fetched once as static bootstrap data. **Remaining open gap**:
      non-English `alt_labels` for concepts — only English resolved (the source MARC field for
      other languages' variant terms wasn't conclusively identified); domains/micro-thesauri do get
      full 6-language labels, since Skosmos provides them directly.
    - **Concepts are now connected to their micro-thesaurus** (revised from this notebook's first
      draft, which deliberately left them unconnected): cross-checking against the reference TTL
      export found the official production hierarchy adds this edge too (confirmed on NORTH
      MACEDONIA / SOUTHERN EUROPE, both had one more `skos:broader` entry in the TTL than MARC's
      `550` alone gives), and filtering by micro-thesaurus is a real requirement, not just a
      nice-to-have — so `broader_table` now includes one edge per resolved `domain_code`, built
      alongside the root→domain→micro-thesaurus edges. `subject_broader`/`subject_closure` are a
      single connected hierarchy end to end now, not two disconnected components. See
      `plans/intermediate_storage_layer_plan.md` for the write-up of this decision.
    - **Cross-checked against `reference_ds/2026_08_20_thesaurus_2.ttl`** (the same file the old
      TTL-based version of this notebook parsed) — 201 freshly-fetched MARC concepts, all 6
      languages matched **100%** (201/201 each). `alt_labels_en` matched 115/119; 3 of the 4
      mismatches are plausible MARC-is-fresher-than-the-snapshot cases (MARC has more alt-labels
      than the 3-week-old TTL), one was a genuine whitespace bug in MARC's own subfield value
      (`' PERSONAL DATA FILES'`, leading space) — worth a defensive `.strip()` if this notebook's
      alt-label handling gets reused elsewhere; not fixed here since it's cosmetic and didn't
      affect anything downstream in this notebook.
    - No Postgres schema/load yet for `subject`/`subject_broader`/`subject_closure` — deliberately
      deferred, same scope limit as the other two prototype notebooks. The existing DDL in the plan
      doc already matches this notebook's column names directly (`broader_table`'s
      `parent_id`/`child_id`, `closure_table`'s `ancestor_id`/`descendant_id`/`depth`), so no
      schema changes look needed there — `node_type` has no `CHECK` constraint in the DDL, so the
      new `'micro_thesaurus'`/`'root'` values need no migration either.
    - A full (non-sampled) run would need `SAMPLE_SIZE` raised to cover all ~8,523 concepts —
      untested at that scale here; the resolution-rate/timing implications of a true full fetch
      aren't validated by this notebook's 300-record sample. The domain/micro-thesaurus bootstrap
      itself, by contrast, *is* already a full (non-sampled) fetch — there are only ~19 domains and
      a few hundred micro-thesauri total, cheap to fetch completely every time.
    - `resolution_subject` (per-resolution subject matching against `label_en`/`alt_labels_en`)
      still isn't exercised by any notebook yet — needs real `991.d` subject strings from
      `ingestion_prototype.py`'s output matched against this notebook's `subject_table`, per the
      plan's matching grammar.
    """)
    return


if __name__ == "__main__":
    app.run()
