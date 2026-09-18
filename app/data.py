import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Re-exported, not used here: the feature modules reach these through `app.data`
# (`data.get_country_region(...)` in multilateral_scatter.py, profile_page.py) rather than
# importing app.features.country_utils themselves. noqa: F401 keeps a linter from removing an
# import whose only purpose is the re-export — doing so breaks those call sites at *callback*
# time, not at import time, so nothing catches it until the tab is opened.
from .features.country_utils import (  # noqa: F401
    get_country_region,
    get_country_subregion,
)
from .un_data_stream import DataRepository, ResolutionQueryEngine
from .un_data_stream.analysis.snapshot import DataSnapshot
from .un_data_stream.data import reloader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "data_sources.yaml"
repo = DataRepository(config_path=str(_CONFIG_PATH))
query_engine = ResolutionQueryEngine(repo=repo)

available_countries = query_engine.get_available_countries()

# Supported UI languages and their column in the member-states authority table.
SUPPORTED_LANGS = ("en", "fr", "es", "ar", "zh", "ru")
_LANG_COL = {lang: f"name_{lang}" for lang in SUPPORTED_LANGS}

# ISO codes that appear in UN voting data but have no member_states row of their own. Each maps to
# the former-state ('fs') record describing the same entity, filed under its successor's ISO code —
# matched on (iso_code, name_en) because a successor can have several fs rows — plus any
# translations to patch in.
#
# The English name, variant spellings and the membership year range all come from that row, so none
# of those are hand-maintained any more. The translations still are: former-state authority records
# carry almost no non-English names (1 of 50 fs rows has any, against all 193 current ones), so
# without this patch a non-English UI would fall back to English for these two. Once GA/SC voting
# data switches to the canonical codes (GER->DEU, SCG->SRB), these entries can go.
_LEGACY_VOTING_CODES: dict[str, dict[str, Any]] = {
    "GER": {  # West Germany; voting records 1973-1990
        "source": ("DEU", "Germany, Federal Republic of"),
        "names": {
            "fr": "République fédérale d'Allemagne",
            "es": "República Federal de Alemania",
            "ru": "Федеративная Республика Германия",
        },
    },
    "SCG": {  # Serbia and Montenegro; voting records 2003-2006
        "source": ("SRB", "Serbia and Montenegro"),
        "names": {
            "fr": "Serbie-et-Monténégro",
            "es": "Serbia y Montenegro",
            "ru": "Сербия и Черногория",
        },
    },
}


# TODO: quick check that the years make sense? Also i feel like this code is fragile
def _coverage_years(coverage_periods: Any) -> tuple[list[int], list[int]]:
    """Start and end years from a `'start:end|start:end'` coverage_periods value.

    Dates are ISO (`'1955-12-14'`). A period with a blank end is still open and contributes no end
    year — that's what separates an ongoing membership from a closed one, and it's why a state with
    several periods (KHM: 1955-1970, 1975-1976, 1990-) reads as current.
    """
    if pd.isna(coverage_periods):
        return [], []
    starts: list[int] = []
    ends: list[int] = []
    for period in str(coverage_periods).split("|"):
        start, _, end = period.partition(":")
        if start[:4].isdigit():
            starts.append(int(start[:4]))
        if end[:4].isdigit():
            ends.append(int(end[:4]))
    return starts, ends


def _split_names(value: Any) -> list[str]:
    """Split a ';'-joined authority-record name list (`other_names`) into its entries."""
    if pd.isna(value):
        return []
    return [name.strip() for name in str(value).split(";") if name.strip()]


def _build_name_record(group: pd.DataFrame) -> dict:
    """Build one ISO's name record from its member_states rows.

    An ISO normally has one current row (`record_type == 'ms'`) plus a former-state row per
    historical name — MMR is Myanmar + Burma. Ten ISOs are `fs`-only: states that dissolved (SUN,
    CSK, YUG, DDR, ...), which get a year range appended to their display name.
    """
    current = group[group["record_type"] == "ms"]
    if not current.empty:
        # Every current member state's final coverage period is open, so `record_type` alone
        # identifies the canonical row — no date arithmetic needed to tell "still a member".
        canonical = current.iloc[0]
        year_range = None
    else:
        # Retired ISO: the most recently-ended row holds the name worth showing, and the range
        # spans every period the entity existed (YMD has two such rows: Southern then Democratic
        # Yemen).
        latest_end = group["coverage_periods"].apply(
            lambda cp: max(_coverage_years(cp)[1], default=0)
        )
        canonical = group.loc[latest_end.idxmax()]
        starts = [y for cp in group["coverage_periods"] for y in _coverage_years(cp)[0]]
        ends = [y for cp in group["coverage_periods"] for y in _coverage_years(cp)[1]]
        year_range = (min(starts), max(ends)) if starts and ends else None

    en_name = canonical["name_en"]
    names: dict[str, str] = {"en": en_name}
    for lang in SUPPORTED_LANGS:
        if lang == "en":
            continue
        val = canonical.get(_LANG_COL[lang])
        names[lang] = val if isinstance(val, str) and val.strip() else en_name

    # Display aliases: the group's other rows' own names. These are real predecessor entities
    # ("Burma" -> "Myanmar"), not variant spellings.
    display_aliases: list[str] = []
    for name in group["name_en"]:
        if isinstance(name, str) and name != en_name and name not in display_aliases:
            display_aliases.append(name)

    # Search aliases: display aliases plus every `other_names` entry (variant spellings,
    # translations, abbreviations like "BRD"/"Soviet Union") so search stays permissive.
    search_aliases: list[str] = list(display_aliases)
    for other in group["other_names"]:
        for name in _split_names(other):
            if name != en_name and name not in search_aliases:
                search_aliases.append(name)

    return {
        "names": names,
        "display_aliases": display_aliases,
        "search_aliases": search_aliases,
        "year_range": year_range,
    }


def _build_name_index(member_states_df: pd.DataFrame) -> dict[str, dict]:
    """Build per-ISO name records keyed by ISO3.

    Each record contains:
        names:           dict[lang, str]        - canonical name per language (English fallback)
        display_aliases: list[str]              - real historical names (predecessor entities) for display
        search_aliases:  list[str]              - display aliases plus variant spellings, translations,
                                                  abbreviations from the 'other_names' column
        year_range:      tuple[int, int] | None - set only for retired ISOs (no current row)
    """
    index: dict[str, dict] = {}

    for iso, group in member_states_df.groupby("iso_code"):
        if not isinstance(iso, str) or not iso.strip():
            continue
        index[iso] = _build_name_record(group)

    # Merge voting-only codes (see _LEGACY_VOTING_CODES above). Each resolves to a single fs row,
    # which has no current row and therefore picks up a year range automatically.
    for legacy_iso, payload in _LEGACY_VOTING_CODES.items():
        if legacy_iso in index:
            continue
        successor_iso, name_en = payload["source"]
        rows = member_states_df[
            (member_states_df["iso_code"] == successor_iso)
            & (member_states_df["name_en"] == name_en)
        ]
        if rows.empty:
            print(
                f"Legacy voting code {legacy_iso}: no '{name_en}' row found under "
                f"{successor_iso}; votes cast under {legacy_iso} will show the bare code"
            )
            continue
        record = _build_name_record(rows)
        record["names"].update(payload.get("names", {}))
        index[legacy_iso] = record

    return index


_NAME_INDEX: dict[str, dict] = _build_name_index(repo.get_data()["member_states"])


def get_country_name(iso3_code: str | None, lang: str = "en") -> str:
    """Return the country name for an ISO3 code in the requested language.

    Falls back to English when the requested language has no translation.
    Appends a (start-end) year range for retired ISOs (e.g. SUN, CSK).
    """
    if not iso3_code:
        return "Unknown"
    rec = _NAME_INDEX.get(iso3_code)
    if not rec:
        return iso3_code
    name = rec["names"].get(lang) or rec["names"]["en"]
    if rec["year_range"]:
        start, end = rec["year_range"]
        return f"{name} ({start}–{end})"
    return name


def get_country_display_name(iso3_code: str, lang: str = "en") -> str:
    """Country name with historical aliases in brackets, e.g. 'Myanmar (historical: Burma)'."""
    base_name = get_country_name(iso3_code, lang=lang)
    rec = _NAME_INDEX.get(iso3_code)
    if not rec or not rec["display_aliases"]:
        return base_name
    return f"{base_name} (historical: {', '.join(rec['display_aliases'])})"


def get_country_search_terms(iso3_code: str, lang: str = "en") -> str:
    """Search string including the localised name, the English name, and all variant aliases."""
    rec = _NAME_INDEX.get(iso3_code)
    if not rec:
        return iso3_code
    terms = {rec["names"]["en"], rec["names"].get(lang, rec["names"]["en"])}
    terms.update(rec["search_aliases"])
    return " ".join(t for t in terms if t)


# Build M49 region tree for AntdTreeSelect
def get_region_tree_data() -> list[dict]:
    """Build nested treeData for AntdTreeSelect from M49 regional groupings CSV.

    Returns a nested hierarchy: World → Region → Sub-region → [Intermediate Region] → Country.
    Each node has: key, title, value, and children (list of child nodes).
    """
    from .features.country_utils import _load_m49

    df = _load_m49()

    # Collect nodes by code for deduplication and parent lookup
    regions: dict[str, dict] = {}
    sub_regions: dict[str, dict] = {}
    inter_regions: dict[str, dict] = {}
    countries: dict[str, dict] = {}

    # Track parent relationships
    region_to_world: dict[str, bool] = {}
    sub_to_region: dict[str, str] = {}
    inter_to_sub: dict[str, str] = {}
    country_to_parent: dict[str, str] = {}

    for _, row in df.iterrows():
        iso3 = row.get("ISO-alpha3 Code")
        if not isinstance(iso3, str) or not iso3.strip():
            continue

        # Region
        r_code = str(int(float(row["Region Code"]))).zfill(3)
        if r_code not in regions:
            regions[r_code] = {"key": r_code, "title": row["Region Name"], "value": r_code}
            region_to_world[r_code] = True

        # Sub-region
        sr_code = str(int(float(row["Sub-region Code"]))).zfill(3)
        if sr_code not in sub_regions:
            sub_regions[sr_code] = {
                "key": sr_code,
                "title": row["Sub-region Name"],
                "value": sr_code,
            }
            sub_to_region[sr_code] = r_code

        # Intermediate region (optional)
        parent_code = sr_code
        ir_raw = row.get("Intermediate Region Code")
        if isinstance(ir_raw, float) and not pd.isna(ir_raw):
            ir_code = str(int(ir_raw)).zfill(3)
            if ir_code not in inter_regions:
                inter_regions[ir_code] = {
                    "key": ir_code,
                    "title": row["Intermediate Region Name"],
                    "value": ir_code,
                }
                inter_to_sub[ir_code] = sr_code
            parent_code = ir_code

        # Country leaf
        if iso3 not in countries:
            countries[iso3] = {"key": iso3, "title": get_country_display_name(iso3), "value": iso3}
            country_to_parent[iso3] = parent_code

    # Build tree bottom-up: attach countries to their parents
    for iso3, node in countries.items():
        parent = country_to_parent[iso3]
        if parent in inter_regions:
            inter_regions[parent].setdefault("children", []).append(node)
        elif parent in sub_regions:
            sub_regions[parent].setdefault("children", []).append(node)

    # Attach intermediate regions to sub-regions
    for ir_code, node in inter_regions.items():
        sr_code = inter_to_sub[ir_code]
        sub_regions[sr_code].setdefault("children", []).append(node)

    # Attach sub-regions to regions
    for sr_code, node in sub_regions.items():
        r_code = sub_to_region[sr_code]
        regions[r_code].setdefault("children", []).append(node)

    # Use joining_dates.csv as the authoritative source for countries with voting data
    from .features.country_utils import _load_joining_dates

    valid = set(_load_joining_dates()["country"].tolist())

    for parent_dict in [inter_regions, sub_regions]:
        for code, node in parent_dict.items():
            if "children" in node:
                node["children"] = [
                    c
                    for c in node["children"]
                    if "children" in c or c["value"] in valid  # keep groups, filter leaves
                ]

    # Remove empty intermediate/sub-region/region nodes (no children left)
    for ir_code, node in list(inter_regions.items()):
        if not node.get("children"):
            sr_code = inter_to_sub[ir_code]
            sub_regions[sr_code]["children"] = [
                c for c in sub_regions[sr_code].get("children", []) if c["key"] != ir_code
            ]

    for sr_code, node in list(sub_regions.items()):
        if not node.get("children"):
            r_code = sub_to_region[sr_code]
            regions[r_code]["children"] = [
                c for c in regions[r_code].get("children", []) if c["key"] != sr_code
            ]

    # Build world root with regions as children
    world = {
        "key": "001",
        "title": "World",
        "value": "001",
        "children": [r for r in regions.values() if r.get("children")],
    }

    # Add historical countries (voted but no longer in M49)
    m49_codes = set(countries.keys())
    historical_codes = sorted(valid - m49_codes)
    historical_children = []
    for code in historical_codes:
        name = get_country_name(code)
        historical_children.append({"key": code, "title": name, "value": code})

    if historical_children:
        historical_node = {
            "key": "historical",
            "title": "Historical States",
            "value": "historical",
            "children": historical_children,
        }
        print(
            f"Region tree: {len(m49_codes)} current + {len(historical_children)} historical countries"
        )
        return [world, historical_node]

    print(f"Region tree: {len(m49_codes)} current countries, no historical found")
    return [world]


REGION_TREE_DATA = get_region_tree_data()

# Top-level subjects: the thesaurus's 18 domains (`node_type == 'scheme'`).
TOP_LEVEL_SUBJECTS = set(
    repo.get_data()["subject"].loc[lambda df: df["node_type"] == "scheme", "subject_id"].tolist()
)

# Map subject IDs to labels TODO: If we want to add multiple languages just add the other languages here
SUBJECT_ID_TO_LABEL_MAP = {
    row["subject_id"]: row["label_en"] for _, row in repo.get_data()["subject"].iterrows()
}


def available_subjects() -> list[dict[str, Any]]:
    data = repo.get_data()

    # data["subject"]: pd.DataFrame, totally 7341 subjects
    subject_options_list = data["subject"].to_dict("records")

    subject_options = [
        {"label": r["label_en"], "value": r["subject_id"]} for r in subject_options_list
    ]
    # ! not sure about sequence of subjects
    subject_options = subject_options[::-1]

    return subject_options


def get_subject_tree_data() -> list[dict]:
    """Build nested treeData for AntdTreeSelect from the thesaurus hierarchy.

    Uses the broader_table (direct SKOS.broader / hasTopConcept pairs only),
    not the closure_table, so the source reflects the actual declared hierarchy.
    A globally-visited set ensures each node appears exactly once even if the
    source data contains cycles.
    """
    _data = repo.get_data()
    subject_df = _data["subject"]
    broader_df = _data["broader"]

    # Build label lookup (subject_id -> English label)
    label_map: dict[str, str] = {
        row["subject_id"]: row["label_en"]
        for _, row in subject_df.iterrows()
        if row.get("label_en") and isinstance(row.get("label_en"), str)
    }

    valid_subjects = set(subject_df["subject_id"].tolist())

    # Build direct parent -> children map from the broader table
    direct_children: dict[str, list[str]] = {}
    for _, row in broader_df.iterrows():
        direct_children.setdefault(row["parent_id"], []).append(row["child_id"])

    # Global visited set: each node is placed in the tree exactly once.
    # This naturally breaks any cycles present in the source SKOS data.
    visited: set[str] = set()

    def build_node(subject_id: str) -> dict | None:
        if subject_id in visited:
            return None
        label = label_map.get(subject_id)
        if not label:
            return None
        visited.add(subject_id)
        node: dict = {"key": subject_id, "title": label, "value": subject_id}
        children_ids = direct_children.get(subject_id, [])
        children = []
        for child_id in sorted(children_ids, key=lambda x: label_map.get(x, "")):
            if child_id not in visited and child_id in valid_subjects:
                child_node = build_node(child_id)
                if child_node:
                    children.append(child_node)
        if children:
            node["children"] = children
        return node

    scheme_nodes = []
    for scheme_id in sorted(TOP_LEVEL_SUBJECTS, key=lambda x: label_map.get(x, "")):
        node = build_node(scheme_id)
        if node:
            scheme_nodes.append(node)

    no_subject_node = {
        "key": "__no_subject__",
        "title": "No Subject",
        "value": "__no_subject__",
    }

    root = {
        "key": "__all_subjects__",
        "title": "All Subjects",
        "value": "__all_subjects__",
        "children": [no_subject_node] + scheme_nodes,
    }
    return [root]


SUBJECT_TREE_DATA = get_subject_tree_data()


def get_earliest_data_date():
    # 1946-01-26
    data = repo.get_data()

    return pd.to_datetime(data["resolution"]["date"].min())


def get_latest_data_date():
    # 2025-09-05
    data = repo.get_data()

    return pd.to_datetime(data["resolution"]["date"].max())


def get_earliest_year():
    return get_earliest_data_date().year


def get_latest_year():
    return get_latest_data_date().year


# ---------------------------------------------------------------------------
# Periodic reload
#
# The ingestion jobs refresh Postgres twice a day; this lets a running process pick that up
# without a deploy. See T12 in plans/app_postgres_migration_plan.md.
# ---------------------------------------------------------------------------


def _rebuild_derived_state() -> None:
    """Rebuild everything in this module that is derived from the repository.

    `query_engine` is *not* rebuilt — its snapshot is swapped instead, because three features
    close over the engine instance when their callbacks are registered. Everything here is a
    module global that functions read at call time (or that page layouts read per view), so
    rebinding is enough.
    """
    global _NAME_INDEX, REGION_TREE_DATA, TOP_LEVEL_SUBJECTS, SUBJECT_ID_TO_LABEL_MAP
    global SUBJECT_TREE_DATA, available_countries

    available_countries = query_engine.get_available_countries()
    _NAME_INDEX = _build_name_index(repo.get_data()["member_states"])
    subject_df = repo.get_data()["subject"]
    TOP_LEVEL_SUBJECTS = set(
        subject_df.loc[lambda df: df["node_type"] == "scheme", "subject_id"].tolist()
    )
    SUBJECT_ID_TO_LABEL_MAP = {
        row["subject_id"]: row["label_en"] for _, row in subject_df.iterrows()
    }
    # Both trees are read inside `filters.layout()` / page layout functions, so a rebuilt tree
    # reaches the UI on the next page view.
    REGION_TREE_DATA = get_region_tree_data()
    SUBJECT_TREE_DATA = get_subject_tree_data()


def _invalidate_feature_caches() -> None:
    """Drop every cache a feature module derives from the repository.

    Each module exposes its own `invalidate()`. They are looked up in `sys.modules` rather than
    imported, for two reasons: these modules import `app.data`, so a top-level import here would
    be circular; and importing a *page* module has side effects — `trends_page` calls
    `register_page()` at import time, which raises unless a Dash app already exists. A module
    that was never imported holds no cache to clear, so skipping it is also the correct answer.

    If you memoise anything derived from `query_engine` or `repo` — an `lru_cache`, a lazily
    built index — give its module an `invalidate()` and list it here, or it will serve pre-reload
    data for the life of the process. Caches derived only from the static CSV assets
    (`country_utils`) do not belong here; a reload cannot change them.
    """
    module_names = (
        f"{__package__}.features.wordcloud_interactive",
        f"{__package__}.features.recent_resolutions_panel",
        f"{__package__}.features.general_stats_panel",
        f"{__package__}.pages.trends_page",
    )
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is not None:
            module.invalidate()


def reload_if_stale() -> bool:
    """Rebuild and swap in fresh data if a newer successful ingestion run exists.

    Returns whether a reload happened. Raises nothing the caller must handle: a failure leaves
    the previous data in place, and the reloader logs it and retries on the next tick.
    """
    global repo

    live_marker = reloader.read_marker()
    loaded_marker = query_engine.snapshot.source_marker
    if not reloader.is_stale(loaded_marker, live_marker):
        repo.logger.debug("Data still current (marker %s); no reload", loaded_marker)
        return False

    repo.logger.info("Newer ingestion detected (%s > %s); reloading", live_marker, loaded_marker)
    started = time.monotonic()

    # Build the whole new state before touching anything that is being served. If this raises,
    # the swap below never happens and the old data keeps serving.
    new_repo = DataRepository(config_path=str(_CONFIG_PATH))
    snapshot = DataSnapshot.from_repo(new_repo)

    query_engine.swap(snapshot)
    repo = new_repo
    _rebuild_derived_state()

    _invalidate_feature_caches()

    repo.logger.info(
        "Reload complete in %.1fs: %s", time.monotonic() - started, snapshot.describe()
    )
    return True


def start_reloader() -> bool:
    """Start this process's reload poller. Safe to call repeatedly; see reloader.start()."""
    return reloader.start(reload_if_stale)
