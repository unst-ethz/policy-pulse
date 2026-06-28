from pathlib import Path
from typing import Any

import pandas as pd

from .un_data_stream import DataRepository, ResolutionQueryEngine

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
repo = DataRepository(config_path=str(_PROJECT_ROOT / "config" / "data_sources.yaml"))
query_engine = ResolutionQueryEngine(repo=repo)

available_countries = query_engine.get_available_countries()

# Supported UI languages and their column names in the member-states authority table.
SUPPORTED_LANGS = ("en", "fr", "es", "ar", "zh", "ru")
_LANG_COL = {
    "en": "Member State",
    "fr": "French",
    "es": "Spanish",
    "ar": "Arabic",
    "zh": "Chinese",
    "ru": "Russian",
}

# Legacy ISO codes that still appear in UN voting data but have been merged into
# their successor codes in the authority table. Once GA/SC voting data switches
# to the canonical codes (GER->DEU, SCG->SRB), these fallback entries can go.
_LEGACY_VOTING_CODES: dict[str, dict[str, Any]] = {
    "GER": {  # West Germany; voting records 1973-1990
        "en": "Germany, Federal Republic of",
        "fr": "République fédérale d'Allemagne",
        "es": "República Federal de Alemania",
        "ru": "Федеративная Республика Германия",
        "_aliases": ["West Germany", "BRD"],
        "_year_range": (1973, 1990),
    },
    "SCG": {  # Serbia and Montenegro; voting records 2003-2006
        "en": "Serbia and Montenegro",
        "fr": "Serbie-et-Monténégro",
        "es": "Serbia y Montenegro",
        "ru": "Сербия и Черногория",
        "_aliases": ["Federal Republic of Yugoslavia"],
        "_year_range": (2003, 2006),
    },
}


def _parse_year(date_str) -> int | None:
    """Extract the earliest 4-digit year from a possibly comma-separated date string."""
    if pd.isna(date_str):
        return None
    for part in str(date_str).split(","):
        head = part.strip()[:4]
        if head.isdigit():
            return int(head)
    return None


def _parse_years(date_str) -> list[int]:
    """Return all 4-digit years found in a possibly comma-separated date string."""
    if pd.isna(date_str):
        return []
    years: list[int] = []
    for part in str(date_str).split(","):
        head = part.strip()[:4]
        if head.isdigit():
            years.append(int(head))
    return years


def _build_name_index(member_states_df: pd.DataFrame) -> dict[str, dict]:
    """Build per-ISO name records keyed by ISO3.

    Each record contains:
        names:           dict[lang, str]        - canonical name per language (English fallback)
        display_aliases: list[str]              - real historical names (predecessor entities) for display
        search_aliases:  list[str]              - display aliases plus variant spellings, translations,
                                                  abbreviations from the 'Other Names' column
        year_range:      tuple[int, int] | None - set only for retired ISOs (no active row)
    """
    # A row is "currently active" if either:
    #  (a) End date is NaN, OR
    #  (b) it encodes multi-period membership where the final period is open
    #      (more Start segments than End segments in the comma-separated strings,
    #      e.g. KHM Cambodia: starts 1955/1975/1990, ends 1970/1976 -> third period is open).
    def _is_active(row) -> bool:
        if pd.isna(row["End date"]):
            return True
        return len(_parse_years(row["Start date"])) > len(_parse_years(row["End date"]))

    index: dict[str, dict] = {}

    for iso, group in member_states_df.groupby("ISO Code"):
        if not isinstance(iso, str) or not iso.strip():
            continue

        active_mask = group.apply(_is_active, axis=1)
        active = group[active_mask]
        if len(active) >= 1:
            canonical = active.iloc[0]
            year_range = None
        else:
            # Retired ISO: pick row with latest end year as canonical, attach year range
            with_end = group.assign(_end=group["End date"].apply(_parse_year))
            canonical = with_end.sort_values("_end").iloc[-1]
            start_years = [y for s in group["Start date"] for y in _parse_years(s)]
            end_years = [y for s in group["End date"] for y in _parse_years(s)]
            year_range = (
                (min(start_years), max(end_years)) if start_years and end_years else None
            )

        en_name = canonical["Member State"]
        names: dict[str, str] = {"en": en_name}
        for lang in SUPPORTED_LANGS:
            if lang == "en":
                continue
            val = canonical.get(_LANG_COL[lang])
            names[lang] = val if isinstance(val, str) and val.strip() else en_name

        # Display aliases: only the Member State names of non-canonical rows. These are the
        # real predecessor entities (e.g. "Burma" -> "Myanmar"), not variant spellings.
        display_aliases: list[str] = []
        for _, row in group.iterrows():
            other_name = row["Member State"]
            if (
                other_name != en_name
                and isinstance(other_name, str)
                and other_name not in display_aliases
            ):
                display_aliases.append(other_name)

        # Search aliases: display aliases plus everything in 'Other Names' (variant spellings,
        # translations, abbreviations like "FYROM"/"BRD"/"Soviet Union") so search remains permissive.
        search_aliases: list[str] = list(display_aliases)
        for other in group["Other Names"].dropna():
            token = str(other).strip()
            if token and token != en_name and token not in search_aliases:
                search_aliases.append(token)

        index[iso] = {
            "names": names,
            "display_aliases": display_aliases,
            "search_aliases": search_aliases,
            "year_range": year_range,
        }

    # Merge legacy voting-only codes (see _LEGACY_VOTING_CODES above).
    for iso, payload in _LEGACY_VOTING_CODES.items():
        if iso in index:
            continue
        names = {lang: payload.get(lang, payload["en"]) for lang in SUPPORTED_LANGS}
        legacy_aliases = list(payload.get("_aliases", []))
        index[iso] = {
            "names": names,
            "display_aliases": legacy_aliases,
            "search_aliases": legacy_aliases,
            "year_range": payload.get("_year_range"),
        }

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
            sub_regions[sr_code] = {"key": sr_code, "title": row["Sub-region Name"], "value": sr_code}
            sub_to_region[sr_code] = r_code

        # Intermediate region (optional)
        parent_code = sr_code
        ir_raw = row.get("Intermediate Region Code")
        if isinstance(ir_raw, float) and not pd.isna(ir_raw):
            ir_code = str(int(ir_raw)).zfill(3)
            if ir_code not in inter_regions:
                inter_regions[ir_code] = {"key": ir_code, "title": row["Intermediate Region Name"], "value": ir_code}
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
                    c for c in node["children"]
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
        print(f"Region tree: {len(m49_codes)} current + {len(historical_children)} historical countries")
        return [world, historical_node]

    print(f"Region tree: {len(m49_codes)} current countries, no historical found")
    return [world]


REGION_TREE_DATA = get_region_tree_data()

from .features.country_utils import get_country_region, get_country_subregion

# Top level subjects (level 0 in the hierarchy)
TOP_LEVEL_SUBJECTS = {
    'http://metadata.un.org/thesaurus/10', 
    'http://metadata.un.org/thesaurus/09', 
    'http://metadata.un.org/thesaurus/16', 
    'http://metadata.un.org/thesaurus/00', 
    'http://metadata.un.org/thesaurus/07', 
    'http://metadata.un.org/thesaurus/04', 
    'http://metadata.un.org/thesaurus/06', 
    'http://metadata.un.org/thesaurus/15', 
    'http://metadata.un.org/thesaurus/05', 
    'http://metadata.un.org/thesaurus/03', 
    'http://metadata.un.org/thesaurus/17', 
    'http://metadata.un.org/thesaurus/11', 
    'http://metadata.un.org/thesaurus/12', 
    'http://metadata.un.org/thesaurus/13', 
    'http://metadata.un.org/thesaurus/14', 
    'http://metadata.un.org/thesaurus/18', 
    'http://metadata.un.org/thesaurus/08', 
    'http://metadata.un.org/thesaurus/01', 
    'http://metadata.un.org/thesaurus/02'
}

# Map subject IDs to labels TODO: If we want to add multiple languages just add the other languages here
SUBJECT_ID_TO_LABEL_MAP = {row["subject_id"]: row["label_en"] for _, row in repo.get_data()["subject"].iterrows()}

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
