import os
import sys
from typing import Any
import pandas as pd
import pycountry

from .un_data_stream import DataRepository, ResolutionQueryEngine

repo = DataRepository(config_path="config/data_sources.yaml")
query_engine = ResolutionQueryEngine(repo=repo)

available_countries = query_engine.get_available_countries()

# Build name lookups from successor_states.csv
_SUCCESSOR_STATES_PATH = os.path.join(os.path.dirname(__file__), "assets", "successor_states.csv")
_successor_df = pd.read_csv(_SUCCESSOR_STATES_PATH, parse_dates=["start_date", "end_date"])

# Historical codes that pycountry can't resolve (truly retired states like SUN, DDR, CSK)
# Maps code → (display_name, start_year, end_year)
_HISTORICAL_NAME_MAP: dict[str, tuple[str, int, int]] = {}

# Same-code name changes (Burma→Myanmar, Dahomey→Benin, etc.)
# Maps current code → list of historical names for alias search
_SAME_CODE_ALIASES: dict[str, list[str]] = {}

for _ms_code, _group in _successor_df[_successor_df["status"] == "fs"].groupby("ms_code"):
    if pycountry.countries.get(alpha_3=_ms_code) is None:
        # Truly retired code — use the last (most recent) name as display name
        _last = _group.sort_values("start_date").iloc[-1]
        _start_year = int(_group["start_date"].min().year)
        _end_year = int(_group["end_date"].dropna().max().year) if _group["end_date"].notna().any() else None
        if _end_year:
            _HISTORICAL_NAME_MAP[_ms_code] = (_last["ms_name"], _start_year, _end_year)
    else:
        # Same ISO3 code persists — collect historical names as search aliases
        _aliases = _group["ms_name"].tolist()
        if _aliases:
            _SAME_CODE_ALIASES[_ms_code] = _aliases


def get_country_name(
    iso3_code: str | None,
) -> str:  # we need to support multiple languages at some point
    """Get English country name from ISO3 code, with year range for historical states."""
    if iso3_code is None:
        return "Unknown"
    try:
        country = pycountry.countries.get(alpha_3=iso3_code)
        if country:
            return country.name
        if iso3_code in _HISTORICAL_NAME_MAP:
            name, start_year, end_year = _HISTORICAL_NAME_MAP[iso3_code]
            return f"{name} ({start_year}\u2013{end_year})"
        return iso3_code
    except:
        return iso3_code


def get_country_display_name(iso3_code: str) -> str:
    """Get country name with historical aliases in brackets, e.g. 'Myanmar (Burma)'."""
    base_name = get_country_name(iso3_code)
    aliases = _SAME_CODE_ALIASES.get(iso3_code, [])
    # Filter out aliases that match the current name
    historical = [a for a in aliases if a != base_name]
    if historical:
        return f"{base_name} (historical: {', '.join(historical)})"
    return base_name


def get_country_search_terms(iso3_code: str) -> str:
    """Return search string including historical name aliases for same-ISO3 name changes."""
    base_name = get_country_name(iso3_code)
    aliases = _SAME_CODE_ALIASES.get(iso3_code, [])
    if aliases:
        return " ".join([base_name] + aliases)
    return base_name


# Build M49 region tree for AntdTreeSelect
_M49_PATH = os.path.join(os.path.dirname(__file__), "assets", "m49_regional_groupings.csv")


def get_region_tree_data() -> list[dict]:
    """Build nested treeData for AntdTreeSelect from M49 regional groupings CSV.

    Returns a nested hierarchy: World → Region → Sub-region → [Intermediate Region] → Country.
    Each node has: key, title, value, and children (list of child nodes).
    """
    df = pd.read_csv(_M49_PATH, sep=";")

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
            countries[iso3] = {"key": iso3, "title": get_country_name(iso3), "value": iso3}
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
    _joining_path = os.path.join(os.path.dirname(__file__), "assets", "joining_dates.csv")
    _voting_df = pd.read_csv(_joining_path)
    valid = set(_voting_df["country"].tolist())

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
