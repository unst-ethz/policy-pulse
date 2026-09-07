from pathlib import Path
from typing import Any

import pandas as pd

from .un_data_stream import DataRepository, ResolutionQueryEngine

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
repo = DataRepository(config_path=str(_PROJECT_ROOT / "config" / "data_sources.yaml"))
query_engine = ResolutionQueryEngine(repo=repo)

available_countries = query_engine.get_available_countries()

# Country authority semantics are shared with the HTTP API.
from .un_data_stream.countries import CountryCatalog, SUPPORTED_LANGS

_country_catalog = CountryCatalog(repo.get_data()["member_states"])
_NAME_INDEX = _country_catalog.index
get_country_name = _country_catalog.get_country_name
get_country_display_name = _country_catalog.get_country_display_name
get_country_search_terms = _country_catalog.get_country_search_terms


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
