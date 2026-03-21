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


def get_country_search_terms(iso3_code: str) -> str:
    """Return search string including historical name aliases for same-ISO3 name changes."""
    base_name = get_country_name(iso3_code)
    aliases = _SAME_CODE_ALIASES.get(iso3_code, [])
    if aliases:
        return " ".join([base_name] + aliases)
    return base_name

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
