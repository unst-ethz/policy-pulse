import os
import sys
from typing import Any
import pandas as pd
import pycountry

from .un_data_stream import DataRepository, ResolutionQueryEngine

repo = DataRepository(config_path="config/data_sources.yaml")
query_engine = ResolutionQueryEngine(repo=repo)

available_countries = query_engine.get_available_countries()

def get_country_name(
    iso3_code: str | None,
) -> str:  # we need to support multiple languages at some point
    """Get English country name from ISO3 code."""
    if iso3_code is None:
        return "Unknown"
    try:
        country = pycountry.countries.get(alpha_3=iso3_code)
        return country.name if country else iso3_code
    except:
        return iso3_code

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
