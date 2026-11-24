"""M49 regional groupings per country.

Membership and voting-activity dates live in `app.data`, which reads them from MongoDB.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

_M49_CSV = Path(__file__).resolve().parent.parent / "config" / "m49_regional_groupings.csv"


@lru_cache(maxsize=None)
def _load_m49() -> pd.DataFrame:
    return pd.read_csv(_M49_CSV, sep=";")


@lru_cache(maxsize=None)
def _country_to_region() -> dict[str, str]:
    return {
        row["ISO-alpha3 Code"].strip(): row["Region Name"].strip()
        for _, row in _load_m49().iterrows()
        if isinstance(row.get("ISO-alpha3 Code"), str) and isinstance(row.get("Region Name"), str)
    }


@lru_cache(maxsize=None)
def _country_to_subregion() -> dict[str, str]:
    return {
        row["ISO-alpha3 Code"].strip(): row["Sub-region Name"].strip()
        for _, row in _load_m49().iterrows()
        if isinstance(row.get("ISO-alpha3 Code"), str)
        and isinstance(row.get("Sub-region Name"), str)
    }


def get_country_region(iso3: str) -> str:
    """Return M49 top-level region name for a country, or 'Other' if not found."""
    return _country_to_region().get(iso3, "Other")


def get_country_subregion(iso3: str) -> str | None:
    """Return M49 sub-region name for a country, or None if not found."""
    return _country_to_subregion().get(iso3)
