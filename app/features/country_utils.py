"""Shared country-level lookup utilities."""

from pathlib import Path

import pandas as pd

joining_dates = pd.read_csv(
    Path(__file__).resolve().parent.parent / "assets" / "joining_dates.csv",
    parse_dates=["min_date", "max_date"],
)


def get_un_membership_years(country_alpha3: str) -> tuple[int, int] | None:
    """Return (first_year, last_year) of UN membership for a country.

    For countries with multiple membership periods, returns the widest range
    (earliest join to latest leave/current year). Returns None if not found.
    """
    rows = joining_dates[joining_dates["country"] == country_alpha3]
    if rows.empty:
        return None
    return rows["min_date"].min().year, rows["max_date"].max().year
