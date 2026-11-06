"""Thesaurus domains as a whole: how many resolutions each has, per year.

A resolution belongs to a domain when any of its subjects falls under it, so it can count towards
several domains; per domain it counts once. Sponsors are `sponsorship.subject_sponsors`; votes
come from the `/multilateral` and `/resolutions` endpoints with a `subject` filter.
"""

from typing import Iterable

import pandas as pd

from .profile import RECORDED_VOTE
from .snapshot import DataSnapshot


def domain_pairs(
    snapshot: DataSnapshot, ids: Iterable[str], domains: Iterable[str]
) -> pd.DataFrame:
    """Distinct (undl_id, subject_id) pairs: resolution `undl_id` is in domain `subject_id`."""
    links = snapshot.resolution_subject_table
    links = links[links["undl_id"].isin(pd.Index(ids))]
    closure = snapshot.closure_table
    closure = closure[closure["ancestor_id"].isin(list(domains))]
    joined = links.merge(closure, left_on="subject_id", right_on="descendant_id")
    return (
        joined[["undl_id", "ancestor_id"]]
        .drop_duplicates()
        .rename(columns={"ancestor_id": "subject_id"})
    )


def domain_years(
    period: pd.DataFrame, snapshot: DataSnapshot, domains: Iterable[str]
) -> pd.DataFrame:
    """subject_id, year, resolutions, recorded: per domain and year, over `period`'s resolutions.

    Years without a resolution in the domain are absent.
    """
    pairs = domain_pairs(snapshot, period["undl_id"], domains)
    rows = pairs.merge(period[["undl_id", "date", "modality"]], on="undl_id")
    return (
        pd.DataFrame(
            {
                "subject_id": rows["subject_id"],
                "year": rows["date"].dt.year,
                "resolutions": 1,
                "recorded": rows["modality"] == RECORDED_VOTE,
            }
        )
        .groupby(["subject_id", "year"], as_index=False)
        .sum()
        .astype({"year": int, "resolutions": int, "recorded": int})
    )
