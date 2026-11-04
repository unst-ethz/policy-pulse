"""Who sponsored a resolution, and a country's sponsorship record.

Sponsorship is read off the draft a resolution was adopted from (see `undl-ingest`), so coverage
follows the catalogue: ~90% from 2000 on, little before. Everything here keeps "unknown" (no
sponsorship data) apart from "not sponsored", and reports how many resolutions had data.

Country views count states only (`country_code` set): UN bodies such as committee chairs and
groups such as the G77 are sponsors of a resolution but not of a country's record.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .profile import RECORDED_VOTE
from .query_engine import ResolutionQueryEngine
from .snapshot import DataSnapshot

ORIGINAL = "original"


def known_ids(snapshot: DataSnapshot) -> pd.Index:
    """Resolutions with sponsorship data (at least one sponsor read off the draft)."""
    table = snapshot.resolution_table
    if "sponsor_count" not in table.columns:
        return pd.Index([])
    return pd.Index(table.loc[table["sponsor_count"].notna(), "undl_id"])


def sponsored_ids(snapshot: DataSnapshot, country: str, original_only: bool = False) -> pd.Index:
    """Resolutions `country` sponsored, optionally only as an original sponsor."""
    sponsors = snapshot.sponsor_table
    mask = sponsors["country_code"] == country
    if original_only:
        mask &= sponsors["sponsor_kind"] == ORIGINAL
    return pd.Index(sponsors.loc[mask, "undl_id"].unique())


def resolution_sponsors(snapshot: DataSnapshot, undl_id: str) -> tuple[pd.DataFrame, str | None]:
    """(sponsor rows, originals first, catalogue order kept; the placeholder note if unknown)."""
    rows = snapshot.sponsor_table[snapshot.sponsor_table["undl_id"] == undl_id]
    named = rows[rows["sponsor"].notna()]
    note = None if len(named) or rows.empty else rows["note"].iloc[0]
    named = named.assign(_late=named["sponsor_kind"] != ORIGINAL).sort_values(
        "_late", kind="stable"
    )
    return named.drop(columns="_late"), note


@dataclass(frozen=True)
class CountrySponsorship:
    resolutions: int  # in the period
    with_data: int  # of those, with sponsorship data
    sponsored: int
    original: int
    # year, with_data, original, additional — one row per year with any resolution
    per_year: pd.DataFrame
    # subject_id -> resolutions sponsored in that domain (descendants included), most first
    by_subject: pd.Series
    # per other country, its votes on the recorded-vote resolutions `country` sponsored:
    # country, yes, no, abstain — the ones voting against it most first
    reception: pd.DataFrame
    # sponsored resolutions on which `country` itself abstained or voted no, newest first
    not_yes: pd.DataFrame


def country_sponsorship(
    engine: ResolutionQueryEngine,
    country: str,
    start_date: str | None,
    end_date: str | None,
    domains: Iterable[str],
) -> CountrySponsorship:
    """`country`'s sponsorship over [start_date, end_date] (inclusive, 'YYYY-MM-DD')."""
    snapshot = engine.snapshot
    table = snapshot.resolution_table
    in_period = pd.Series(True, index=table.index)
    if start_date:
        in_period &= table["date"] >= pd.Timestamp(start_date)
    if end_date:
        in_period &= table["date"] <= pd.Timestamp(end_date)
    period = table[in_period]

    sponsors = snapshot.sponsor_table
    own = sponsors[
        (sponsors["country_code"] == country) & sponsors["undl_id"].isin(period["undl_id"])
    ]
    original_ids = set(own.loc[own["sponsor_kind"] == ORIGINAL, "undl_id"])
    is_sponsored = period["undl_id"].isin(own["undl_id"])

    years = period["date"].dt.year
    per_year = (
        pd.DataFrame(
            {
                "year": years,
                "with_data": period["sponsor_count"].notna(),
                "original": period["undl_id"].isin(original_ids),
                "additional": is_sponsored & ~period["undl_id"].isin(original_ids),
            }
        )
        .groupby("year", as_index=False)
        .sum()
        .astype(int)
    )

    sponsored = period[is_sponsored]
    return CountrySponsorship(
        resolutions=len(period),
        with_data=int(period["sponsor_count"].notna().sum()),
        sponsored=len(sponsored),
        original=len(original_ids),
        per_year=per_year,
        by_subject=_by_domain(snapshot, sponsored["undl_id"], domains),
        reception=_reception(snapshot, sponsored, country),
        not_yes=(
            sponsored[sponsored[country].isin(["N", "A"])].sort_values("date", ascending=False)
            if country in sponsored.columns
            else sponsored.iloc[0:0]
        ),
    )


def _by_domain(snapshot: DataSnapshot, ids: pd.Series, domains: Iterable[str]) -> pd.Series:
    links = snapshot.resolution_subject_table
    links = links[links["undl_id"].isin(ids)]
    closure = snapshot.closure_table
    closure = closure[closure["ancestor_id"].isin(list(domains))]
    joined = links.merge(closure, left_on="subject_id", right_on="descendant_id")
    # A resolution counts once per domain, however many of its subjects fall under it.
    counts = joined.drop_duplicates(["undl_id", "ancestor_id"])["ancestor_id"].value_counts()
    return counts.rename_axis("subject_id").rename("count")


def _reception(snapshot: DataSnapshot, sponsored: pd.DataFrame, country: str) -> pd.DataFrame:
    recorded = sponsored.loc[sponsored["modality"] == RECORDED_VOTE, "undl_id"]
    rows = [snapshot.row_index[i] for i in recorded if i in snapshot.row_index]
    columns = ["country", "yes", "no", "abstain"]
    if not rows:
        return pd.DataFrame(columns=columns)
    rows = np.asarray(rows)
    out = pd.DataFrame(
        {
            "country": snapshot.country_columns,
            "yes": snapshot.yes[rows].sum(axis=0),
            "no": snapshot.no[rows].sum(axis=0),
            "abstain": snapshot.abstained[rows].sum(axis=0),
        }
    )
    out = out[(out["country"] != country) & (out[["yes", "no", "abstain"]].sum(axis=1) > 0)]
    return out.sort_values(["no", "abstain", "country"], ascending=[False, False, True])
