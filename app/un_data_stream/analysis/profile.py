"""A country's voting profile over a period: what it voted, who it votes with, notable votes."""

from dataclasses import dataclass

import pandas as pd

from .query_engine import ResolutionQueryEngine

P5 = ["USA", "GBR", "FRA", "RUS", "CHN"]
# Minimum number of shared votes for a country to appear in the most/least aligned rankings.
MIN_JOINT_VOTES = 100
RANKING_SIZE = 10
NOTABLE_VOTES = 5
# `modality` of a resolution with per-country votes; the others were adopted without a vote
# ("Without a vote") or by a vote not recorded per country ("Vote, non-recorded").
RECORDED_VOTE = "Vote, recorded"
NON_RECORDED_VOTE = "Vote, non-recorded"


@dataclass(frozen=True)
class CountryProfile:
    country: str
    start_date: str | None  # after clamping to the country's participation range
    end_date: str | None
    comparison: list[str]  # countries the alignment-over-time is computed against
    comparison_is_default: bool  # True when `comparison` fell back to the P5
    total_resolutions: int
    recorded_votes: int  # how many of `total_resolutions` were adopted by recorded vote
    votes: dict[str, int]  # over recorded votes only: "Y", "N", "A", "absent" -> count
    adopted_without_vote: int
    non_recorded_votes: int  # adopted by a vote not recorded per country
    multilateral_alignment: float | None
    multilateral_rank: int | None  # 1 = most aligned
    multilateral_ranked: int  # how many countries were ranked
    most_aligned: pd.Series  # iso3 -> mean agreement, best first
    least_aligned: pd.Series  # iso3 -> mean agreement, worst first
    yearly_alignment: pd.DataFrame  # index year, one column per comparison country
    opposed_high_consensus: pd.DataFrame  # voted N/A where most of the world agreed
    supported_low_consensus: pd.DataFrame  # voted Y on the most divisive resolutions


def clamp_to_participation(
    start_date: str | None, end_date: str | None, participation: tuple[int, int] | None
) -> tuple[str | None, str | None]:
    """Narrow a date range to the years a country took part in."""
    if not participation:
        return start_date, end_date
    first = f"{participation[0]}-01-01"
    last = f"{participation[1]}-12-31"
    if start_date is None or start_date < first:
        start_date = first
    if end_date is None or end_date > last:
        end_date = last
    return start_date, end_date


def country_profile(
    engine: ResolutionQueryEngine,
    country: str,
    start_date: str | None = None,
    end_date: str | None = None,
    comparison: list[str] | None = None,
    participation: tuple[int, int] | None = None,
    min_joint_votes: int = MIN_JOINT_VOTES,
) -> CountryProfile | None:
    """Build the profile of `country` (an ISO3 in `engine.country_columns`).

    `participation` is the (first, last) year the country took part; the date range is clamped to
    it. `comparison` falls back to the P5 when none of its codes is usable. Returns None when no
    resolution falls in the (clamped) range.
    """
    start_date, end_date = clamp_to_participation(start_date, end_date, participation)

    available = set(engine.country_columns)
    requested = [c for c in (comparison or []) if c in available and c != country]
    comparison = requested or P5

    res_df = engine.query_resolutions(start_date=start_date, end_date=end_date)
    if res_df.empty:
        return None
    res_ids = res_df["undl_id"].tolist()

    res_table = engine.resolution_table
    cols = [
        c
        for c in ["undl_id", "date", "modality", "consensus_score", country]
        if c in res_table.columns
    ]
    vote_df = res_table[res_table["undl_id"].isin(res_ids)][cols].copy()
    vote_df["date"] = pd.to_datetime(vote_df["date"])
    vote_df["year"] = vote_df["date"].dt.year

    # Multilateral alignment score + rank
    score = rank = None
    ranked = 0
    ml_stats = engine.query_multilateral_stats(res_ids)
    if not ml_stats.empty:
        ml_valid = ml_stats.dropna(subset=["multilateral_alignment"]).copy()
        ml_valid["_rank"] = (
            ml_valid["multilateral_alignment"].rank(ascending=False, method="min").astype(int)
        )
        ranked = len(ml_valid)
        row = ml_valid[ml_valid["country"] == country]
        if not row.empty:
            score = float(row["multilateral_alignment"].iloc[0])
            rank = int(row["_rank"].iloc[0])

    total = len(vote_df)
    recorded = vote_df["modality"] == RECORDED_VOTE
    n_recorded = int(recorded.sum())
    n_non_recorded = int((vote_df["modality"] == NON_RECORDED_VOTE).sum())
    cast = vote_df.loc[recorded, country]
    n_yes = int((cast == "Y").sum())
    n_no = int((cast == "N").sum())
    n_abs = int((cast == "A").sum())

    # Bilateral agreement: rankings and alignment over time
    scores_df = engine.query_agreement_between_countries(
        country, resolution_ids=res_ids, average=False
    )
    score_cols = [c for c in scores_df.columns if c != "undl_id"]
    joint_counts = scores_df[score_cols].notna().sum()
    qualified = joint_counts[joint_counts >= min_joint_votes].index
    avg_by_country = scores_df[score_cols].mean().dropna().sort_values(ascending=False)
    avg_qualified = avg_by_country[avg_by_country.index.isin(qualified)]

    valid_comp = [c for c in comparison if c in scores_df.columns]
    yearly = (
        scores_df.merge(vote_df[["undl_id", "year"]], on="undl_id", how="left")
        .groupby("year")[valid_comp]
        .mean()
        if valid_comp
        else pd.DataFrame()
    )

    # Notable votes
    merged = vote_df.merge(
        res_df[["undl_id", "title", "resolution", "undl_link"]], on="undl_id", how="left"
    )
    detail = merged.dropna(subset=["consensus_score", country])
    opposed = (
        detail[detail[country].isin(["N", "A"])]
        .sort_values("consensus_score", ascending=False)
        .head(NOTABLE_VOTES)
    )
    supported = (
        detail[detail[country] == "Y"]
        .sort_values("consensus_score", ascending=True)
        .head(NOTABLE_VOTES)
    )

    return CountryProfile(
        country=country,
        start_date=start_date,
        end_date=end_date,
        comparison=comparison,
        comparison_is_default=not requested,
        total_resolutions=total,
        recorded_votes=n_recorded,
        votes={"Y": n_yes, "N": n_no, "A": n_abs, "absent": n_recorded - n_yes - n_no - n_abs},
        adopted_without_vote=total - n_recorded - n_non_recorded,
        non_recorded_votes=n_non_recorded,
        multilateral_alignment=score,
        multilateral_rank=rank,
        multilateral_ranked=ranked,
        most_aligned=avg_qualified.head(RANKING_SIZE),
        least_aligned=avg_qualified.tail(RANKING_SIZE).sort_values(),
        yearly_alignment=yearly,
        opposed_high_consensus=opposed,
        supported_low_consensus=supported,
    )
