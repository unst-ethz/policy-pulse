"""Agreement analyses: who votes with whom, over which resolutions, topics and sessions.

Pure functions over the query engine — no presentation. The scores themselves come from
`ResolutionQueryEngine.query_agreement_between_countries` (1 = always voted alike, 0 = always
Yes-vs-No, 0.5 per step of difference in between) and `query_multilateral_stats`.
"""

import logging
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .query_engine import ResolutionQueryEngine

logger = logging.getLogger(__name__)

# Minimum number of shared votes for a session to get an agreement value in the timeline.
TIMELINE_MIN_VOTES = 3
# Minimum number of shared votes for a subject to appear in agreement-by-subject.
SUBJECT_MIN_VOTES = 30
# Minimum number of votes cast for a country to appear in the multilateral overview.
MULTILATERAL_MIN_VOTES = 10


def agreement_map(
    engine: ResolutionQueryEngine, country: str, resolution_ids: list[str]
) -> pd.Series:
    """Mean pairwise agreement between `country` and every other country over `resolution_ids`.

    Returns a Series indexed by ISO3 (excluding `country`); NaN means the pair never both cast a
    vote on those resolutions. Empty if `country` has no vote data.
    """
    averaged = engine.query_agreement_between_countries(
        country, resolution_ids=resolution_ids, average=True
    )
    if averaged.empty:
        return pd.Series(dtype=float)
    scores = averaged.drop(columns=["source_country", "resolution_count"]).iloc[0]
    return pd.to_numeric(scores, errors="coerce")


def session_agreement(
    engine: ResolutionQueryEngine,
    country: str,
    others: Iterable[str],
    min_votes: int = TIMELINE_MIN_VOTES,
) -> pd.DataFrame | None:
    """Per-session mean agreement between `country` and each of `others`, over all resolutions.

    Returns one row per session with columns `session`, `year` (median resolution date's year),
    and per other country `avg_agreement_<iso3>` / `n_votes_<iso3>`. A session's average is NaN
    below `min_votes` shared votes, and after the last session the other country voted in.

    Returns None when `country` and none of `others` ever both voted.
    """
    selected = [c for c in others if c != country]
    if not selected:
        return None

    vote_mapping = {"Y": 1, "A": 0, "N": -1}

    cols = ["date", "session", country] + selected
    df = engine.query_resolutions()[cols].copy()

    # Normalize and map votes to numeric; "X" (did not vote) falls out as NaN.
    vote_cols = [country] + selected
    df[vote_cols] = df[vote_cols].astype(str).apply(lambda s: s.str.strip().str.upper())
    df[vote_cols] = df[vote_cols].replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA})
    df[vote_cols] = df[vote_cols].replace(vote_mapping)
    df[vote_cols] = df[vote_cols].apply(pd.to_numeric, errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    mask_any = (~df[country].isna()) & df[selected].notna().any(axis=1)
    if not mask_any.any():
        return None

    # Per-resolution agreement (vectorized)
    diffs = df[selected].subtract(df[country], axis=0).abs()
    agreement = 1.0 - (diffs / 2.0)

    for col in selected:
        both_voted = df[[country, col]].notna().all(axis=1)
        agreement[col] = agreement[col].where(both_voted, np.nan)

    tmp = pd.DataFrame({"session": df["session"], "date": df["date"]})
    for col in selected:
        tmp[f"agreement_{col}"] = agreement[col]

    # Year per session = the median date's year
    session_year = tmp.groupby("session")["date"].median().dt.year
    session_year.name = "year"

    out_parts = [session_year]
    for col in selected:
        grouped = tmp.groupby("session")[f"agreement_{col}"]
        avg_agreement = grouped.mean()
        avg_agreement.name = f"avg_agreement_{col}"
        n_votes = grouped.apply(lambda x: x.notna().sum())
        n_votes.name = f"n_votes_{col}"

        avg_agreement = avg_agreement.where(n_votes >= min_votes, np.nan)
        out_parts.extend([avg_agreement, n_votes])

    out = pd.concat(out_parts, axis=1).reset_index()
    out = out.sort_values(["year", "session"]).reset_index(drop=True)

    # Last-vote cutoff: nothing after the last session each country voted in
    for col in selected:
        avg_col = f"avg_agreement_{col}"
        has_votes = out[avg_col].notna()
        if has_votes.any():
            last_idx = has_votes.values[::-1].argmax()
            last_session_pos = len(has_votes) - 1 - last_idx
            out.loc[out.index[last_session_pos + 1 :], avg_col] = np.nan

    return out


def agreement_by_subject(
    engine: ResolutionQueryEngine,
    country: str,
    other: str,
    start_date: str | None,
    end_date: str | None,
    subject_ids: Iterable[str],
    subject_labels: Mapping[str, str],
    min_votes: int = SUBJECT_MIN_VOTES,
) -> pd.DataFrame:
    """Mean agreement between two countries per subject (descendants included), in a date range.

    Returns columns `subject_id`, `subject_label`, `agreement_score` (0..1), `total_votes` — only
    subjects with at least `min_votes` shared votes, ordered by subject id. Empty if none qualify.
    """
    all_resolutions = engine.query_resolutions(start_date=start_date, end_date=end_date)
    if all_resolutions.empty:
        return pd.DataFrame()

    agreement_df = engine.query_agreement_between_countries(
        country_code=country,
        resolution_ids=all_resolutions["undl_id"].tolist(),
        average=False,
    )
    if agreement_df.empty or other not in agreement_df.columns:
        return pd.DataFrame()

    agreement_df = agreement_df[["undl_id", other]].rename(columns={other: "agreement_score"})
    resolution_subject_df = engine.resolution_subject_table
    closure = engine.closure_table

    results = []
    for subject_id in sorted(subject_ids):
        descendants = closure[closure["ancestor_id"] == subject_id]["descendant_id"].unique()
        subject_filter = list({subject_id, *descendants})

        subject_resolution_ids = resolution_subject_df[
            resolution_subject_df["subject_id"].isin(subject_filter)
        ]["undl_id"].unique()

        subject_agreements = agreement_df[
            agreement_df["undl_id"].isin(subject_resolution_ids)
        ].dropna(subset=["agreement_score"])

        total_votes = len(subject_agreements)
        if total_votes < min_votes or total_votes == 0:
            continue

        results.append(
            {
                "subject_id": subject_id,
                "subject_label": subject_labels.get(subject_id, f"ID: {subject_id}"),
                "agreement_score": subject_agreements["agreement_score"].mean(),
                "total_votes": total_votes,
            }
        )

    return pd.DataFrame(results)


def multilateral_overview(
    engine: ResolutionQueryEngine,
    resolution_ids: list[str],
    min_votes: int = MULTILATERAL_MIN_VOTES,
) -> pd.DataFrame:
    """Per-country multilateral alignment and Y/N/A shares over `resolution_ids`.

    Columns as `query_multilateral_stats`; countries without an alignment score or with fewer than
    `min_votes` votes cast are dropped. Empty if there is no vote data.
    """
    return qualified_multilateral(engine.query_multilateral_stats(resolution_ids), min_votes)


def qualified_multilateral(
    stats: pd.DataFrame, min_votes: int = MULTILATERAL_MIN_VOTES
) -> pd.DataFrame:
    """Drop countries without an alignment score or with fewer than `min_votes` votes cast."""
    if stats.empty:
        return stats
    stats = stats.dropna(subset=["multilateral_alignment"])
    return stats[stats["participation_count"] >= min_votes]


def consensus_scale(
    scores: pd.Series, lo: float | None = None, hi: float | None = None
) -> tuple[float, float, float, float]:
    """Parameters for a colour scale anchored at the mean consensus score.

    Returns `(lo, avg, hi, midpoint_frac)`: lo/hi default to the 1st/99th percentile (clamped to
    [0, 1]) unless given; avg is the data mean; midpoint_frac is avg's position within [lo, hi].
    """
    clean = scores.dropna()
    if clean.empty:
        # No scored resolutions in this slice. Resolutions adopted without a vote or by a
        # non-recorded vote carry no consensus score at all (~72% of the table), so a narrow
        # enough date range selects nothing scorable. Callers are expected to skip the adaptive
        # path entirely in that case; returning a neutral finite scale here keeps a missed guard
        # from feeding NaN into a chart's ticks.
        lo = 0.0 if lo is None else lo
        hi = 1.0 if hi is None else hi
        return lo, (lo + hi) / 2.0, hi, 0.5
    lo = float(max(clean.quantile(0.01), 0.0)) if lo is None else lo
    hi = float(min(clean.quantile(0.99), 1.0)) if hi is None else hi
    avg = float(clean.mean())
    midpoint_frac = (avg - lo) / (hi - lo) if hi > lo else 0.5
    return lo, avg, hi, midpoint_frac
