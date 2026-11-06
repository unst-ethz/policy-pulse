"""Counts for the resolution list's facets: how many resolutions each choice would give.

Each facet ignores its own filter and keeps all others, so its counts show what picking another
value would return (the years ignore the period, the domains ignore the subjects, the modalities
ignore how a resolution was adopted).
"""

from dataclasses import replace
from typing import Iterable

import pandas as pd

from .filtering import MembershipMask, ResolutionFilter, filter_resolutions, group_vote_masks
from .query_engine import ResolutionQueryEngine
from .subjects import domain_pairs


def year_counts(
    engine: ResolutionQueryEngine,
    criteria: ResolutionFilter,
    membership_mask: MembershipMask | None = None,
) -> pd.Series:
    """Resolutions per year under every filter but the period; years without one are absent."""
    df = filter_resolutions(
        engine, replace(criteria, start_date=None, end_date=None), membership_mask=membership_mask
    )
    return df.groupby(df["date"].dt.year).size().astype(int)


def modality_counts(
    engine: ResolutionQueryEngine,
    criteria: ResolutionFilter,
    membership_mask: MembershipMask | None = None,
) -> pd.Series:
    """Resolutions per `modality` (how adopted) under every filter but the modality."""
    df = filter_resolutions(
        engine, replace(criteria, modalities=None), membership_mask=membership_mask
    )
    return df["modality"].value_counts().astype(int)


def group_vote_counts(
    engine: ResolutionQueryEngine,
    criteria: ResolutionFilter,
    membership_mask: MembershipMask | None = None,
) -> dict[str, int]:
    """Resolutions per way the group voted (see `group_vote_masks`), under every other filter."""
    if not criteria.group:
        return {}
    df = filter_resolutions(
        engine, replace(criteria, group_vote=None), membership_mask=membership_mask, columns=None
    )
    masks = group_vote_masks(df, criteria.group, criteria.group_all_voted)
    return {vote: int(mask.sum()) for vote, mask in masks.items()}


def domain_counts(
    engine: ResolutionQueryEngine,
    criteria: ResolutionFilter,
    domains: Iterable[str],
    membership_mask: MembershipMask | None = None,
) -> pd.Series:
    """Resolutions per domain under every filter but the subjects; empty domains are absent."""
    df = filter_resolutions(
        engine, replace(criteria, subject_ids=None), membership_mask=membership_mask
    )
    pairs = domain_pairs(engine.snapshot, df["undl_id"], domains)
    return pairs.groupby("subject_id").size().astype(int)
