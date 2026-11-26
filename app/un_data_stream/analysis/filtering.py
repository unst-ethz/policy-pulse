"""The resolution filter every view works from, and the list operations built on it.

One implementation for every API route: date range and thesaurus subjects (via the
query engine), a country participation filter, and the keyword expression (see `keywords.py`).
"""

from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd

from . import sponsorship
from .keywords import keyword_index, matched_ids
from .query_engine import ResolutionQueryEngine

# One state's point of view: voted on, while a member, or how it voted
CountryMode = Literal["voted", "member", "none", "yes", "no", "abstain"]
# How a group of states voted: all alike, split, opposed (some yes, some no), or a majority one way
GroupVote = Literal["alike", "split", "opposed", "yes", "no", "abstain"]

# Votes a country can cast; the vote columns also hold "X" for "did not vote".
CAST_VOTES = ("Y", "N", "A")

# The columns list views read. Deliberately not the whole frame: callers that want votes ask for
# them explicitly (`attach_votes`) rather than receiving all ~200 vote columns.
COLUMNS = [
    "undl_id",
    "resolution",
    "session",
    "date",
    "title",
    "consensus_score",
    "total_yes",
    "total_no",
    "total_abstentions",
    "modality",
    "undl_link",
    "sponsor_count",
]

# The columns of a CSV export, before the per-country vote columns.
EXPORT_COLUMNS = [
    "undl_id",
    "resolution",
    "date",
    "session",
    "title",
    "agenda_title",
    "subjects",
    "draft",
    "undl_link",
]

# `membership_mask(iso3, dates) -> bool Series`: was the country a UN member on each date.
MembershipMask = Callable[[str, pd.Series], pd.Series]


@dataclass(frozen=True)
class ResolutionFilter:
    """What the user narrowed the resolutions down to. Every field is optional."""

    start_date: str | None = None  # 'YYYY-MM-DD', inclusive
    end_date: str | None = None  # 'YYYY-MM-DD', inclusive
    subject_ids: tuple[str, ...] | None = None  # thesaurus ids, descendants included
    country: str | None = None  # ISO3; only used by `country_mode`
    country_mode: CountryMode = "none"
    keyword: str | None = None  # see keywords.matched_ids for the syntax
    sponsor: str | None = None  # ISO3; only resolutions this state sponsored
    original_sponsor: bool = False  # with `sponsor`: only as an original sponsor
    sponsorship_known: bool = False  # only resolutions with sponsorship data
    modalities: tuple[str, ...] | None = None  # how adopted: `modality` values, any of them
    group: tuple[str, ...] | None = None  # ISO3 codes, judged together by `group_vote`
    group_vote: GroupVote | None = None
    group_all_voted: bool = False  # with `group_vote`: only where every state in `group` voted


def filter_resolutions(
    engine: ResolutionQueryEngine,
    criteria: ResolutionFilter,
    *,
    membership_mask: MembershipMask | None = None,
    apply_country: bool = True,
    apply_keyword: bool = True,
    columns: list[str] | None = COLUMNS,
) -> pd.DataFrame:
    """The resolutions matching `criteria`, oldest first as stored.

    `apply_country` / `apply_keyword` let a view opt out of filters that mean nothing to it (the
    word cloud has no country context; the agreement views ignore keywords).
    `membership_mask` is required for `country_mode == "member"`. `columns=None` keeps the whole
    frame, vote columns included.
    """
    df = engine.query_resolutions(
        start_date=criteria.start_date,
        end_date=criteria.end_date,
        subject_ids=list(criteria.subject_ids) if criteria.subject_ids else None,
        include_descendants=True,
    )

    country = criteria.country
    mode = criteria.country_mode or "none"
    if apply_country and mode != "none" and country in df.columns:
        masks = country_mode_masks(df, country, membership_mask)
        if mode not in masks:
            raise ValueError("country_mode='member' needs a membership_mask")
        df = df[masks[mode]]

    if criteria.sponsor:
        ids = sponsorship.sponsored_ids(
            engine.snapshot, criteria.sponsor, criteria.original_sponsor
        )
        df = df[df["undl_id"].isin(ids)]
    if criteria.sponsorship_known:
        df = df[df["undl_id"].isin(sponsorship.known_ids(engine.snapshot))]

    if criteria.modalities:
        df = df[df["modality"].isin(criteria.modalities)]
    if criteria.group and criteria.group_vote:
        df = df[group_vote_masks(df, criteria.group, criteria.group_all_voted)[criteria.group_vote]]

    if apply_keyword:
        df = filter_by_keyword(engine, df, criteria.keyword)

    if columns is None:
        return df
    return df[[column for column in columns if column in df.columns]]


def country_mode_masks(
    df: pd.DataFrame, country: str, membership_mask: MembershipMask | None = None
) -> dict[CountryMode, pd.Series]:
    """Per `CountryMode` but `none`, which rows of `df` (with vote columns) it keeps.

    `member` only with a `membership_mask`.
    """
    # No normalization: the vote columns are a category of exactly Y/N/A/X with no nulls
    # (repository.py), and normalizing would decategorize 20k values on every call.
    votes = df[country]
    cast_a_vote = votes.isin(CAST_VOTES)
    masks: dict[CountryMode, pd.Series] = {"voted": cast_a_vote}
    if membership_mask is not None:
        # Member on the day, *or* voted on this resolution. The second half is not
        # redundant: membership periods are keyed by ISO code, so a state that voted under a
        # predecessor's code has votes outside its recorded periods.
        masks["member"] = membership_mask(country, df["date"]) | cast_a_vote
    masks.update(yes=votes == "Y", no=votes == "N", abstain=votes == "A")
    return masks


def group_vote_masks(
    df: pd.DataFrame, group: tuple[str, ...], all_voted: bool = False
) -> dict[GroupVote, pd.Series]:
    """Per `GroupVote`, which rows of `df` (with vote columns) the group voted that way on.

    Only states that cast a vote count, and a row needs at least two of them; `all_voted`
    instead needs every state of the group. A majority is more than half of the votes cast.
    `opposed` is a split with both a yes and a no; for two states, exactly Y/N or N/Y.
    """
    votes = df[[c for c in group if c in df.columns]]
    yes, no, abstain = ((votes == v).sum(axis=1) for v in CAST_VOTES)
    cast = yes + no + abstain
    judged = cast == len(group) if all_voted else cast >= 2
    judged &= cast >= 2  # a group of one has nothing to agree or split on
    alike = (yes == cast) | (no == cast) | (abstain == cast)
    return {
        "alike": judged & alike,
        "split": judged & ~alike,
        "opposed": judged & (yes > 0) & (no > 0),
        "yes": judged & (yes * 2 > cast),
        "no": judged & (no * 2 > cast),
        "abstain": judged & (abstain * 2 > cast),
    }


def filter_by_keyword(
    engine: ResolutionQueryEngine, df: pd.DataFrame, keyword: str | None
) -> pd.DataFrame:
    """Keep the rows of `df` matching the keyword expression (all rows if there is none)."""
    if not keyword or not keyword.strip() or df.empty:
        return df
    ids = matched_ids(df, keyword, keyword_index(engine.snapshot))
    return df[df["undl_id"].isin(ids)]


def attach_votes(
    engine: ResolutionQueryEngine, df: pd.DataFrame, countries: list[str]
) -> pd.DataFrame:
    """`df` with one vote column (Y/N/A/X) per country in `countries`, unknown codes skipped."""
    res_table = engine.resolution_table
    available = list(dict.fromkeys(c for c in countries if c and c in res_table.columns))
    if not available or df.empty:
        return df
    vote_df = res_table.loc[res_table["undl_id"].isin(df["undl_id"]), ["undl_id"] + available]
    return df.merge(vote_df, on="undl_id", how="left")


SORT_ORDERS = {
    "date_desc": ("date", False),
    "date_asc": ("date", True),
    "consensus_desc": ("consensus_score", False),
    "consensus_asc": ("consensus_score", True),
}

PairAgreement = Literal["AGREED", "DISAGREED", "STRONGLY_DISAGREED"]


def sort_resolutions(df: pd.DataFrame, order: str | None) -> pd.DataFrame:
    """Sort by one of `SORT_ORDERS` (default newest first); missing values last.

    Ties (same date, same score) are broken by the resolution number (the symbol's last number,
    so A/RES/81/13 sits next to /12 and /100 after /99), then the full symbol (keeping parts
    like 68/95[A] and [B] together), then `undl_id`, so the order, and with it every page of a
    paginated list, doesn't depend on the row order the data happened to load in.
    """
    sort_by, ascending = SORT_ORDERS.get(order or "", SORT_ORDERS["date_desc"])
    number = df["resolution"].str.extract(r"(\d+)\D*$", expand=False).astype(float)
    return (
        df.assign(_number=number)
        .sort_values(
            [sort_by, "_number", "resolution", "undl_id"],
            ascending=ascending,
            na_position="last",
        )
        .drop(columns="_number")
    )


def filter_by_pair_agreement(
    df: pd.DataFrame, country: str, other: str, agreement: PairAgreement
) -> pd.DataFrame:
    """Keep resolutions where `country` and `other` voted alike / differently / Yes-vs-No.

    Both vote columns must be present (see `attach_votes`); otherwise `df` is returned as is.

    Note: AGREED/DISAGREED compare the raw vote codes, so two countries that both did not vote
    ("X" — which includes every resolution adopted without a vote) count as agreeing.
    """
    if country not in df.columns or other not in df.columns:
        return df
    df = df.dropna(subset=[country, other])
    if agreement == "AGREED":
        return df[df[country] == df[other]]
    if agreement == "DISAGREED":
        return df[df[country] != df[other]]
    if agreement == "STRONGLY_DISAGREED":
        yes_no = (df[country] == "Y") & (df[other] == "N")
        no_yes = (df[country] == "N") & (df[other] == "Y")
        return df[yes_no | no_yes]
    return df


def filter_by_vote(df: pd.DataFrame, country: str, vote: str) -> pd.DataFrame:
    """Keep resolutions on which `country` cast `vote` (Y, N, A or X)."""
    if country not in df.columns:
        return df
    return df[df[country] == vote]


def export_frame(
    engine: ResolutionQueryEngine,
    criteria: ResolutionFilter,
    vote_countries: list[str],
    country_name: Callable[[str], str],
    *,
    membership_mask: MembershipMask | None = None,
    apply_country: bool = False,
) -> pd.DataFrame:
    """The table behind the CSV download: filtered resolutions plus named vote columns.

    Applies dates, subjects and keyword. The country participation filter only with
    `apply_country` (a country report's resolutions): by default an export is about the
    resolutions themselves, not one country's record.
    """
    df = filter_resolutions(
        engine,
        criteria,
        membership_mask=membership_mask,
        apply_country=apply_country,
        columns=None,
    )

    vote_cols: list[str] = []
    for country in vote_countries:
        if country and country in df.columns and country not in vote_cols:
            vote_cols.append(country)

    cols = [c for c in EXPORT_COLUMNS + vote_cols if c in df.columns]
    result = df[cols].copy() if not df.empty else pd.DataFrame(columns=cols)
    return result.rename(columns={c: country_name(c) for c in vote_cols})
