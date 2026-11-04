"""Resolution lists, CSV export and single-resolution detail."""

from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from ... import data
from ...un_data_stream.analysis import filtering, overview, sponsorship
from ...un_data_stream.analysis.filtering import ResolutionFilter
from .. import schemas
from ..params import (
    LangParam,
    country_list,
    day,
    integer,
    known_country,
    num,
    resolution_filter,
    subject_labels,
    text,
)

router = APIRouter(tags=["resolutions"])

VotesParam = Annotated[
    list[str] | None,
    Query(description="ISO3 codes whose votes to include (repeatable or comma-separated)"),
]


def _resolution(row: dict, vote_countries: list[str]) -> schemas.Resolution:
    return schemas.Resolution(
        undl_id=str(row["undl_id"]),
        symbol=text(row.get("resolution")),
        session=text(row.get("session")),
        date=day(row.get("date")),
        title=text(row.get("title")),
        consensus_score=num(row.get("consensus_score")),
        yes=integer(row.get("total_yes")),
        no=integer(row.get("total_no")),
        abstentions=integer(row.get("total_abstentions")),
        modality=text(row.get("modality")),
        undl_link=text(row.get("undl_link")),
        votes={c: row[c] for c in vote_countries if c in row},
        category=text(row.get("category_tag")),
        sponsor_count=integer(row.get("sponsor_count")),
    )


@router.get("/resolutions", response_model=schemas.ResolutionPage)
def list_resolutions(
    criteria: Annotated[ResolutionFilter, Depends(resolution_filter)],
    votes: VotesParam = None,
    sort: Annotated[
        Literal["date_desc", "date_asc", "consensus_desc", "consensus_asc"], Query()
    ] = "date_desc",
    vote: Annotated[
        Literal["Y", "N", "A", "X"] | None,
        Query(description="Only resolutions on which `country` cast this vote"),
    ] = None,
    pair: Annotated[
        str | None, Query(description="ISO3 to compare `country` with, for `agreement`")
    ] = None,
    agreement: Annotated[
        Literal["AGREED", "DISAGREED", "STRONGLY_DISAGREED"] | None,
        Query(description="With `pair`: did `country` and `pair` vote alike/differently/Y-vs-N"),
    ] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> schemas.ResolutionPage:
    """Filtered, sorted, paginated resolutions; optionally with some countries' votes."""
    if (vote or agreement) and not criteria.country:
        raise HTTPException(422, "`vote` and `agreement` need `country`")
    if bool(pair) != bool(agreement):
        raise HTTPException(422, "`pair` and `agreement` go together")
    pair_iso = known_country(pair, "pair") if pair else None

    engine = data.query_engine
    df = filtering.filter_resolutions(engine, criteria, membership_mask=data.membership_mask)
    vote_countries = country_list(votes, "votes")
    needed = vote_countries + [c for c in (criteria.country, pair_iso) if c]
    df = filtering.attach_votes(engine, df, needed)

    df = filtering.sort_resolutions(df, sort)
    if pair_iso and agreement:
        df = filtering.filter_by_pair_agreement(df, criteria.country, pair_iso, agreement)
    if vote:
        df = filtering.filter_by_vote(df, criteria.country, vote)

    page = df.iloc[offset : offset + limit]
    return schemas.ResolutionPage(
        total=len(df),
        recorded_votes=int((df["modality"] == "Vote, recorded").sum()) if len(df) else 0,
        offset=offset,
        limit=limit,
        items=[_resolution(row, vote_countries) for row in page.to_dict("records")],
    )


@router.get(
    "/resolutions.csv",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}}},
)
def export_resolutions(
    criteria: Annotated[ResolutionFilter, Depends(resolution_filter)],
    votes: VotesParam = None,
    lang: LangParam = "en",
) -> Response:
    """The filtered resolutions as CSV, with a named vote column per requested country.

    Applies dates, subjects and keyword; `country_mode` is ignored, as in the Dash download.
    """
    vote_countries = country_list(votes, "votes")
    if criteria.country and criteria.country not in vote_countries:
        vote_countries.insert(0, criteria.country)
    frame = filtering.export_frame(
        data.query_engine,
        criteria,
        vote_countries,
        country_name=lambda iso3: data.get_country_name(iso3, lang),
    )
    return Response(
        frame.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="resolutions.csv"'},
    )


@router.get("/resolutions/recent", response_model=schemas.ResolutionPage)
def recent_resolutions(
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> schemas.ResolutionPage:
    """Newest resolutions first, each tagged with one of its subjects."""
    df = overview.recent_resolutions(data.query_engine)
    page = df.iloc[offset : offset + limit]
    return schemas.ResolutionPage(
        total=len(df),
        recorded_votes=int((df["modality"] == "Vote, recorded").sum()) if len(df) else 0,
        offset=offset,
        limit=limit,
        items=[_resolution(row, []) for row in page.to_dict("records")],
    )


@router.get("/resolutions/{undl_id}", response_model=schemas.ResolutionDetail)
def resolution_detail(undl_id: str, lang: LangParam = "en") -> schemas.ResolutionDetail:
    """One resolution with every country's vote, its subjects and keywords."""
    snapshot = data.query_engine.snapshot
    table = snapshot.resolution_table
    rows = table[table["undl_id"] == undl_id]
    if rows.empty:
        raise HTTPException(404, f"No resolution with undl_id {undl_id!r}")
    row = rows.iloc[0].to_dict()

    links = snapshot.resolution_subject_table
    labels = subject_labels(lang)
    subject_ids = links.loc[links["undl_id"] == undl_id, "subject_id"].tolist()

    keywords: dict[str, list[str]] = {}
    keyword_rows = snapshot.keyword_table[snapshot.keyword_table["undl_id"] == undl_id]
    for dimension, keyword in keyword_rows[["dimension", "keyword"]].itertuples(index=False):
        keywords.setdefault(str(dimension), []).append(keyword)

    sponsors, sponsorship_note = sponsorship.resolution_sponsors(snapshot, undl_id)
    base = _resolution(row, [])
    return schemas.ResolutionDetail(
        **base.model_dump(),
        agenda_title=text(row.get("agenda_title")),
        draft=text(row.get("draft")),
        non_voting=integer(row.get("total_non_voting")),
        subjects=[schemas.Subject(id=s, label=labels.get(s, s)) for s in subject_ids],
        keywords=keywords,
        all_votes={c: row[c] for c in snapshot.country_columns if not pd.isna(row[c])},
        sponsors=[
            schemas.Sponsor(
                # States by code, so a name catalogued in a dozen spellings reads the same.
                name=data.get_country_name(s.country_code, lang) if s.country_code else s.sponsor,
                country=s.country_code,
                type=s.sponsor_type,
                kind=s.sponsor_kind,
            )
            for s in sponsors.itertuples(index=False)
        ],
        sponsorship_note=text(sponsorship_note),
    )
