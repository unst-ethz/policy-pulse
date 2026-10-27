"""Agreement, alignment, word-cloud and country-profile analyses.

Endpoints taking the shared filter parameters run over exactly the resolutions the filter selects
(the client decides which filters a view uses). Exceptions are noted per endpoint.
"""

import datetime as dt
from typing import Annotated, Literal, get_args

from fastapi import APIRouter, Depends, HTTPException, Query

from ... import data
from ...un_data_stream.analysis import agreement, filtering, keywords, profile, word_cloud
from ...un_data_stream.analysis.filtering import ResolutionFilter
from .. import schemas
from ..params import (
    LangParam,
    country_list,
    day,
    known_country,
    num,
    resolution_filter,
    subject_labels,
    text,
)

router = APIRouter(tags=["analysis"])

CloudMode = Literal["default", "geopolitical", "thematic", "action", "category"]
assert set(get_args(CloudMode)) == set(keywords.MODES), "keep CloudMode in sync with MODES"

CompareParam = Annotated[
    list[str],
    Query(description="ISO3 codes to compare with (repeatable or comma-separated)"),
]


def _iso(day_: dt.date | None) -> str | None:
    return day_.isoformat() if day_ else None


def _selection(criteria: ResolutionFilter):
    return filtering.filter_resolutions(
        data.query_engine, criteria, membership_mask=data.membership_mask
    )


def _consensus(scores) -> schemas.ConsensusScale | None:
    if scores.notna().any():
        lo, avg, hi, _ = agreement.consensus_scale(scores)
        return schemas.ConsensusScale(lo=lo, avg=avg, hi=hi)
    return None


@router.get("/agreement/map", response_model=schemas.AgreementMap)
def agreement_map(
    criteria: Annotated[ResolutionFilter, Depends(resolution_filter)],
) -> schemas.AgreementMap:
    """Mean pairwise agreement of `country` with every other country over the selection.

    `consensus` gives the 1st/99th percentile and mean consensus of the selected resolutions,
    for a colour scale centred on the average.
    """
    if not criteria.country:
        raise HTTPException(422, "`country` is required")
    df = _selection(criteria)
    scores = agreement.agreement_map(data.query_engine, criteria.country, df["undl_id"].tolist())
    return schemas.AgreementMap(
        country=criteria.country,
        resolution_count=len(df),
        scores=[schemas.CountryScore(country=c, score=num(s)) for c, s in scores.items()],
        consensus=_consensus(df["consensus_score"]) if len(df) else None,
    )


@router.get("/agreement/timeline", response_model=schemas.AgreementTimeline)
def agreement_timeline(
    country: str,
    compare: CompareParam,
    min_votes: Annotated[int, Query(ge=1)] = agreement.TIMELINE_MIN_VOTES,
    special_sessions: Annotated[
        bool,
        Query(
            description="Give special and emergency special sessions points of their own "
            "instead of counting them towards the regular session in progress"
        ),
    ] = False,
) -> schemas.AgreementTimeline:
    """Mean agreement per General Assembly session over the *whole* history (no filters).

    A session's value is null below `min_votes` shared votes and after the compared country's
    last vote.
    """
    iso3 = known_country(country, "country")
    others = [c for c in country_list(compare, "compare") if c != iso3]
    if not others:
        raise HTTPException(422, "`compare` needs at least one other country")
    out = agreement.session_agreement(
        data.query_engine,
        iso3,
        others,
        min_votes=min_votes,
        special_sessions=special_sessions,
    )
    sessions = []
    for row in out.to_dict("records") if out is not None else []:
        sessions.append(
            schemas.TimelineSession(
                session=str(row["session"]),
                special=not str(row["session"]).isdigit(),
                date=row["date"].date(),
                year=int(row["year"]),
                values={
                    c: schemas.SessionValue(
                        agreement=num(row[f"avg_agreement_{c}"]),
                        shared_votes=int(row[f"n_votes_{c}"]),
                    )
                    for c in others
                },
            )
        )
    return schemas.AgreementTimeline(
        country=iso3,
        compare=others,
        min_votes=min_votes,
        special_sessions=special_sessions,
        sessions=sessions,
    )


@router.get("/agreement/by-subject", response_model=schemas.AgreementBySubject)
def agreement_by_subject(
    country: str,
    compare: str,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    min_votes: Annotated[int, Query(ge=1)] = agreement.SUBJECT_MIN_VOTES,
    lang: LangParam = "en",
) -> schemas.AgreementBySubject:
    """Agreement between two countries per thesaurus domain, best agreement first.

    Uses the date range only: subjects are the dimension being compared.
    """
    iso3 = known_country(country, "country")
    other = known_country(compare, "compare")
    if iso3 == other:
        raise HTTPException(422, "`country` and `compare` must differ")
    out = agreement.agreement_by_subject(
        data.query_engine,
        iso3,
        other,
        _iso(start_date),
        _iso(end_date),
        data.TOP_LEVEL_SUBJECTS,
        subject_labels(lang),
        min_votes=min_votes,
    )
    if len(out):
        out = out.sort_values("agreement_score", ascending=False)
    rows = out.to_dict("records")
    return schemas.AgreementBySubject(
        country=iso3,
        compare=other,
        min_votes=min_votes,
        subjects=[
            schemas.SubjectAgreement(
                subject_id=r["subject_id"],
                label=r["subject_label"],
                agreement=r["agreement_score"],
                shared_votes=r["total_votes"],
            )
            for r in rows
        ],
    )


@router.get("/multilateral", response_model=schemas.Multilateral)
def multilateral(
    criteria: Annotated[ResolutionFilter, Depends(resolution_filter)],
    min_votes: Annotated[int, Query(ge=1)] = agreement.MULTILATERAL_MIN_VOTES,
    lang: LangParam = "en",
) -> schemas.Multilateral:
    """Each country's mean agreement with all others, and its Yes/No/Abstain shares.

    Countries with fewer than `min_votes` votes cast in the selection are left out.
    """
    df = _selection(criteria)
    stats = agreement.multilateral_overview(data.query_engine, df["undl_id"].tolist(), min_votes)
    return schemas.Multilateral(
        resolution_count=len(df),
        min_votes=min_votes,
        mean_alignment=num(stats["multilateral_alignment"].mean()) if len(stats) else None,
        countries=[
            schemas.MultilateralCountry(
                country=r["country"],
                name=data.get_country_name(r["country"], lang),
                region=data.get_country_region(r["country"]),
                alignment=r["multilateral_alignment"],
                yes_rate=num(r["yes_rate"]),
                no_rate=num(r["no_rate"]),
                abstention_rate=num(r["abstention_rate"]),
                votes_cast=int(r["participation_count"]),
            )
            for r in stats.to_dict("records")
        ],
    )


@router.get("/keywords/cloud", response_model=schemas.WordCloud)
def keyword_cloud(
    criteria: Annotated[ResolutionFilter, Depends(resolution_filter)],
    mode: Annotated[
        CloudMode, Query(description="Keyword dimension, or `category` for thesaurus subjects")
    ] = keywords.DEFAULT_MODE,
    limit: Annotated[int, Query(ge=1, le=100)] = word_cloud.MAX_TERMS,
) -> schemas.WordCloud:
    """The most prominent terms of the selection, sized by how many resolutions each finds.

    `count` already accounts for the active `keyword`: it is what clicking the term (adding it
    as an exact `&` term) would return.
    """
    df = _selection(criteria)
    index = keywords.keyword_index(data.query_engine.snapshot)
    terms = word_cloud.cloud_terms(index, df, mode, keyword=criteria.keyword, limit=limit)
    consensus = word_cloud.mean_consensus(index, [t.term for t in terms], mode, df)
    subject_ids = index.category_term_to_subject_ids if mode == "category" else {}
    return schemas.WordCloud(
        mode=mode,
        resolution_count=len(df),
        terms=[
            schemas.CloudTerm(
                term=t.term,
                count=t.count,
                frequency=t.frequency,
                mean_consensus=score,
                subject_ids=subject_ids.get(t.term, []),
            )
            for t, score in zip(terms, consensus)
        ],
        consensus=_consensus(df["consensus_score"]) if len(df) else None,
    )


def _notable(df, country: str) -> list[schemas.NotableVote]:
    return [
        schemas.NotableVote(
            undl_id=str(r["undl_id"]),
            symbol=text(r.get("resolution")),
            date=day(r.get("date")),
            title=text(r.get("title")),
            vote=r[country],
            consensus_score=r["consensus_score"],
            undl_link=text(r.get("undl_link")),
        )
        for r in df.to_dict("records")
    ]


@router.get("/countries/{iso3}/profile", response_model=schemas.CountryProfile)
def country_profile(
    iso3: str,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    compare: Annotated[
        list[str] | None, Query(description="ISO3 codes for the alignment over time")
    ] = None,
    min_joint_votes: Annotated[int, Query(ge=1)] = profile.MIN_JOINT_VOTES,
    lang: LangParam = "en",
) -> schemas.CountryProfile:
    """A country's record over a period (clamped to its participation), as on the profile page.

    `compare` defaults to the P5 when empty.
    """
    country = known_country(iso3, "iso3", status_code=404)
    prof = profile.country_profile(
        data.query_engine,
        country,
        start_date=_iso(start_date),
        end_date=_iso(end_date),
        comparison=country_list(compare, "compare"),
        participation=data.get_participation_year_range(country),
        min_joint_votes=min_joint_votes,
    )
    if prof is None:
        raise HTTPException(404, "No resolutions in the selected period")

    yearly = prof.yearly_alignment
    return schemas.CountryProfile(
        country=country,
        name=data.get_country_name(country, lang),
        subregion=data.get_country_subregion(country),
        start_date=day(prof.start_date),
        end_date=day(prof.end_date),
        total_resolutions=prof.total_resolutions,
        recorded_votes=prof.recorded_votes,
        votes=prof.votes,
        adopted_without_vote=prof.adopted_without_vote,
        non_recorded_votes=prof.non_recorded_votes,
        multilateral_alignment=prof.multilateral_alignment,
        multilateral_rank=prof.multilateral_rank,
        multilateral_ranked=prof.multilateral_ranked,
        min_joint_votes=min_joint_votes,
        most_aligned=[
            schemas.RankedCountry(country=c, agreement=s) for c, s in prof.most_aligned.items()
        ],
        least_aligned=[
            schemas.RankedCountry(country=c, agreement=s) for c, s in prof.least_aligned.items()
        ],
        comparison=prof.comparison,
        comparison_is_default=prof.comparison_is_default,
        yearly_alignment=[
            schemas.YearlyAlignment(year=int(year), values={c: num(v) for c, v in row.items()})
            for year, row in yearly.iterrows()
        ],
        opposed_high_consensus=_notable(prof.opposed_high_consensus, country),
        supported_low_consensus=_notable(prof.supported_low_consensus, country),
    )
