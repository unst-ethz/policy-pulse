"""Response models. They define the OpenAPI schema the frontend's TypeScript client is built from.

Conventions: ISO3 codes for countries, `undl_id` (a string) for resolutions, scores in [0, 1],
`null` wherever a value is undefined (no shared vote, no recorded vote, no data) — never NaN.
"""

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field

Vote = Literal["Y", "N", "A", "X"]


class Meta(BaseModel):
    data_version: str = Field(description="Changes whenever the served data changes")
    last_ingestion: dt.datetime | None = Field(description="Last successful ingestion run")
    earliest_date: dt.date | None
    latest_date: dt.date | None
    resolution_count: int
    country_count: int
    languages: list[str]


class Period(BaseModel):
    start: dt.date
    end: dt.date | None = Field(description="null while ongoing")


class Country(BaseModel):
    iso3: str
    name: str
    display_name: str = Field(description="Name with historical names, for pickers")
    historical_names: list[str]
    search_terms: str = Field(description="Everything a search box should match on")
    region: str
    subregion: str | None
    former_state: bool = Field(description="The code no longer exists (e.g. SUN, YUG)")
    membership: list[Period]
    first_vote: dt.date | None
    last_vote: dt.date | None
    participation: tuple[int, int] | None = Field(
        description="First and last year of membership or voting, within the data's range"
    )


class TreeNode(BaseModel):
    id: str
    label: str
    children: list["TreeNode"] = []


class VoteTotals(BaseModel):
    Y: int
    N: int
    A: int
    X: int


class DatasetStats(BaseModel):
    total_resolutions: int
    year_min: int | None
    year_max: int | None
    countries: int
    subjects: int
    subject_links: int
    resolutions_with_subjects: int
    votes: VoteTotals


class Resolution(BaseModel):
    undl_id: str
    symbol: str | None
    session: str | None
    date: dt.date | None
    title: str | None
    consensus_score: float | None = Field(description="null unless adopted by recorded vote")
    yes: int | None
    no: int | None
    abstentions: int | None
    modality: str | None
    undl_link: str | None
    votes: dict[str, Vote] = Field(default={}, description="Votes of the requested countries")
    category: str | None = None
    sponsor_count: int | None = Field(
        default=None, description="null when sponsorship is unknown (no draft could be read)"
    )


class ResolutionPage(BaseModel):
    total: int
    recorded_votes: int = Field(description="How many of `total` were adopted by recorded vote")
    offset: int
    limit: int
    items: list[Resolution]


class Subject(BaseModel):
    id: str
    label: str


class Sponsor(BaseModel):
    name: str
    country: str | None = Field(description="ISO3 for states; null for observers, bodies, groups")
    type: Literal["state", "observer", "un_body", "group"]
    kind: Literal["original", "additional"]


class ResolutionDetail(Resolution):
    agenda_title: str | None
    draft: str | None
    non_voting: int | None
    subjects: list[Subject]
    keywords: dict[str, list[str]] = Field(description="Keyword terms per dimension")
    all_votes: dict[str, Vote] = Field(description="Every country's vote (X = did not vote)")
    sponsors: list[Sponsor] = Field(description="Original sponsors first; empty when unknown")
    sponsorship_note: str | None = Field(description="Why the sponsors are unknown, if they are")


class ConsensusScale(BaseModel):
    """Colour-scale anchors: 1st/99th percentile (or fixed bounds) and the mean consensus."""

    lo: float
    avg: float
    hi: float


class CountryScore(BaseModel):
    country: str
    score: float | None


class AgreementMap(BaseModel):
    country: str
    resolution_count: int
    scores: list[CountryScore]
    consensus: ConsensusScale | None


class SessionValue(BaseModel):
    agreement: float | None
    shared_votes: int


class TimelineSession(BaseModel):
    session: str
    special: bool = Field(description="A special or emergency special session, e.g. 11emsp")
    date: dt.date = Field(description="Median date of the session's resolutions")
    year: int
    values: dict[str, SessionValue]


class AgreementTimeline(BaseModel):
    country: str
    compare: list[str]
    min_votes: int
    special_sessions: bool
    sessions: list[TimelineSession]


class SubjectAgreement(BaseModel):
    subject_id: str
    label: str
    agreement: float
    shared_votes: int


class AgreementBySubject(BaseModel):
    country: str
    compare: str
    min_votes: int
    subjects: list[SubjectAgreement]


class MultilateralCountry(BaseModel):
    country: str
    name: str
    region: str
    alignment: float
    yes_rate: float | None
    no_rate: float | None
    abstention_rate: float | None
    votes_cast: int


class Multilateral(BaseModel):
    resolution_count: int
    min_votes: int
    mean_alignment: float | None
    countries: list[MultilateralCountry]


class CloudTerm(BaseModel):
    term: str
    count: int = Field(description="Resolutions a click on the term returns")
    frequency: int = Field(description="Resolutions in the selection carrying the term")
    mean_consensus: float | None
    subject_ids: list[str] = Field(default=[], description="`category` mode: subjects to filter")


class WordCloud(BaseModel):
    mode: str
    resolution_count: int
    terms: list[CloudTerm]
    consensus: ConsensusScale | None


class RankedCountry(BaseModel):
    country: str
    agreement: float


class YearlyAlignment(BaseModel):
    year: int
    values: dict[str, float | None]


class NotableVote(BaseModel):
    undl_id: str
    symbol: str | None
    date: dt.date | None
    title: str | None
    vote: Vote
    consensus_score: float
    undl_link: str | None


class CountryProfile(BaseModel):
    country: str
    name: str
    subregion: str | None
    start_date: dt.date | None
    end_date: dt.date | None
    total_resolutions: int
    recorded_votes: int = Field(description="How many of `total_resolutions` had a recorded vote")
    votes: dict[str, int] = Field(
        description='"Y", "N", "A" and "absent" (did not take part), over recorded votes only'
    )
    adopted_without_vote: int
    non_recorded_votes: int = Field(description="Adopted by a vote not recorded per country")
    multilateral_alignment: float | None
    multilateral_rank: int | None
    multilateral_ranked: int
    min_joint_votes: int
    most_aligned: list[RankedCountry]
    least_aligned: list[RankedCountry]
    comparison: list[str]
    comparison_is_default: bool
    yearly_alignment: list[YearlyAlignment]
    opposed_high_consensus: list[NotableVote]
    supported_low_consensus: list[NotableVote]


class SponsorshipYear(BaseModel):
    year: int
    with_data: int = Field(description="Resolutions that year with sponsorship data")
    original: int
    additional: int


class SubjectCount(BaseModel):
    subject_id: str
    label: str
    count: int


class SponsorReception(BaseModel):
    country: str
    yes: int
    no: int
    abstain: int


class CountrySponsorship(BaseModel):
    country: str
    name: str
    start_date: dt.date | None
    end_date: dt.date | None
    resolutions: int = Field(description="Resolutions in the period")
    with_data: int = Field(description="Of those, with sponsorship data")
    sponsored: int
    original: int = Field(description="Of `sponsored`, as an original sponsor")
    per_year: list[SponsorshipYear]
    by_subject: list[SubjectCount] = Field(description="Per thesaurus domain, most first")
    reception: list[SponsorReception] = Field(
        description="Other countries' votes on the recorded-vote resolutions it sponsored, "
        "most votes against first"
    )
    not_yes: list[NotableVote] = Field(
        description="Resolutions it sponsored but then abstained on or voted against"
    )


class DomainYear(BaseModel):
    year: int
    resolutions: int
    recorded: int = Field(description="Of those, adopted by a recorded vote")


class Domain(BaseModel):
    subject_id: str
    label: str
    resolutions: int
    recorded: int
    per_year: list[DomainYear] = Field(description="Years without a resolution are left out")


class Domains(BaseModel):
    resolutions: int = Field(description="All resolutions in the period, with or without subject")
    per_year: list[DomainYear] = Field(description="All resolutions per year, for shares")
    domains: list[Domain] = Field(description="Most resolutions first")


class YearCount(BaseModel):
    year: int
    resolutions: int


class AdoptionCount(BaseModel):
    adoption: Literal["recorded", "non_recorded", "without_vote"]
    resolutions: int


class GroupVoteCount(BaseModel):
    group_vote: Literal["alike", "split", "yes", "no", "abstain"]
    resolutions: int


class DomainCount(BaseModel):
    subject_id: str
    label: str
    resolutions: int


class ResolutionFacets(BaseModel):
    years: list[YearCount] = Field(
        description="Every filter but the period applied; years without a resolution left out"
    )
    adoption: list[AdoptionCount] = Field(
        description="Every filter but `adoption` applied; every kind, zero included"
    )
    group_votes: list[GroupVoteCount] = Field(
        description="With `group`: every filter but `group_vote` applied; empty without a group"
    )
    domains: list[DomainCount] = Field(
        description="Every filter but the subjects applied; most first, empty domains left out"
    )


class SubjectSponsor(BaseModel):
    country: str
    sponsored: int
    original: int = Field(description="Of `sponsored`, as an original sponsor")
    own_total: int = Field(description="Everything the state sponsored in the period, any subject")


class SubjectSponsors(BaseModel):
    subject_id: str
    label: str
    resolutions: int = Field(description="Resolutions in the subject and period")
    with_data: int = Field(description="Of those, with sponsorship data")
    sponsors: list[SubjectSponsor] = Field(description="States only, most sponsored first")
