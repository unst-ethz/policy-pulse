"""Public API contracts. Scores are proportions; unavailable values are null."""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Language = Literal["en", "fr", "es", "ar", "zh", "ru"]
Vote = Literal["Y", "N", "A", "X"]
WordMode = Literal["default", "geopolitical", "thematic", "action", "category"]
# The scientific engine uses pandas nanosecond timestamps. Validate at the API
# boundary so dates outside that representation produce a useful client error.
AnalysisDate = Annotated[
    date,
    Field(
        ge=date(1678, 1, 1),
        le=date(2261, 12, 31),
        description="Date between 1678-01-01 and 2261-12-31, inclusive.",
    ),
]


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_date: AnalysisDate | None = None
    end_date: AnalysisDate | None = None
    subject: list[str] = Field(default_factory=list, max_length=100)
    include_descendants: bool = True
    country: str | None = Field(None, pattern=r"^[A-Z]{3}$")
    country_mode: Literal["none", "voted", "member"] = "none"
    keyword: str = Field("", max_length=300)

    @model_validator(mode="after")
    def ordered_dates(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        if self.country_mode != "none" and not self.country:
            raise ValueError("country is required for country_mode")
        return self


class ResolutionFilters(Filters):
    compare: list[str] = Field(default_factory=list, max_length=50)
    vote: Vote | None = None
    agreement: Literal["AGREED", "DISAGREED", "STRONGLY_DISAGREED"] | None = None
    sort: Literal["date_desc", "date_asc", "consensus_desc", "consensus_asc"] = "date_desc"
    offset: int = Field(0, ge=0, le=1_000_000)
    limit: int = Field(20, ge=1, le=100)

    @model_validator(mode="after")
    def compatible_vote_filters(self):
        if self.agreement and (not self.country or len(self.compare) != 1):
            raise ValueError("agreement requires a country and exactly one comparison")
        if self.vote and (not self.country or self.compare):
            raise ValueError("vote requires a country and no comparisons")
        return self


class Country(BaseModel):
    code: str
    name: str
    display_name: str
    search_terms: str
    region: str
    subregion: str | None
    m49: str | None
    membership_start: str | None
    membership_end: str | None


class Subject(BaseModel):
    id: str
    label: str
    parents: list[str]
    top_level: bool


class Era(BaseModel):
    id: str
    label: str
    start: int
    end: int | None


class CountryPreset(BaseModel):
    id: str
    label: str
    countries: list[str]


class Metadata(BaseModel):
    countries: list[Country]
    subjects: list[Subject]
    eras: list[Era]
    country_presets: list[CountryPreset]
    earliest_date: str | None
    latest_date: str | None
    dataset_version: str
    languages: list[str]
    word_modes: list[str]


class VoteCounts(BaseModel):
    yes: int | None
    no: int | None
    abstain: int | None
    not_voting: int | None


class Resolution(BaseModel):
    id: str
    symbol: str | None
    title: str
    date: str | None
    session: str | None
    source_url: str | None
    consensus: float | None
    counts: VoteCounts
    votes: dict[str, Vote | None]


class ResolutionPage(BaseModel):
    items: list[Resolution]
    total: int
    offset: int
    limit: int


class ResolutionDetail(Resolution):
    subjects: list[Subject]


class Overview(BaseModel):
    resolutions: int
    countries: int
    subjects: int
    subject_links: int
    earliest_date: str | None
    latest_date: str | None
    votes: VoteCounts
    recent: list[Resolution]


class AgreementRow(BaseModel):
    country: str
    score: float | None
    shared_votes: int


class AgreementResult(BaseModel):
    country: str
    resolution_count: int
    reference_longitude: float
    consensus_midpoint: float | None
    items: list[AgreementRow]


class TimelinePoint(BaseModel):
    session: str
    year: int | None
    special: bool
    scores: dict[str, float | None]
    shared_votes: dict[str, int]


class TimelineResult(BaseModel):
    country: str
    comparisons: list[str]
    minimum_shared_votes: int
    items: list[TimelinePoint]


class SubjectScore(BaseModel):
    subject_id: str
    subject_label: str
    score: float
    shared_votes: int


class SubjectResult(BaseModel):
    country: str
    comparison: str
    minimum_shared_votes: int
    items: list[SubjectScore]


class MultilateralRow(BaseModel):
    country: str
    multilateral_alignment: float | None
    yes_rate: float | None
    no_rate: float | None
    abstention_rate: float | None
    participation_count: int


class MultilateralResult(BaseModel):
    resolution_count: int
    minimum_votes: int
    mean_alignment: float | None
    items: list[MultilateralRow]


class Word(BaseModel):
    term: str
    count: int
    consensus: float | None
    subject_ids: list[str]


class WordResult(BaseModel):
    mode: WordMode
    available: bool
    resolution_count: int
    items: list[Word]


class YearPoint(BaseModel):
    year: int
    scores: dict[str, float | None]


class Profile(BaseModel):
    country: Country
    start_date: str | None
    end_date: str | None
    resolution_count: int
    votes: VoteCounts
    alignment: float | None
    rank: int | None
    ranked_countries: int
    minimum_shared_votes: int
    most_aligned: list[AgreementRow]
    least_aligned: list[AgreementRow]
    yearly: list[YearPoint]
    opposed_high_consensus: list[Resolution]
    supported_low_consensus: list[Resolution]


class Methodology(BaseModel):
    version: str
    scope: str
    vote_encoding: dict[str, int]
    formula: str
    definitions: dict[str, str]
    thresholds: dict[str, int]
    limitations: list[str]
    sources: list[str]
