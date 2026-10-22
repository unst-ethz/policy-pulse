"""Reference data: dataset metadata, countries, the region and subject trees, headline stats."""

from fastapi import APIRouter, HTTPException

from ... import data
from ...un_data_stream.analysis import overview
from .. import schemas
from ..params import LangParam, day, known_country, subject_labels

router = APIRouter(tags=["reference"])


def data_version() -> str:
    """Identifies the served data; changes whenever a reload swaps it."""
    snapshot = data.query_engine.snapshot
    return (snapshot.source_marker or snapshot.loaded_at).isoformat()


@router.get("/meta", response_model=schemas.Meta)
def meta() -> schemas.Meta:
    snapshot = data.query_engine.snapshot
    table = snapshot.resolution_table
    return schemas.Meta(
        data_version=data_version(),
        last_ingestion=snapshot.source_marker,
        earliest_date=day(table["date"].min()) if not table.empty else None,
        latest_date=day(table["date"].max()) if not table.empty else None,
        resolution_count=len(table),
        country_count=len(snapshot.country_columns),
        languages=list(data.SUPPORTED_LANGS),
    )


def _country(iso3: str, lang: str) -> schemas.Country:
    activity = data.get_voting_activity(iso3)
    periods = data.get_membership_periods(iso3)
    return schemas.Country(
        iso3=iso3,
        name=data.get_country_name(iso3, lang),
        display_name=data.get_country_display_name(iso3, lang),
        historical_names=data.get_historical_names(iso3),
        search_terms=data.get_country_search_terms(iso3, lang),
        region=data.get_country_region(iso3),
        subregion=data.get_country_subregion(iso3),
        former_state=bool(periods) and all(end is not None for _, end in periods),
        membership=[schemas.Period(start=day(start), end=day(end)) for start, end in periods],
        first_vote=day(activity[0]) if activity else None,
        last_vote=day(activity[1]) if activity else None,
        participation=data.get_participation_year_range(iso3),
    )


@router.get("/countries", response_model=list[schemas.Country])
def countries(lang: LangParam = "en") -> list[schemas.Country]:
    """Every country with vote data, sorted by name."""
    result = [_country(iso3, lang) for iso3 in data.available_countries]
    return sorted(result, key=lambda c: c.name)


@router.get("/countries/{iso3}", response_model=schemas.Country)
def country(iso3: str, lang: LangParam = "en") -> schemas.Country:
    return _country(known_country(iso3, "iso3", status_code=404), lang)


def _tree(nodes: list[dict], label) -> list[schemas.TreeNode]:
    return [
        schemas.TreeNode(
            id=node["value"],
            label=label(node),
            children=_tree(node.get("children", []), label),
        )
        for node in nodes
    ]


@router.get("/regions/tree", response_model=list[schemas.TreeNode])
def region_tree(lang: LangParam = "en") -> list[schemas.TreeNode]:
    """M49 World → region → sub-region → [intermediate region] → country, plus historical states.

    Leaves are ISO3 codes with localised names; region names are English (M49 publishes them so).
    """
    countries_with_votes = set(data.available_countries)

    def label(node: dict) -> str:
        if node["value"] in countries_with_votes:
            return data.get_country_display_name(node["value"], lang)
        return node["title"]

    return _tree(data.REGION_TREE_DATA, label)


@router.get("/subjects/tree", response_model=list[schemas.TreeNode])
def subject_tree(lang: LangParam = "en") -> list[schemas.TreeNode]:
    """The UNBIS thesaurus domains with their subjects, as the resolutions use them."""
    labels = subject_labels(lang)
    root = data.SUBJECT_TREE_DATA[0]
    domains = [n for n in root.get("children", []) if not n["value"].startswith("__")]
    return _tree(domains, lambda node: labels.get(node["value"], node["title"]))


@router.get("/stats/overview", response_model=schemas.DatasetStats)
def stats_overview() -> schemas.DatasetStats:
    stats = overview.dataset_stats(data.query_engine)
    if stats is None:
        raise HTTPException(503, "No data loaded")
    return schemas.DatasetStats(
        total_resolutions=stats.total_resolutions,
        year_min=stats.year_min,
        year_max=stats.year_max,
        countries=stats.countries,
        subjects=stats.subjects,
        subject_links=stats.subject_links,
        resolutions_with_subjects=stats.resolutions_with_subjects,
        votes=schemas.VoteTotals(**stats.votes),
    )
