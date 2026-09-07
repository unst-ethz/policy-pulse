"""Read-only versioned routes. Validation and serialization live at this boundary."""

import csv
import io
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from . import models as m
from .methodology import METHODOLOGY
from .service import AnalysisService

router = APIRouter(prefix="/api/v1")


def get_service(request: Request) -> AnalysisService:
    service = request.app.state.service
    if service is None:
        raise HTTPException(
            503,
            detail={
                "code": "data_unavailable",
                "message": "The dataset is loading or unavailable. Please retry shortly.",
            },
            headers={"Retry-After": "10"},
        )
    return service


Service = Annotated[AnalysisService, Depends(get_service)]
FilterQuery = Annotated[m.Filters, Query()]
ResolutionQuery = Annotated[m.ResolutionFilters, Query()]


@router.get("/metadata", response_model=m.Metadata, tags=["Catalog"])
def metadata(service: Service, language: m.Language = "en"):
    return service.catalog.metadata(language)


@router.get("/methodology", response_model=m.Methodology, tags=["Catalog"])
def methodology():
    return METHODOLOGY


@router.get("/overview", response_model=m.Overview, tags=["Resolutions"])
def overview(service: Service):
    return service.overview()


@router.get("/resolutions", response_model=m.ResolutionPage, tags=["Resolutions"])
def resolutions(service: Service, filters: ResolutionQuery):
    return service.resolutions(filters)


@router.get(
    "/resolutions/export.csv",
    tags=["Resolutions"],
    responses={
        200: {
            "content": {"text/csv": {}},
            "description": "All matching resolutions, using exactly the list's filters and ordering (pagination is ignored).",
        }
    },
)
def export_resolutions(service: Service, filters: ResolutionQuery):
    frame = service.resolution_frame(filters)
    countries = list(dict.fromkeys(c for c in [filters.country, *filters.compare] if c))
    columns = [
        "undl_id",
        "resolution",
        "session",
        "date",
        "title",
        "consensus_score",
        "total_yes",
        "total_no",
        "total_abstentions",
        "total_non_voting",
        "undl_link",
        *countries,
    ]
    columns = [c for c in columns if c in frame]
    frame = frame[columns].copy()
    frame["date"] = frame.date.dt.strftime("%Y-%m-%d")
    frame = frame.rename(columns={c: service.catalog.names.get_country_name(c) for c in countries})

    def rows():
        buffer = io.StringIO()
        writer = csv.writer(buffer)

        # Prevent spreadsheet formula evaluation without changing API values.
        def cell(value):
            from .serialization import clean

            value = clean(value)
            if isinstance(value, str) and value.lstrip().startswith(
                ("=", "+", "-", "@", "\t", "\r")
            ):
                return "'" + value
            return "" if value is None else value

        writer.writerow(frame.columns)
        yield buffer.getvalue()
        for row in frame.itertuples(index=False, name=None):
            buffer.seek(0)
            buffer.truncate(0)
            writer.writerow([cell(v) for v in row])
            yield buffer.getvalue()

    return StreamingResponse(
        rows(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="policy-pulse-resolutions.csv"'},
    )


@router.get("/resolutions/{resolution_id}", response_model=m.ResolutionDetail, tags=["Resolutions"])
def resolution_detail(resolution_id: str, service: Service):
    result = service.resolution_detail(resolution_id)
    if result is None:
        raise HTTPException(404, "Resolution not found")
    return result


@router.get("/analysis/agreement/{country}", response_model=m.AgreementResult, tags=["Analysis"])
def agreement(country: str, service: Service, filters: FilterQuery):
    return service.agreement(country, filters)


@router.get("/analysis/timeline/{country}", response_model=m.TimelineResult, tags=["Analysis"])
def timeline(
    country: str,
    service: Service,
    compare: Annotated[list[str] | None, Query(max_length=50)] = None,
    include_special: bool = False,
):
    """Full-history session means. Date/subject filters intentionally do not apply."""
    return service.timeline(country, compare or [], include_special)


@router.get(
    "/analysis/subjects/{country}/{comparison}", response_model=m.SubjectResult, tags=["Analysis"]
)
def subjects(
    country: str,
    comparison: str,
    service: Service,
    start_date: m.AnalysisDate | None = None,
    end_date: m.AnalysisDate | None = None,
    parent: str | None = None,
):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(422, "start_date must be on or before end_date")
    return service.subjects(country, comparison, start_date, end_date, parent)


@router.get("/analysis/multilateral", response_model=m.MultilateralResult, tags=["Analysis"])
def multilateral(service: Service, filters: FilterQuery):
    return service.multilateral(filters)


@router.get("/analysis/words/{mode}", response_model=m.WordResult, tags=["Analysis"])
def words(mode: m.WordMode, service: Service, filters: FilterQuery):
    return service.words(filters, mode)


@router.get("/countries/{country}/profile", response_model=m.Profile, tags=["Countries"])
def profile(
    country: str,
    service: Service,
    start_date: m.AnalysisDate | None = None,
    end_date: m.AnalysisDate | None = None,
    compare: Annotated[list[str] | None, Query(max_length=50)] = None,
):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(422, "start_date must be on or before end_date")
    return service.profile(country, start_date, end_date, compare or [])
