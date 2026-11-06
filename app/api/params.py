"""Query parameters shared across endpoints, and conversions from pandas to JSON-safe values.

Everything that depends on the loaded data is read off `app.data` at call time, never imported by
value: a reload rebinds those module globals.
"""

import datetime as dt
import math
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import HTTPException, Query

from .. import data
from ..un_data_stream.analysis.filtering import CountryMode, ResolutionFilter
from ..un_data_stream.analysis.profile import NON_RECORDED_VOTE, RECORDED_VOTE, WITHOUT_VOTE

Lang = Literal["en", "fr", "es", "ar", "zh", "ru"]
LangParam = Annotated[Lang, Query(description="Language for country and subject names")]

# How a resolution was adopted, as the API names it -> the data's `modality`
Adoption = Literal["recorded", "non_recorded", "without_vote"]
MODALITIES: dict[str, str] = {
    "recorded": RECORDED_VOTE,
    "non_recorded": NON_RECORDED_VOTE,
    "without_vote": WITHOUT_VOTE,
}


def resolution_filter(
    start_date: Annotated[dt.date | None, Query(description="Inclusive")] = None,
    end_date: Annotated[dt.date | None, Query(description="Inclusive")] = None,
    subject: Annotated[
        list[str] | None,
        Query(
            description="Thesaurus subject ids (repeatable); descendants are included. "
            "`__no_subject__` selects resolutions without any subject."
        ),
    ] = None,
    country: Annotated[
        str | None, Query(description="ISO3 code, used by `country_mode`", min_length=3)
    ] = None,
    country_mode: Annotated[
        CountryMode,
        Query(
            description="`voted`: only resolutions the country voted on; "
            "`member`: only while it was a UN member (or voted)"
        ),
    ] = "none",
    keyword: Annotated[
        str | None,
        Query(description='Title/keyword search: "," = OR, "&" = AND, "\\"…\\"" = exact term'),
    ] = None,
    sponsor: Annotated[
        str | None, Query(description="ISO3 code: only resolutions this state sponsored")
    ] = None,
    original_sponsor: Annotated[
        bool, Query(description="With `sponsor`: only where it was an original sponsor")
    ] = False,
    sponsorship_known: Annotated[
        bool, Query(description="Only resolutions with sponsorship data (mostly since 2000)")
    ] = False,
    adoption: Annotated[
        list[Adoption] | None,
        Query(
            description="How adopted (repeatable, any of them): `recorded` vote with each "
            "state's vote, `non_recorded` vote, or `without_vote`"
        ),
    ] = None,
) -> ResolutionFilter:
    """The resolution selection every list and analysis endpoint works from."""
    return ResolutionFilter(
        start_date=start_date.isoformat() if start_date else None,
        end_date=end_date.isoformat() if end_date else None,
        subject_ids=tuple(subject) if subject else None,
        country=known_country(country, "country") if country else None,
        country_mode=country_mode,
        keyword=keyword.strip() if keyword and keyword.strip() else None,
        sponsor=known_country(sponsor, "sponsor") if sponsor else None,
        original_sponsor=original_sponsor,
        sponsorship_known=sponsorship_known,
        modalities=tuple(MODALITIES[a] for a in adoption) if adoption else None,
    )


def known_country(code: str, param: str, status_code: int = 422) -> str:
    """Normalise an ISO3 code and check it has vote data, else raise 422 (or 404 for paths)."""
    iso3 = code.strip().upper()
    if iso3 not in data.available_countries:
        raise HTTPException(status_code, f"{param}: unknown country code {code!r}")
    return iso3


def country_list(values: list[str] | None, param: str) -> list[str]:
    """Repeatable and/or comma-separated ISO3 codes, validated, order kept, duplicates dropped."""
    codes: list[str] = []
    for value in values or []:
        for part in value.split(","):
            if part.strip():
                iso3 = known_country(part, param)
                if iso3 not in codes:
                    codes.append(iso3)
    return codes


def subject_labels(lang: str) -> dict[str, str]:
    """subject_id -> label in `lang`, falling back to English."""
    subjects = data.query_engine.subject_table
    column = f"label_{lang}"
    labels = subjects["label_en"]
    if column in subjects.columns:
        labels = subjects[column].where(subjects[column].notna() & (subjects[column] != ""), labels)
    return dict(zip(subjects["subject_id"], labels))


# ---------------------------------------------------------------------------
# pandas -> JSON-safe
# ---------------------------------------------------------------------------


def num(value: Any) -> float | None:
    """A float, or None for NaN/None/NA."""
    if value is None or value is pd.NA:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def integer(value: Any) -> int | None:
    number = num(value)
    return None if number is None else int(number)


def text(value: Any) -> str | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return str(value)


def day(value: Any) -> dt.date | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date()
