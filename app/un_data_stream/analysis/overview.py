"""Dataset-wide figures: headline statistics and the most recent resolutions."""

from dataclasses import dataclass

import pandas as pd

from .query_engine import ResolutionQueryEngine

NO_CATEGORY = "No Category"


@dataclass(frozen=True)
class DatasetStats:
    total_resolutions: int
    year_min: int | None
    year_max: int | None
    countries: int  # entities with a vote column (historical states included)
    subjects: int  # distinct thesaurus subjects assigned to at least one resolution
    subject_links: int  # (resolution, subject) assignments
    resolutions_with_subjects: int
    votes: dict[str, int]  # "Y", "N", "A", "X" (not voting) summed over all resolutions


def _total(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    value = df[column].fillna(0).sum()
    try:
        return 0 if pd.isna(value) else int(value)
    except (TypeError, ValueError):
        return 0


def dataset_stats(engine: ResolutionQueryEngine) -> DatasetStats | None:
    """Headline numbers for the whole dataset; None if nothing is loaded."""
    df = engine.query_resolutions()
    if df.empty:
        return None
    dates = pd.to_datetime(df.get("date"), errors="coerce").dropna()
    links = engine.resolution_subject_table
    subject_ids = links.get("subject_id", pd.Series(dtype="object"))
    return DatasetStats(
        total_resolutions=len(df),
        year_min=int(dates.dt.year.min()) if not dates.empty else None,
        year_max=int(dates.dt.year.max()) if not dates.empty else None,
        countries=len(engine.country_columns or []),
        subjects=subject_ids.nunique(),
        subject_links=len(links),
        resolutions_with_subjects=links["undl_id"].nunique() if "undl_id" in links else 0,
        votes={
            "Y": _total(df, "total_yes"),
            "N": _total(df, "total_no"),
            "A": _total(df, "total_abstentions"),
            "X": _total(df, "total_non_voting"),
        },
    )


def category_tags(engine: ResolutionQueryEngine) -> dict[str, str]:
    """undl_id -> the English label of one of its subjects (the first assigned), for display."""
    links = engine.resolution_subject_table
    subjects = engine.subject_table
    if links.empty or subjects.empty or "label_en" not in subjects.columns:
        return {}

    labels = subjects[["subject_id", "label_en"]].dropna(subset=["subject_id", "label_en"])
    merged = links.merge(labels, on="subject_id", how="left").dropna(subset=["label_en"])
    if merged.empty:
        return {}

    merged["undl_id"] = merged["undl_id"].astype(str)
    merged["label_en"] = merged["label_en"].astype(str).str.strip()
    merged = merged[merged["label_en"] != ""]
    if merged.empty:
        return {}

    first = merged.groupby("undl_id", as_index=False)["label_en"].first()
    return dict(zip(first["undl_id"], first["label_en"]))


def recent_resolutions(engine: ResolutionQueryEngine) -> pd.DataFrame:
    """Every resolution, newest first, with a `category_tag` column."""
    df = engine.query_resolutions()
    if df.empty:
        return df

    tags = category_tags(engine)
    if "undl_id" in df.columns:
        df["category_tag"] = df["undl_id"].astype(str).map(tags).fillna(NO_CATEGORY)
    else:
        df["category_tag"] = NO_CATEGORY

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(by=["date"], ascending=False, na_position="last")
    return df
