"""Use cases over the existing scientific engine; no web or React dependencies."""

from datetime import date
from pathlib import Path

import pandas as pd

from app.un_data_stream import ResolutionQueryEngine
from app.un_data_stream.analysis.subjects import calculate_agreement
from app.un_data_stream.country_utils import get_country_longitude

from .catalog import TOP_LEVEL_SUBJECTS, Catalog
from .keywords import KeywordIndex
from .methodology import METHODOLOGY
from .models import Filters, ResolutionFilters
from .serialization import clean, records


class InvalidSelection(ValueError):
    pass


class AnalysisService:
    def __init__(self, repository, assets: Path):
        self.engine = ResolutionQueryEngine(repository)
        self.catalog = Catalog(
            repository.get_data(), assets, str(repository.config.get("version", "unknown"))
        )
        self.keywords = KeywordIndex(self.engine, assets)

    def check_countries(self, *countries):
        unknown = [c for c in countries if c and c not in self.engine.country_columns]
        if unknown:
            raise InvalidSelection(f"Unknown country code: {', '.join(unknown)}")

    def filtered(self, filters: Filters, *, countries=True, keywords=True):
        self.check_countries(filters.country)
        allowed_subjects = set(self.engine.subject_table.subject_id) | {
            "__no_subject__",
            "__all_subjects__",
        }
        if set(filters.subject) - allowed_subjects:
            raise InvalidSelection("Unknown subject identifier")
        # The UI's all-subject sentinel represents no restriction.
        subjects = [s for s in filters.subject if s != "__all_subjects__"]
        df = self.engine.query_resolutions(
            start_date=filters.start_date,
            end_date=filters.end_date,
            subject_ids=subjects,
            include_descendants=filters.include_descendants,
        )
        if countries and filters.country:
            if filters.country_mode == "voted":
                df = df[
                    df[filters.country].astype(str).str.strip().str.upper().isin(["Y", "N", "A"])
                ]
            elif filters.country_mode == "member":
                info = self.catalog.country(filters.country)
                if info["membership_start"] and info["membership_end"]:
                    df = df[df.date.between(info["membership_start"], info["membership_end"])]
        if keywords and filters.keyword.strip() and not df.empty:
            df = df[df.undl_id.isin(self.keywords.matching_ids(df, filters.keyword))]
        return df

    def resolution_frame(self, filters: ResolutionFilters):
        self.check_countries(*filters.compare)
        df = self.filtered(filters)
        if filters.agreement:
            a, b = filters.country, filters.compare[0]
            df = df.dropna(subset=list(dict.fromkeys([a, b])))
            if filters.agreement == "AGREED":
                df = df[df[a] == df[b]]
            elif filters.agreement == "DISAGREED":
                df = df[df[a] != df[b]]
            else:
                df = df[((df[a] == "Y") & (df[b] == "N")) | ((df[a] == "N") & (df[b] == "Y"))]
        if filters.vote:
            df = df[df[filters.country] == filters.vote]
        column, direction = filters.sort.rsplit("_", 1)
        column = "consensus_score" if column == "consensus" else column
        return df.sort_values(
            [column, "undl_id"],
            ascending=[direction == "asc", True],
            na_position="last",
            kind="stable",
        )

    @staticmethod
    def resolution(row, countries=()):
        def string(value):
            value = clean(value)
            return str(value) if value is not None else None

        return clean(
            {
                "id": str(row["undl_id"]),
                "symbol": string(row.get("resolution")),
                "title": clean(row.get("title")) or "Untitled resolution",
                "date": row.get("date"),
                "session": string(row.get("session")),
                "source_url": row.get("undl_link"),
                "consensus": row.get("consensus_score"),
                "counts": {
                    "yes": row.get("total_yes"),
                    "no": row.get("total_no"),
                    "abstain": row.get("total_abstentions"),
                    "not_voting": row.get("total_non_voting"),
                },
                "votes": {
                    c: clean(row.get(c)) if clean(row.get(c)) in ("Y", "N", "A", "X") else None
                    for c in countries
                },
            }
        )

    def resolutions(self, filters):
        df = self.resolution_frame(filters)
        countries = list(dict.fromkeys(c for c in [filters.country, *filters.compare] if c))
        return {
            "items": [
                self.resolution(r, countries)
                for r in df.iloc[filters.offset : filters.offset + filters.limit].to_dict("records")
            ],
            "total": len(df),
            "offset": filters.offset,
            "limit": filters.limit,
        }

    def resolution_detail(self, rid):
        df = self.engine.resolution_table
        rows = df[df.undl_id.astype(str) == rid]
        if rows.empty:
            return None
        row = rows.iloc[0]
        links = self.engine.resolution_subject_table
        subjects = set(links.loc[links.undl_id == row.undl_id, "subject_id"])
        return {
            **self.resolution(row, self.engine.country_columns),
            "subjects": [s for s in self.catalog.subject_list() if s["id"] in subjects],
        }

    def overview(self):
        df = self.engine.query_resolutions()
        links = self.engine.resolution_subject_table
        return clean(
            {
                "resolutions": len(df),
                "countries": len(self.engine.country_columns),
                "subjects": links.subject_id.nunique(),
                "subject_links": len(links),
                "earliest_date": self.catalog.earliest,
                "latest_date": self.catalog.latest,
                "votes": {
                    name: int(df[column].fillna(0).sum())
                    for name, column in [
                        ("yes", "total_yes"),
                        ("no", "total_no"),
                        ("abstain", "total_abstentions"),
                        ("not_voting", "total_non_voting"),
                    ]
                },
                "recent": [
                    self.resolution(row)
                    for row in df.sort_values(["date", "undl_id"], ascending=[False, True])
                    .head(6)
                    .to_dict("records")
                ],
            }
        )

    def agreement_frame(self, country, df):
        self.check_countries(country)
        if df.empty:
            return pd.DataFrame()
        return self.engine.query_agreement_between_countries(
            country, df.undl_id.tolist(), average=False
        )

    def agreement_rows(self, scores):
        return clean(
            [
                {
                    "country": c,
                    "score": scores[c].mean(),
                    "shared_votes": int(scores[c].notna().sum()),
                }
                for c in scores.columns
                if c != "undl_id"
            ]
        )

    def agreement(self, country, filters):
        # Map uses pairwise participation inside the engine, not a global voter filter.
        self.check_countries(country)
        df = self.filtered(filters, countries=False, keywords=False)
        # Preserve the legacy map's optional colour midpoint: mean resolution
        # consensus after dropping missing reference-country codes, retaining X.
        reference_records = df.dropna(subset=[country])
        midpoint = clean(reference_records.consensus_score.mean())
        return {
            "country": country,
            "resolution_count": len(df),
            "reference_longitude": get_country_longitude(country),
            "consensus_midpoint": midpoint,
            "items": self.agreement_rows(self.agreement_frame(country, df)),
        }

    def timeline(self, country, compare, include_special=False):
        self.check_countries(country, *compare)
        compare = list(dict.fromkeys(c for c in compare if c != country))
        if not compare:
            return {"country": country, "comparisons": [], "minimum_shared_votes": 3, "items": []}
        df = self.engine.query_resolutions()
        scores = self.agreement_frame(country, df)
        if scores.empty:
            return {
                "country": country,
                "comparisons": compare,
                "minimum_shared_votes": 3,
                "items": [],
            }
        joined = df[["undl_id", "session", "date"]].merge(
            scores[["undl_id", *compare]], on="undl_id", validate="one_to_one"
        )
        joined["session"] = joined.session.astype(str)
        groups = joined.groupby("session")
        means, counts = groups[compare].mean(), groups[compare].count()
        means = means.where(counts >= 3)
        years = groups.date.median().dt.year
        items = []
        for session in means.index:
            special = "sp" in session.lower()
            if special and not include_special:
                continue
            items.append(
                {
                    "session": session,
                    "year": years.loc[session],
                    "special": special,
                    "scores": means.loc[session].to_dict(),
                    "shared_votes": counts.loc[session].to_dict(),
                }
            )
        items = sorted(clean(items), key=lambda p: (p["year"] or 0, p["session"]))
        return {
            "country": country,
            "comparisons": compare,
            "minimum_shared_votes": 3,
            "items": items,
        }

    def subjects(self, country, comparison, start_date=None, end_date=None, parent=None):
        self.check_countries(country, comparison)
        labels = self.engine.subject_table.set_index("subject_id").label_en.to_dict()
        if parent and parent not in labels:
            raise InvalidSelection("Unknown subject identifier")
        subject_ids = TOP_LEVEL_SUBJECTS
        if parent:
            broader = self.catalog.broader
            subject_ids = broader.loc[broader.parent_id == parent, "child_id"].unique().tolist()
        result = calculate_agreement(
            self.engine, country, comparison, start_date, end_date, sorted(subject_ids), labels
        )
        items = [
            {
                "subject_id": row["subject_id"],
                "subject_label": row["subject_label"],
                "score": row["agreement_score"],
                "shared_votes": row["total_votes"],
            }
            for row in records(result)
        ]
        return {
            "country": country,
            "comparison": comparison,
            "minimum_shared_votes": 30,
            "items": items,
        }

    def multilateral(self, filters):
        df = self.filtered(filters, keywords=False)
        # The original engine treats [] as ALL resolutions; do not call it for an empty selection.
        stats = (
            self.engine.query_multilateral_stats(df.undl_id.tolist())
            if not df.empty
            else pd.DataFrame()
        )
        minimum = METHODOLOGY["thresholds"]["multilateral_votes"]
        if not stats.empty:
            stats = stats.dropna(subset=["multilateral_alignment"])
            stats = stats[stats.participation_count >= minimum]
        return {
            "resolution_count": len(df),
            "minimum_votes": minimum,
            "mean_alignment": clean(stats.multilateral_alignment.mean())
            if not stats.empty
            else None,
            "items": records(stats),
        }

    def words(self, filters, mode):
        df = self.filtered(filters, countries=False)
        return {
            "mode": mode,
            "available": self.keywords.available[mode],
            "resolution_count": len(df),
            "items": clean(self.keywords.frequencies(df, mode)),
        }

    def profile(self, country, start_date=None, end_date=None, compare=()):
        self.check_countries(country, *compare)
        info = self.catalog.country(country)
        if info["membership_start"]:
            start_date = max(start_date or date.min, date(int(info["membership_start"][:4]), 1, 1))
            end_date = min(end_date or date.max, date(int(info["membership_end"][:4]), 12, 31))
        if start_date and end_date and start_date > end_date:
            df = self.engine.resolution_table.iloc[:0].copy()
        else:
            df = self.filtered(Filters(start_date=start_date, end_date=end_date))
        scores = self.agreement_frame(country, df)
        rankings = sorted(
            [
                r
                for r in self.agreement_rows(scores)
                if r["score"] is not None and r["shared_votes"] >= 100
            ],
            key=lambda r: (-r["score"], r["country"]),
        )
        stats = (
            self.engine.query_multilateral_stats(df.undl_id.tolist())
            if not df.empty
            else pd.DataFrame()
        )
        alignment, rank, ranked = None, None, 0
        if not stats.empty:
            valid = stats.dropna(subset=["multilateral_alignment"]).copy()
            ranked = len(valid)
            valid["rank"] = valid.multilateral_alignment.rank(ascending=False, method="min")
            row = valid[valid.country == country]
            if not row.empty:
                alignment, rank = (
                    float(row.multilateral_alignment.iloc[0]),
                    int(row["rank"].iloc[0]),
                )
        comparison = [
            c
            for c in (compare or ["USA", "GBR", "FRA", "RUS", "CHN"])
            if c != country and c in scores.columns
        ]
        yearly = []
        if comparison and not df.empty:
            joined = df[["undl_id", "date"]].merge(
                scores[["undl_id", *comparison]], on="undl_id", validate="one_to_one"
            )
            for year, row in (
                joined.groupby(pd.to_datetime(joined.date).dt.year)[comparison].mean().iterrows()
            ):
                yearly.append({"year": int(year), "scores": clean(row.to_dict())})
        counts = {
            "yes": int((df[country] == "Y").sum()),
            "no": int((df[country] == "N").sum()),
            "abstain": int((df[country] == "A").sum()),
        }
        counts["not_voting"] = len(df) - sum(counts.values())
        details = df.dropna(subset=["consensus_score", country])
        opposed = (
            details[details[country].isin(["N", "A"])]
            .sort_values("consensus_score", ascending=False)
            .head(5)
        )
        supported = details[details[country] == "Y"].sort_values("consensus_score").head(5)
        return {
            "country": info,
            "start_date": clean(start_date),
            "end_date": clean(end_date),
            "resolution_count": len(df),
            "votes": counts,
            "alignment": alignment,
            "rank": rank,
            "ranked_countries": ranked,
            "minimum_shared_votes": 100,
            "most_aligned": rankings[:10],
            "least_aligned": sorted(rankings[-10:], key=lambda r: (r["score"], r["country"])),
            "yearly": yearly,
            "opposed_high_consensus": [
                self.resolution(r, [country]) for r in opposed.to_dict("records")
            ],
            "supported_low_consensus": [
                self.resolution(r, [country]) for r in supported.to_dict("records")
            ],
        }
