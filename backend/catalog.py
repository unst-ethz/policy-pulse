from pathlib import Path

import pandas as pd

from app.un_data_stream.countries import SUPPORTED_LANGS, CountryCatalog
from app.un_data_stream.presets import COUNTRY_PRESETS, ERA_PRESETS

from .keywords import MODES
from .serialization import clean

TOP_LEVEL_SUBJECTS = {f"http://metadata.un.org/thesaurus/{i:02d}" for i in range(19)}


class Catalog:
    def __init__(self, data, assets: Path, version: str):
        self.names = CountryCatalog(data["member_states"])
        self.countries = data["country_columns"]
        self.subjects = data["subject"]
        self.broader = data["broader"]
        self.version = version
        self._subject_cache = {}
        self._metadata_cache = {}
        self.joining = pd.read_csv(
            assets / "joining_dates.csv", parse_dates=["min_date", "max_date"]
        )
        self.regions = pd.read_csv(
            assets / "m49_regional_groupings.csv", sep=";", dtype=str
        ).set_index("ISO-alpha3 Code")
        dates = pd.to_datetime(data["resolution"].date, errors="coerce")
        self.earliest = clean(dates.min())
        self.latest = clean(dates.max())

    def country(self, code, language="en"):
        row = self.regions.loc[code] if code in self.regions.index else {}
        membership = self.joining[self.joining.country == code]
        return clean(
            {
                "code": code,
                "name": self.names.get_country_name(code, language),
                "display_name": self.names.get_country_display_name(code, language),
                "search_terms": self.names.get_country_search_terms(code, language),
                "region": row.get("Region Name", "Other"),
                "subregion": row.get("Sub-region Name"),
                "m49": row.get("M49 Code"),
                "membership_start": membership.min_date.min() if not membership.empty else None,
                "membership_end": membership.max_date.max() if not membership.empty else None,
            }
        )

    def subject_list(self, language="en"):
        if language in self._subject_cache:
            return self._subject_cache[language]
        result = []
        for row in self.subjects.to_dict("records"):
            sid = row["subject_id"]
            label = row.get(f"label_{language}")
            if not isinstance(label, str) or not label.strip():
                label = row.get("label_en", sid)
            parents = (
                self.broader.loc[self.broader.child_id == sid, "parent_id"].tolist()
                if not self.broader.empty
                else []
            )
            result.append(
                {
                    "id": sid,
                    "label": label,
                    "parents": parents,
                    "top_level": sid in TOP_LEVEL_SUBJECTS,
                }
            )
        self._subject_cache[language] = sorted(
            result, key=lambda s: (not s["top_level"], s["label"])
        )
        return self._subject_cache[language]

    def metadata(self, language="en"):
        if language in self._metadata_cache:
            return self._metadata_cache[language]
        self._metadata_cache[language] = {
            "countries": sorted(
                [self.country(c, language) for c in self.countries], key=lambda c: c["name"]
            ),
            "subjects": self.subject_list(language),
            "eras": [{"id": key, **value} for key, value in ERA_PRESETS.items()],
            "country_presets": [{"id": key, **value} for key, value in COUNTRY_PRESETS.items()],
            "earliest_date": self.earliest,
            "latest_date": self.latest,
            "dataset_version": self.version,
            "languages": list(SUPPORTED_LANGS),
            "word_modes": list(MODES),
        }
        return self._metadata_cache[language]
