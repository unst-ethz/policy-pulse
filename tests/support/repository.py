import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.un_data_stream.data.processor import DataProcessor
from backend.service import AnalysisService

ASSETS = Path(__file__).resolve().parents[2] / "app/assets"
ROOT_SUBJECT = "http://metadata.un.org/thesaurus/03"
CHILD_SUBJECT = "http://metadata.un.org/thesaurus/test-child"
OTHER_SUBJECT = "http://metadata.un.org/thesaurus/04"
COUNTRIES = ["USA", "CHE", "FRA", "CHN", "IND", "GER"]


class MemoryRepository:
    """Runs the real processor over clearly synthetic, reproducible records."""

    def __init__(self, frame=None):
        self.logger = logging.getLogger("policy_pulse.fixture")
        self.config = {"version": "test-fixture", "logs": False, "debug": False}
        if frame is None:
            rng = np.random.default_rng(42)
            rows = []
            for i in range(160):
                votes = dict(
                    zip(
                        COUNTRIES,
                        rng.choice(["Y", "N", "A", "X"], len(COUNTRIES), p=[0.5, 0.2, 0.25, 0.05]),
                        strict=False,
                    )
                )
                votes["GER"] = np.nan
                if i % 9 == 0:
                    votes["IND"] = np.nan
                rows.append(
                    {
                        "undl_id": 9000000 + i,
                        "resolution": f"TEST/RES/{i + 1}",
                        "title": f"Synthetic test resolution {i + 1}: {'nuclear disarmament' if i % 2 else 'development cooperation'}",
                        "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                        "session": "1sp" if i >= 150 else str(77 + i // 50),
                        "undl_link": f"https://digitallibrary.un.org/record/{9000000 + i}",
                        **votes,
                    }
                )
            frame = pd.DataFrame(rows)
        frame = frame.copy()
        for col in ["resolution", "title", "session", "undl_link"]:
            if col not in frame:
                frame[col] = "test"
        if "date" not in frame:
            frame["date"] = pd.Timestamp("2024-01-01")
        self.countries = [c for c in COUNTRIES if c in frame]
        for name, vote in [
            ("total_yes", "Y"),
            ("total_no", "N"),
            ("total_abstentions", "A"),
            ("total_non_voting", "X"),
        ]:
            frame[name] = (frame[self.countries] == vote).sum(axis=1)
        consensus, country_columns, multilateral, arrays = DataProcessor(
            self.config, self.logger
        ).calculate_agreement_data(frame)
        frame["consensus_score"] = frame.undl_id.map(consensus)
        subject_links = [
            (rid, CHILD_SUBJECT if i % 2 else OTHER_SUBJECT)
            for i, rid in enumerate(frame.undl_id)
            if i % 7 != 0
        ]
        member_states = pd.DataFrame(
            [
                {
                    "ISO Code": code,
                    "Member State": name,
                    "French": french,
                    "Spanish": name,
                    "Arabic": name,
                    "Chinese": name,
                    "Russian": name,
                    "Other Names": np.nan,
                    "Start date": "1945-01-01",
                    "End date": np.nan,
                }
                for code, name, french in [
                    ("USA", "United States of America", "États-Unis d'Amérique"),
                    ("CHE", "Switzerland", "Suisse"),
                    ("FRA", "France", "France"),
                    ("CHN", "China", "Chine"),
                    ("IND", "India", "Inde"),
                ]
            ]
        )
        self.data = {
            "resolution": frame,
            "country_columns": country_columns,
            "resolution_subject": pd.DataFrame(subject_links, columns=["undl_id", "subject_id"]),
            "subject": pd.DataFrame(
                [
                    {"subject_id": ROOT_SUBJECT, "label_en": "Disarmament"},
                    {"subject_id": CHILD_SUBJECT, "label_en": "Nuclear disarmament"},
                    {"subject_id": OTHER_SUBJECT, "label_en": "Development"},
                ]
            ),
            "broader": pd.DataFrame([{"parent_id": ROOT_SUBJECT, "child_id": CHILD_SUBJECT}]),
            "closure": pd.DataFrame(
                [{"ancestor_id": ROOT_SUBJECT, "descendant_id": CHILD_SUBJECT}]
            ),
            "member_states": member_states,
            "multilateral_scores": multilateral,
            "vote_bool_arrays": arrays,
        }

    def get_data(self):
        return self.data


def build_service():
    return AnalysisService(MemoryRepository(), ASSETS)
