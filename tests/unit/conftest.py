"""A small, hand-checkable voting world for the analysis-layer unit tests.

Four countries, six resolutions over three sessions, a three-node subject tree and a handful of
keywords. The vote arrays are built by the real `DataProcessor`, so the engine under test sees
exactly the shapes the MongoDB load produces.

Votes (Y/N/A, X = did not vote):

    undl_id  date        session  title                          AAA BBB CCC DDD  subject
    101      2000-01-10  55       Nuclear disarmament            Y   Y   N   A    DISARMAMENT
    102      2000-06-01  55       Climate change and the oceans  Y   Y   Y   X    ENVIRONMENT
    103      2001-03-01  56       Human rights of women          Y   N   Y   Y    PEACE
    104      2001-09-01  56       Nuclear-weapon-free zones      N   N   Y   Y    DISARMAMENT
    105      2002-02-01  57       International Day of Peace     X   X   X   X    PEACE (no vote)
    106      2002-05-01  57       Women in development           A   Y   Y   N    -

Sponsors: 101 AAA (original) + BBB (additional); 103 AAA + the GA President (both original);
104 AAA (additional); 106 CCC + the Group of 77 (original). 102 has a placeholder (draft not
found), 105 no rows at all — both "unknown".

DISARMAMENT is a child of PEACE. Pairwise agreement is 1 for equal votes, 0.5 one step apart
(Y/A, A/N), 0 for Y/N; resolutions where either side did not vote are skipped. So:

    AAA-BBB = (1 + 1 + 0 + 1 + 0.5) / 5 = 0.7
    AAA-CCC = (0 + 1 + 1 + 0 + 0.5) / 5 = 0.5
    AAA-DDD = (0.5 + 1 + 0 + 0.5) / 4  = 0.5

Consensus (mean agreement over every pair that both voted): 102 = 1 is the most consensual,
104 = 1/3 the least, and 105 has no score because nobody voted.
"""

import logging

import pandas as pd
import pytest

COUNTRIES = ["AAA", "BBB", "CCC", "DDD"]

RESOLUTIONS = [
    ("101", "2000-01-10", "55", "Nuclear disarmament", "YYNA"),
    ("102", "2000-06-01", "55", "Climate change and the oceans", "YYYX"),
    ("103", "2001-03-01", "56", "Human rights of women", "YNYY"),
    ("104", "2001-09-01", "56", "Nuclear-weapon-free zones", "NNYY"),
    ("105", "2002-02-01", "57", "International Day of Peace", "XXXX"),
    ("106", "2002-05-01", "57", "Women in development", "AYYN"),
]

PEACE, DISARMAMENT, ENVIRONMENT = "s:peace", "s:disarmament", "s:environment"

SUBJECTS = pd.DataFrame(
    {
        "subject_id": [PEACE, DISARMAMENT, ENVIRONMENT],
        "label_en": ["PEACE", "DISARMAMENT", "ENVIRONMENT"],
        "label_fr": ["PAIX", "DÉSARMEMENT", None],
        "node_type": ["scheme", "concept", "scheme"],
    }
)
RESOLUTION_SUBJECTS = pd.DataFrame(
    [("101", DISARMAMENT), ("104", DISARMAMENT), ("102", ENVIRONMENT), ("103", PEACE)]
    + [("105", PEACE)],
    columns=["undl_id", "subject_id"],
)
CLOSURE = pd.DataFrame(
    [(PEACE, PEACE, 0), (PEACE, DISARMAMENT, 1), (DISARMAMENT, DISARMAMENT, 0)]
    + [(ENVIRONMENT, ENVIRONMENT, 0)],
    columns=["ancestor_id", "descendant_id", "depth"],
)
KEYWORDS = pd.DataFrame(
    [
        ("101", "general", "nuclear disarmament"),
        ("101", "thematic", "disarmament"),
        ("104", "general", "nuclear-weapon-free zones"),
        ("104", "thematic", "disarmament"),
        ("103", "general", "women"),
        ("106", "general", "women"),
        ("102", "general", "climate change"),
        ("102", "geopolitical", "states"),
        ("102", "geopolitical", "oceans"),
        ("106", "action", "development"),
    ],
    columns=["undl_id", "dimension", "keyword"],
)


SPONSORS = pd.DataFrame(
    [
        ("101", "AAA land", "AAA", "state", "original", None),
        ("101", "BBB land", "BBB", "state", "additional", None),
        ("103", "AAA land", "AAA", "state", "original", None),
        ("103", "UN. General Assembly. President", None, "un_body", "original", None),
        ("104", "AAA land", "AAA", "state", "additional", None),
        ("106", "CCC land", "CCC", "state", "original", None),
        ("106", "Group of 77", None, "group", "original", None),
        ("102", None, None, None, None, "no draft backlinked, report to library"),
    ],
    columns=["undl_id", "sponsor", "country_code", "sponsor_type", "sponsor_kind", "note"],
)


def build_resolution_table() -> pd.DataFrame:
    from app.un_data_stream.data.repository import VOTE_DTYPE, sponsor_counts

    rows = []
    for undl_id, date, session, title, votes in RESOLUTIONS:
        recorded = votes != "XXXX"
        rows.append(
            {
                "undl_id": undl_id,
                "resolution": f"A/RES/{session}/{undl_id}",
                "date": pd.Timestamp(date),
                "session": session,
                "title": title,
                "agenda_title": title,
                "subjects": "",
                "draft": None,
                "total_yes": votes.count("Y") if recorded else None,
                "total_no": votes.count("N") if recorded else None,
                "total_abstentions": votes.count("A") if recorded else None,
                "total_non_voting": votes.count("X") if recorded else None,
                "total_ms": 4,
                "modality": "Vote, recorded" if recorded else "Without a vote",
                "undl_link": f"https://example.org/{undl_id}",
                **dict(zip(COUNTRIES, votes)),
            }
        )
    table = pd.DataFrame(rows)
    table[COUNTRIES] = table[COUNTRIES].astype(VOTE_DTYPE)
    table["sponsor_count"] = table["undl_id"].map(sponsor_counts(SPONSORS))
    return table


class _Repo:
    """Duck-types what `ResolutionQueryEngine` reads from a `DataRepository`."""

    def __init__(self, logger):
        from app.un_data_stream.data.processor import DataProcessor

        self.logger = logger
        self.source_marker = None
        table = build_resolution_table()
        consensus, multilateral, vote_arrays = DataProcessor({}, logger).calculate_agreement_data(
            table, COUNTRIES
        )
        table["consensus_score"] = table["undl_id"].map(consensus)
        self._data = {
            "resolution": table,
            "resolution_subject": RESOLUTION_SUBJECTS,
            "resolution_keyword": KEYWORDS,
            "resolution_sponsor": SPONSORS,
            "subject": SUBJECTS,
            "closure": CLOSURE,
            "country_columns": COUNTRIES,
            "multilateral_scores": multilateral,
            "vote_bool_arrays": vote_arrays,
        }

    def get_data(self):
        return self._data


@pytest.fixture
def engine():
    from app.un_data_stream.analysis.query_engine import ResolutionQueryEngine

    logger = logging.getLogger("synthetic_world")
    logger.addHandler(logging.NullHandler())
    return ResolutionQueryEngine(repo=_Repo(logger))
