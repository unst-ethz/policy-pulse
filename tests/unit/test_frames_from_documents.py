"""Unit tests for the MongoDB documents -> nine frames step of the load.

`_frames_from_documents` has to hand the rest of the load exactly what the Postgres tables used
to: same columns, same dtypes. The expected dtypes below are what `pd.read_sql_table` returned
on the local Postgres snapshot before the switch. No database needed.
"""

from datetime import datetime

import pandas as pd
import pytest
from bson.tz_util import utc  # the tzinfo PyMongo attaches with tz_aware=True

from app.un_data_stream.data.repository import DataRepository, clean_sponsors, sponsor_counts

LANGS = ("en", "es", "fr", "ar", "ru", "zh")
T = datetime(2026, 11, 3, 15, 40, tzinfo=utc)

RESOLUTIONS = [
    {
        "_id": "100",
        "resolution": "A/RES/80/1",
        "date": datetime(2025, 9, 4, tzinfo=utc),
        "session_label": "80",
        "title": "First",
        "agenda_title": "a",
        "subjects": "PEACE",
        "draft": None,
        "total_yes": 2,
        "total_no": 1,
        "total_abstentions": 0,
        "total_non_voting": 0,
        "total_ms": 3,
        "modality": "Vote, recorded",
        "votes": {"USA": "Y", "CHN": "Y", "FRA": "N"},
        "derived": {
            "subject_ids": ["S1", "S2"],
            "keywords": {
                "geopolitical": ["europe"],
                "thematic": ["peace", "europe"],
                "action": [],
            },
            "sponsors": [
                {
                    "sponsor": "Andorra",
                    "country_code": "AND",
                    "sponsor_type": "state",
                    "sponsor_kind": "original",
                    "source_field": "710",
                    "note": None,
                },
                # an observer: no `country_code`, absent rather than null
                {
                    "sponsor": "Holy See",
                    "sponsor_type": "observer",
                    "sponsor_kind": "additional",
                    "source_field": "710_add",
                    "note": None,
                },
            ],
        },
    },
    {
        # adopted without a vote, never derived: no `votes`, no `derived`
        "_id": "200",
        "resolution": "A/RES/80/2",
        "date": datetime(2025, 9, 5, tzinfo=utc),
        "session_label": "80",
        "title": "Second",
        "agenda_title": "b",
        "subjects": None,
        "draft": None,
        "total_yes": None,
        "total_no": None,
        "total_abstentions": None,
        "total_non_voting": None,
        "total_ms": None,
        "modality": "Without a vote",
    },
]


def _subject(subject_id, broader, ancestors, updated):
    labels = {
        f"{k}_{lang}": None for k in ("label", "alt_labels") for lang in "en es fr ar ru zh".split()
    }
    return {
        "_id": subject_id,
        "record_id": None,
        **labels,
        "label_en": subject_id.lower(),
        "domain_code": None,
        "node_type": "concept",
        "source_updated_at": updated,
        **({"broader": broader} if broader else {}),
        "ancestors": ancestors,
    }


SUBJECTS = [
    _subject("ROOT", [], [{"id": "ROOT", "depth": 0}], None),  # scheme node: no timestamp
    _subject("S1", ["ROOT"], [{"id": "S1", "depth": 0}, {"id": "ROOT", "depth": 1}], T),
    _subject("S2", ["ROOT"], [{"id": "S2", "depth": 0}, {"id": "ROOT", "depth": 1}], T),
]

MEMBER_STATES = [
    {
        "_id": "rec1",
        "iso_code": "CHE",
        "m49_code": "756",
        "record_type": "ms",
        "name_en": "Switzerland",
        **{f"name_{lang}": None for lang in ("fr", "es", "ar", "zh", "ru")},
        "other_names": None,
        "coverage_periods": "2002-",
        "founding_member": False,
        "membership_resolution": None,
        "scope_note": None,
        "earlier_names": None,
        "later_names": None,
        "subject_xref_id": 1,
        "source_updated_at": T,
        "inserted_at": T,
    }
]

OBJ, DATE, FLT, INT, BOOL = "object", "datetime64[ns]", "float64", "int64", "bool"
UTC = "datetime64[ns, UTC]"

# {frame: {column: dtype}} as the Postgres path produced them, column order included.
EXPECTED = {
    "outcomes": {
        "undl_id": OBJ, "resolution": OBJ, "date": DATE, "session_label": OBJ, "title": OBJ,
        "agenda_title": OBJ, "subjects": OBJ, "draft": OBJ, "total_yes": FLT, "total_no": FLT,
        "total_abstentions": FLT, "total_non_voting": FLT, "total_ms": FLT, "modality": OBJ,
    },
    "votes": {"undl_id": OBJ, "country_code": OBJ, "vote": OBJ},
    "resolution_subject": {"undl_id": OBJ, "subject_id": OBJ},
    "keyword": {"undl_id": OBJ, "dimension": OBJ, "keyword": OBJ},
    "subject": {
        "subject_id": OBJ, "record_id": OBJ,
        **{f"{k}_{lang}": OBJ for k in ("label", "alt_labels") for lang in LANGS},
        "domain_code": OBJ, "node_type": OBJ, "source_updated_at": UTC,
    },
    "closure": {"ancestor_id": OBJ, "descendant_id": OBJ, "depth": INT},
    "broader": {"parent_id": OBJ, "child_id": OBJ},
    "member_states": {
        "record_id": OBJ, "iso_code": OBJ, "m49_code": OBJ, "record_type": OBJ, "name_en": OBJ,
        "name_fr": OBJ, "name_es": OBJ, "name_ar": OBJ, "name_zh": OBJ, "name_ru": OBJ,
        "other_names": OBJ, "coverage_periods": OBJ, "founding_member": BOOL,
        "membership_resolution": OBJ, "scope_note": OBJ,
        "earlier_names": OBJ, "later_names": OBJ, "subject_xref_id": INT, "source_updated_at": UTC,
        "inserted_at": UTC,
    },
    "sponsor": {
        "undl_id": OBJ, "sponsor": OBJ, "country_code": OBJ, "sponsor_type": OBJ,
        "sponsor_kind": OBJ, "note": OBJ,
    },
}  # fmt: skip


@pytest.fixture
def frames():
    return DataRepository._frames_from_documents(RESOLUTIONS, SUBJECTS, MEMBER_STATES)


@pytest.mark.parametrize("name", EXPECTED)
def test_columns_and_dtypes_match_the_postgres_tables(frames, name):
    assert {c: str(t) for c, t in frames[name].dtypes.items()} == EXPECTED[name]


def test_outcomes_get_undl_id_back_and_a_naive_date(frames):
    outcomes = frames["outcomes"]
    assert outcomes["undl_id"].tolist() == ["100", "200"]
    assert outcomes["date"].tolist() == [pd.Timestamp("2025-09-04"), pd.Timestamp("2025-09-05")]


def test_votes_are_long_and_only_for_resolutions_with_votes(frames):
    assert sorted(map(tuple, frames["votes"].values.tolist())) == [
        ("100", "CHN", "Y"),
        ("100", "FRA", "N"),
        ("100", "USA", "Y"),
    ]


def test_resolution_subject_unwinds_subject_ids(frames):
    assert frames["resolution_subject"].values.tolist() == [["100", "S1"], ["100", "S2"]]


def test_keyword_general_is_the_deduplicated_union(frames):
    rows = set(map(tuple, frames["keyword"].values.tolist()))
    assert rows == {
        ("100", "geopolitical", "europe"),
        ("100", "thematic", "peace"),
        ("100", "thematic", "europe"),
        ("100", "general", "europe"),
        ("100", "general", "peace"),
    }
    assert len(frames["keyword"]) == len(rows)


def test_subject_hierarchy_is_unwound(frames):
    assert sorted(map(tuple, frames["closure"].values.tolist())) == [
        ("ROOT", "ROOT", 0),
        ("ROOT", "S1", 1),
        ("ROOT", "S2", 1),
        ("S1", "S1", 0),
        ("S2", "S2", 0),
    ]
    assert frames["broader"].values.tolist() == [["ROOT", "S1"], ["ROOT", "S2"]]
    assert frames["subject"]["subject_id"].tolist() == ["ROOT", "S1", "S2"]
    assert frames["member_states"]["record_id"].tolist() == ["rec1"]


def test_frames_feed_the_unchanged_wide_table_build(frames):
    table, country_columns = DataRepository._build_resolution_table(
        frames["outcomes"], frames["votes"]
    )
    assert country_columns == ["CHN", "FRA", "USA"]
    assert table.set_index("undl_id").loc["200", country_columns].tolist() == ["X", "X", "X"]


def test_empty_collections_still_give_every_column():
    frames = DataRepository._frames_from_documents([], [], [])
    for name, columns in EXPECTED.items():
        assert list(frames[name].columns) == list(columns), name
    # downstream reads `sponsor` with `.str`, so the empty frame must keep object dtype too
    assert {c: str(t) for c, t in frames["sponsor"].dtypes.items()} == EXPECTED["sponsor"]


def test_sponsors_unwind_with_absent_fields_as_none(frames):
    assert frames["sponsor"].values.tolist() == [
        ["100", "Andorra", "AND", "state", "original", None],
        ["100", "Holy See", None, "observer", "additional", None],
    ]


def test_sponsor_placeholders_and_resolutions_without_sponsors():
    placeholder = {
        "sponsor": None,
        "country_code": None,
        "sponsor_type": None,
        "sponsor_kind": None,
        "source_field": None,
        "note": "draft has no authors catalogued",
    }
    resolutions = [
        {"_id": "300", "derived": {"sponsors": [placeholder]}},
        {"_id": "400", "derived": {"sponsors": [{"note": "no draft backlinked"}]}},
        {"_id": "500", "derived": {}},  # absent `derived.sponsors`: no rows
        {"_id": "600"},
    ]
    sponsors = DataRepository._frames_from_documents(resolutions, [], [])["sponsor"]
    assert sponsors.values.tolist() == [
        ["300", None, None, None, None, "draft has no authors catalogued"],
        ["400", None, None, None, None, "no draft backlinked"],
    ]
    assert sponsor_counts(clean_sponsors(sponsors)).empty
