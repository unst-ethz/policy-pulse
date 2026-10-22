"""Keyword index building, the keyword search expression, and the per-snapshot cache."""

import dataclasses

import pandas as pd
import pytest

from app.un_data_stream.analysis import keywords

from .conftest import DISARMAMENT, PEACE


def _pairs(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["undl_id", "keyword"])


@pytest.fixture
def index(engine):
    return keywords.KeywordIndex.build(engine.snapshot)


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------


def test_a_term_containing_a_comma_is_indexed_whole():
    """The CSV assets this replaced were split on `[;,]` at read time, because they packed a
    resolution's keywords into one cell. The table stores one term per row, and ~100 of those
    terms contain a comma -- re-introducing a split would shatter each into fragments that match
    no resolution and clutter the cloud with words nobody can click through.
    """
    term = "national or ethnic, religious and linguistic minorities"
    resolution_data, word_map = keywords.build_keyword_indices(_pairs([("1", term)]))

    assert list(word_map) == [term]
    assert list(resolution_data["1"]["word_freq"]) == [term]


def test_excluded_terms_leave_neither_index():
    """A term dropped from the cloud but left in the reverse index would still be searchable and
    still colour the resolution list, from a word the user was never shown."""
    resolution_data, word_map = keywords.build_keyword_indices(
        _pairs([("1", "states"), ("1", "cyprus")]), exclude={"states"}
    )

    assert set(word_map) == {"cyprus"}
    assert set(resolution_data["1"]["word_freq"]) == {"cyprus"}


def test_a_resolution_counts_once_per_term():
    """The cloud sizes a word by how many resolutions carry it. A per-resolution count above 1
    would let one resolution outvote several."""
    resolution_data, word_map = keywords.build_keyword_indices(
        _pairs([("1", "disarmament"), ("2", "disarmament")])
    )

    assert set(resolution_data["1"]["word_freq"].values()) == {1}
    assert word_map["disarmament"] == ["1", "2"]


def test_each_mode_reads_its_own_dimension(index):
    assert set(index.terms("default")) == {
        "nuclear disarmament",
        "nuclear-weapon-free zones",
        "women",
        "climate change",
    }
    assert index.terms("thematic") == {"disarmament": ["101", "104"]}
    assert index.terms("action") == {"development": ["106"]}


def test_geopolitical_exclusions_apply(index):
    assert "states" not in index.terms("geopolitical")
    assert index.terms("geopolitical") == {"oceans": ["102"]}


def test_category_mode_indexes_subject_labels(index):
    assert set(index.terms("category")) == {"peace", "disarmament", "environment"}
    assert sorted(index.terms("category")["peace"]) == ["103", "105"]
    assert index.category_term_to_subject_ids == {
        "peace": [PEACE],
        "disarmament": [DISARMAMENT],
        "environment": ["s:environment"],
    }


def test_keywords_of_unserved_resolutions_are_ignored(engine):
    snapshot = dataclasses.replace(
        engine.snapshot,
        keyword_table=pd.concat(
            [
                engine.snapshot.keyword_table,
                pd.DataFrame(
                    [("999", "general", "ghost")], columns=["undl_id", "dimension", "keyword"]
                ),
            ]
        ),
    )
    assert "ghost" not in keywords.KeywordIndex.build(snapshot).terms("default")


def test_an_empty_keyword_table_gives_empty_modes(engine):
    snapshot = dataclasses.replace(
        engine.snapshot, keyword_table=pd.DataFrame(columns=["undl_id", "dimension", "keyword"])
    )
    index = keywords.KeywordIndex.build(snapshot)
    assert index.terms("default") == {}
    assert index.terms("category"), "subject terms do not depend on the keyword table"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_exact_search_needs_the_whole_term(index):
    assert index.search("women", exact=True) == {"103", "106"}
    assert index.search("wom", exact=True) == set()


def test_loose_search_matches_substrings(index):
    assert index.search("nuclear") == {"101", "104"}


def test_term_frequencies_sum_over_the_given_resolutions(index):
    freq = index.aggregate_term_frequencies(pd.Series(["103", "106", "101"]))
    assert freq == {"women": 2, "nuclear disarmament": 1}


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("women", {"103", "106"}),
        ("nuclear & zones", {"104"}),  # AND within a clause
        ("climate, women", {"102", "103", "106"}),  # OR across clauses
        ('"women"', {"103", "106"}),  # exact keyword
        ("peace", {"105"}),  # title match, no keyword needed
        ("women & nuclear", set()),
        ("", set()),
        ("  ,  ", set()),
    ],
)
def test_keyword_expressions(engine, index, expression, expected):
    df = engine.query_resolutions()
    assert keywords.matched_ids(df, expression, index) == expected


def test_quotes_mark_an_exact_term():
    assert keywords.parse_keyword_term(' "human rights" ') == ("human rights", True)
    assert keywords.parse_keyword_term("human rights") == ("human rights", False)
    assert keywords.parse_keyword_term('"') == ('"', False)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_the_index_is_reused_for_the_same_snapshot(engine):
    first = keywords.keyword_index(engine.snapshot)
    assert keywords.keyword_index(engine.snapshot) is first


def test_a_new_snapshot_gets_a_new_index(engine):
    first = keywords.keyword_index(engine.snapshot)
    replaced = dataclasses.replace(engine.snapshot)
    assert keywords.keyword_index(replaced) is not first
