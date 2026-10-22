"""Word cloud term ranking over the synthetic world in conftest.py."""

import pytest

from app.un_data_stream.analysis import keywords, word_cloud
from app.un_data_stream.analysis.word_cloud import CloudTerm


@pytest.fixture
def index(engine):
    return keywords.KeywordIndex.build(engine.snapshot)


def test_terms_are_ranked_by_searchable_count(engine, index):
    terms = word_cloud.cloud_terms(index, engine.query_resolutions(), "default")
    assert terms == [
        CloudTerm("women", 2, 2),
        CloudTerm("climate change", 1, 1),
        CloudTerm("nuclear disarmament", 1, 1),
        CloudTerm("nuclear-weapon-free zones", 1, 1),
    ]


def test_an_active_keyword_narrows_what_a_click_finds(engine, index):
    terms = word_cloud.cloud_terms(index, engine.query_resolutions(), "default", keyword="nuclear")
    assert [t.term for t in terms] == ["nuclear disarmament", "nuclear-weapon-free zones"]


def test_limit(engine, index):
    terms = word_cloud.cloud_terms(index, engine.query_resolutions(), "default", limit=2)
    assert [t.term for t in terms] == ["women", "climate change"]


def test_category_terms_count_within_the_given_resolutions(engine, index):
    df = engine.query_resolutions(start_date="2001-01-01")
    terms = word_cloud.cloud_terms(index, df, "category")
    assert terms == [CloudTerm("peace", 2, 2), CloudTerm("disarmament", 1, 1)]


def test_no_resolutions_no_terms(engine, index):
    assert word_cloud.cloud_terms(index, engine.query_resolutions().iloc[0:0], "default") == []


def test_mean_consensus_per_term(engine, index):
    df = engine.query_resolutions()
    scores = word_cloud.mean_consensus(index, ["women", "unknown"], "default", df)
    expected = df.set_index("undl_id").loc[["103", "106"], "consensus_score"].mean()
    assert scores == [pytest.approx(expected), None]
