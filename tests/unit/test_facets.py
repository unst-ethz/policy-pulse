"""Facet counts over the synthetic world in conftest.py."""

from app.un_data_stream.analysis import facets, filtering
from app.un_data_stream.analysis.filtering import ResolutionFilter

from .conftest import ENVIRONMENT, PEACE

# PEACE (with DISARMAMENT below it) is 101, 103, 104, 105; AAA sponsored 101, 103, 104.
CRITERIA = ResolutionFilter(start_date="2001-01-01", subject_ids=(PEACE,), sponsor="AAA")


def test_years_ignore_the_period_but_keep_the_other_filters(engine):
    assert facets.year_counts(engine, CRITERIA).to_dict() == {2000: 1, 2001: 2}


def test_domains_ignore_the_subjects_but_keep_the_other_filters(engine):
    # From 2001 and sponsored by AAA: 103 (PEACE) and 104 (DISARMAMENT, so PEACE)
    counts = facets.domain_counts(engine, CRITERIA, [PEACE, ENVIRONMENT])
    assert counts.to_dict() == {PEACE: 2}


def test_unfiltered_domains_count_each_resolution_once(engine):
    counts = facets.domain_counts(engine, ResolutionFilter(), [PEACE, ENVIRONMENT])
    assert counts.to_dict() == {PEACE: 4, ENVIRONMENT: 1}


def test_modalities_ignore_their_own_filter(engine):
    # 105 nobody voted on: adopted without a vote; the other PEACE ones were recorded votes
    criteria = ResolutionFilter(subject_ids=(PEACE,), modalities=("Without a vote",))
    assert len(filtering.filter_resolutions(engine, criteria)) == 1
    counts = facets.modality_counts(engine, criteria)
    assert counts.to_dict() == {"Vote, recorded": 3, "Without a vote": 1}


def ids(engine, **criteria):
    df = filtering.filter_resolutions(engine, ResolutionFilter(**criteria))
    return sorted(df["undl_id"])


def test_group_votes_judge_only_the_states_that_voted(engine):
    # AAA/BBB: 101 YY, 102 YY, 103 YN, 104 NN, 105 nobody voted, 106 AY
    pair = ("AAA", "BBB")
    assert ids(engine, group=pair, group_vote="alike") == ["101", "102", "104"]
    assert ids(engine, group=pair, group_vote="split") == ["103", "106"]
    assert ids(engine, group=pair, group_vote="yes") == ["101", "102"]
    counts = facets.group_vote_counts(engine, ResolutionFilter(group=pair, group_vote="no"))
    assert counts == {"alike": 3, "split": 2, "yes": 2, "no": 1, "abstain": 0}


def test_group_votes_can_require_every_state_to_have_voted(engine):
    # DDD did not vote on 102, so AAA and BBB alone decide it, unless all must have voted
    trio = ("AAA", "BBB", "DDD")
    assert ids(engine, group=trio, group_vote="alike") == ["102"]
    assert ids(engine, group=trio, group_vote="alike", group_all_voted=True) == []
    assert ids(engine, group=trio, group_vote="split", group_all_voted=True) == [
        "101",
        "103",
        "104",
        "106",
    ]


def test_a_group_of_one_matches_nothing(engine):
    assert ids(engine, group=("AAA",), group_vote="alike") == []
