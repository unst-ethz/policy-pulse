"""Country profile and dataset overview over the synthetic world in conftest.py."""

import pytest

from app.un_data_stream.analysis import overview, profile

from .conftest import ENVIRONMENT, PEACE


def test_profile_counts_votes_in_range(engine):
    prof = profile.country_profile(engine, "AAA", participation=(2001, 2002))
    assert (prof.start_date, prof.end_date) == ("2001-01-01", "2002-12-31")
    assert prof.total_resolutions == 4  # 103..106
    # 105 was adopted without a vote, so the tally covers 103, 104 and 106 only
    assert prof.recorded_votes == 3
    assert prof.votes == {"Y": 1, "N": 1, "A": 1, "absent": 0}
    assert (prof.adopted_without_vote, prof.non_recorded_votes) == (1, 0)


def test_profile_narrowed_to_subjects(engine):
    # PEACE takes in DISARMAMENT below it: 101, 103, 104, 105; ENVIRONMENT is 102 only
    assert profile.country_profile(engine, "AAA", subject_ids=[PEACE]).total_resolutions == 4
    prof = profile.country_profile(engine, "AAA", subject_ids=[ENVIRONMENT])
    assert (prof.total_resolutions, prof.votes["Y"]) == (1, 1)


def test_profile_comparison_falls_back_to_the_p5(engine):
    prof = profile.country_profile(engine, "AAA", comparison=["ZZZ", "AAA"])
    assert prof.comparison == profile.P5
    assert prof.comparison_is_default
    assert prof.yearly_alignment.empty  # no P5 member in this world


def test_profile_alignment(engine):
    prof = profile.country_profile(engine, "AAA", comparison=["BBB"], min_joint_votes=4)
    assert not prof.comparison_is_default
    assert prof.most_aligned.index[0] == "BBB"
    assert prof.most_aligned["BBB"] == pytest.approx(0.7)
    assert prof.least_aligned.iloc[-1] == pytest.approx(0.7)
    assert prof.yearly_alignment["BBB"].tolist() == pytest.approx([1.0, 0.5, 0.5])
    assert 1 <= prof.multilateral_rank <= prof.multilateral_ranked == 4


def test_profile_ranking_needs_enough_joint_votes(engine):
    prof = profile.country_profile(engine, "AAA", min_joint_votes=5)
    assert list(prof.most_aligned.index) == ["BBB", "CCC"]  # DDD shares only 4 votes


def test_profile_notable_votes(engine):
    prof = profile.country_profile(engine, "AAA")
    # Opposed (N/A) highest consensus first; supported (Y) lowest consensus first.
    assert prof.opposed_high_consensus["undl_id"].tolist() == ["106", "104"]
    assert prof.supported_low_consensus["undl_id"].tolist() == ["101", "103", "102"]


def test_profile_of_an_empty_range_is_none(engine):
    assert profile.country_profile(engine, "AAA", start_date="2010-01-01") is None


def test_dataset_stats(engine):
    stats = overview.dataset_stats(engine)
    assert stats.total_resolutions == 6
    assert (stats.year_min, stats.year_max) == (2000, 2002)
    assert stats.countries == 4
    assert (stats.subjects, stats.subject_links, stats.resolutions_with_subjects) == (3, 5, 5)
    assert stats.votes == {"Y": 12, "N": 5, "A": 2, "X": 1}


def test_recent_resolutions_are_newest_first_with_a_tag(engine):
    recent = overview.recent_resolutions(engine)
    assert recent["undl_id"].tolist()[:2] == ["106", "105"]
    assert recent["category_tag"].tolist()[:2] == [overview.NO_CATEGORY, "PEACE"]
