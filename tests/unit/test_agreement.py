"""Agreement analyses over the synthetic world in conftest.py (expected values derived there)."""

import math

import pandas as pd
import pytest

from app.un_data_stream.analysis import agreement

from .conftest import DISARMAMENT, ENVIRONMENT, PEACE

ALL_IDS = ["101", "102", "103", "104", "105", "106"]


def test_agreement_map_averages_over_shared_votes(engine):
    scores = agreement.agreement_map(engine, "AAA", ALL_IDS)
    assert scores.to_dict() == pytest.approx({"BBB": 0.7, "CCC": 0.5, "DDD": 0.5})


def test_agreement_map_is_nan_without_a_shared_vote(engine):
    scores = agreement.agreement_map(engine, "AAA", ["102"])
    assert scores["BBB"] == 1.0
    assert math.isnan(scores["DDD"])


def test_agreement_map_of_an_unknown_country_is_empty(engine):
    assert agreement.agreement_map(engine, "ZZZ", ALL_IDS).empty


def test_session_agreement(engine):
    out = agreement.session_agreement(engine, "AAA", ["BBB", "DDD"], min_votes=1)
    assert out["session"].tolist() == ["55", "56", "57"]
    assert out["year"].tolist() == [2000, 2001, 2002]
    assert out["avg_agreement_BBB"].tolist() == pytest.approx([1.0, 0.5, 0.5])
    assert out["n_votes_BBB"].tolist() == [2, 2, 1]
    assert out["avg_agreement_DDD"].tolist() == pytest.approx([0.5, 0.5, 0.5])


def test_sessions_below_the_vote_minimum_have_no_value(engine):
    out = agreement.session_agreement(engine, "AAA", ["BBB"], min_votes=2)
    assert out["avg_agreement_BBB"].isna().tolist() == [False, False, True]


def test_special_sessions_fold_into_the_regular_session_in_progress():
    sessions = pd.Series(["32", "8sp", "33", "10emsp", None, "1sp"])
    dates = pd.to_datetime(
        pd.Series(
            ["1977-11-01", "1978-04-20", "1978-11-01", "2003-05-01", "1979-01-01", "1970-01-01"]
        )
    )
    folded = agreement.fold_into_regular_sessions(sessions, dates)
    # 1sp predates every regular session, so it keeps its own label
    assert folded.tolist() == ["32", "32", "33", "33", "33", "1sp"]
    # Kept apart, only the missing label is folded
    kept = agreement.fold_into_regular_sessions(sessions, dates, keep_special=True)
    assert kept.tolist() == ["32", "8sp", "33", "10emsp", "33", "1sp"]


def test_session_agreement_needs_another_country(engine):
    assert agreement.session_agreement(engine, "AAA", ["AAA"]) is None


def test_agreement_by_subject_includes_descendants(engine):
    labels = {PEACE: "PEACE", DISARMAMENT: "DISARMAMENT"}
    out = agreement.agreement_by_subject(
        engine, "AAA", ["BBB"], None, None, [PEACE, ENVIRONMENT, DISARMAMENT], labels, min_votes=1
    )
    assert out["subject_id"].tolist() == [DISARMAMENT, ENVIRONMENT, PEACE]
    assert out["subject_label"].tolist() == ["DISARMAMENT", f"ID: {ENVIRONMENT}", "PEACE"]
    assert out["agreement_score"].tolist() == pytest.approx([1.0, 1.0, 2 / 3])
    assert out["total_votes"].tolist() == [2, 1, 3]


def test_agreement_by_subject_drops_thin_subjects(engine):
    out = agreement.agreement_by_subject(
        engine, "AAA", ["BBB"], None, None, [PEACE, ENVIRONMENT], {}, min_votes=2
    )
    assert out["subject_id"].tolist() == [PEACE]
    assert agreement.agreement_by_subject(engine, "AAA", ["BBB"], None, None, [PEACE], {}).empty


def test_agreement_by_subject_with_several_countries(engine):
    out = agreement.agreement_by_subject(
        engine, "AAA", ["BBB", "AAA", "DDD", "ZZZ"], None, None, [PEACE], {}, min_votes=1
    )
    # Itself and unknown countries are skipped; one row per qualifying (subject, country) pair
    assert out["country"].tolist() == ["BBB", "DDD"]
    bbb = out[out["country"] == "BBB"].iloc[0]
    assert (bbb["agreement_score"], bbb["total_votes"]) == (pytest.approx(2 / 3), 3)


def test_agreement_by_subject_respects_dates(engine):
    out = agreement.agreement_by_subject(
        engine, "AAA", ["BBB"], "2001-01-01", None, [PEACE], {}, min_votes=1
    )
    assert out["total_votes"].tolist() == [2]  # 103 and 104; 105 has no votes


def test_multilateral_overview_applies_the_vote_minimum(engine):
    out = agreement.multilateral_overview(engine, ALL_IDS, min_votes=5)
    assert sorted(out["country"]) == ["AAA", "BBB", "CCC"]  # DDD voted on 4
    assert set(out.columns) >= {"multilateral_alignment", "yes_rate", "participation_count"}


def test_multilateral_overview_of_nothing_is_empty(engine):
    # Not the whole dataset: the engine itself reads an empty id list as "all resolutions"
    assert agreement.multilateral_overview(engine, [], min_votes=1).empty


def test_consensus_scale_anchors_on_the_mean():
    lo, avg, hi, midpoint = agreement.consensus_scale(pd.Series([0.2, 0.4, 0.9]), lo=0, hi=1)
    assert (lo, hi) == (0, 1)
    assert avg == pytest.approx(0.5)
    assert midpoint == pytest.approx(0.5)
