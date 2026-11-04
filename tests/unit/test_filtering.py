"""The shared resolution filter and the list operations built on it."""

import pandas as pd
import pytest

from app.un_data_stream.analysis import filtering
from app.un_data_stream.analysis.filtering import ResolutionFilter

from .conftest import PEACE


def _ids(df: pd.DataFrame) -> list[str]:
    return sorted(df["undl_id"])


def test_no_criteria_returns_everything_with_the_list_columns(engine):
    df = filtering.filter_resolutions(engine, ResolutionFilter())
    assert _ids(df) == ["101", "102", "103", "104", "105", "106"]
    assert list(df.columns) == filtering.COLUMNS


def test_dates_are_inclusive(engine):
    criteria = ResolutionFilter(start_date="2000-06-01", end_date="2001-09-01")
    assert _ids(filtering.filter_resolutions(engine, criteria)) == ["102", "103", "104"]


def test_a_subject_includes_its_descendants(engine):
    criteria = ResolutionFilter(subject_ids=(PEACE,))
    assert _ids(filtering.filter_resolutions(engine, criteria)) == ["101", "103", "104", "105"]


def test_no_subject_sentinel(engine):
    criteria = ResolutionFilter(subject_ids=("__no_subject__",))
    assert _ids(filtering.filter_resolutions(engine, criteria)) == ["106"]


def test_voted_mode_keeps_resolutions_the_country_voted_on(engine):
    criteria = ResolutionFilter(country="DDD", country_mode="voted")
    assert _ids(filtering.filter_resolutions(engine, criteria)) == ["101", "103", "104", "106"]


def test_member_mode_ors_membership_with_having_voted(engine):
    def member_from_2001(_iso3, dates):
        return dates >= pd.Timestamp("2001-01-01")

    criteria = ResolutionFilter(country="DDD", country_mode="member")
    df = filtering.filter_resolutions(engine, criteria, membership_mask=member_from_2001)
    # 101 predates membership but DDD voted on it; 102 predates it and DDD did not vote.
    assert _ids(df) == ["101", "103", "104", "105", "106"]


def test_member_mode_needs_a_membership_mask(engine):
    with pytest.raises(ValueError):
        filtering.filter_resolutions(engine, ResolutionFilter(country="DDD", country_mode="member"))


def test_country_and_keyword_filters_can_be_switched_off(engine):
    criteria = ResolutionFilter(country="DDD", country_mode="voted", keyword="climate")
    assert _ids(filtering.filter_resolutions(engine, criteria)) == []
    assert _ids(filtering.filter_resolutions(engine, criteria, apply_country=False)) == ["102"]
    assert len(filtering.filter_resolutions(engine, criteria, apply_keyword=False)) == 4


def test_an_unknown_country_does_not_filter(engine):
    criteria = ResolutionFilter(country="ZZZ", country_mode="voted")
    assert len(filtering.filter_resolutions(engine, criteria)) == 6


def test_columns_none_keeps_the_vote_columns(engine):
    df = filtering.filter_resolutions(engine, ResolutionFilter(), columns=None)
    assert {"AAA", "BBB", "CCC", "DDD"} <= set(df.columns)


def test_attach_votes_skips_unknown_codes(engine):
    df = filtering.filter_resolutions(engine, ResolutionFilter())
    with_votes = filtering.attach_votes(engine, df, ["AAA", "ZZZ", None, "AAA"])
    assert list(with_votes.columns) == filtering.COLUMNS + ["AAA"]
    assert with_votes.set_index("undl_id").loc["104", "AAA"] == "N"


@pytest.mark.parametrize(
    "order, first",
    [
        ("date_desc", "106"),
        ("date_asc", "101"),
        ("consensus_asc", "104"),
        ("consensus_desc", "102"),
        (None, "106"),
    ],
)
def test_sort_orders(engine, order, first):
    df = filtering.filter_resolutions(engine, ResolutionFilter())
    assert filtering.sort_resolutions(df, order)["undl_id"].iloc[0] == first


def test_ties_sort_the_same_whatever_the_input_order():
    df = pd.DataFrame(
        {
            "undl_id": ["3", "1", "4", "2"],
            "date": pd.to_datetime(["2026-01-01"] * 3 + ["2025-01-01"]),
            "consensus_score": [0.5, 0.5, None, 0.5],
        }
    )
    for order in filtering.SORT_ORDERS:
        forward = filtering.sort_resolutions(df, order)["undl_id"].tolist()
        backward = filtering.sort_resolutions(df.iloc[::-1], order)["undl_id"].tolist()
        assert forward == backward, order
    assert filtering.sort_resolutions(df, "date_desc")["undl_id"].tolist() == ["4", "3", "1", "2"]


def test_unscored_resolutions_sort_last(engine):
    df = filtering.filter_resolutions(engine, ResolutionFilter())
    for order in ("consensus_asc", "consensus_desc"):
        assert filtering.sort_resolutions(df, order)["undl_id"].iloc[-1] == "105"


@pytest.mark.parametrize(
    "agreement, expected",
    [
        # 105 counts as "agreed": both did not vote (see the docstring's note).
        ("AGREED", ["101", "102", "104", "105"]),
        ("DISAGREED", ["103", "106"]),
        ("STRONGLY_DISAGREED", ["103"]),
    ],
)
def test_pair_agreement(engine, agreement, expected):
    df = filtering.attach_votes(
        engine, filtering.filter_resolutions(engine, ResolutionFilter()), ["AAA", "BBB"]
    )
    assert _ids(filtering.filter_by_pair_agreement(df, "AAA", "BBB", agreement)) == expected


def test_filter_by_vote(engine):
    df = filtering.attach_votes(
        engine, filtering.filter_resolutions(engine, ResolutionFilter()), ["AAA"]
    )
    assert _ids(filtering.filter_by_vote(df, "AAA", "Y")) == ["101", "102", "103"]
    assert _ids(filtering.filter_by_vote(df, "BBB", "Y")) == _ids(df), "no column: unfiltered"


def test_export_names_vote_columns_and_ignores_the_country_filter(engine):
    criteria = ResolutionFilter(
        subject_ids=(PEACE,), country="DDD", country_mode="voted", keyword="peace, nuclear"
    )
    df = filtering.export_frame(engine, criteria, ["AAA", "AAA", "ZZZ"], lambda c: f"Land {c}")
    # 105 stays although DDD did not vote on it
    assert _ids(df) == ["101", "104", "105"]
    assert list(df.columns) == filtering.EXPORT_COLUMNS + ["Land AAA"]


def test_filter_store_dicts_translate(engine):
    criteria = ResolutionFilter.from_filter_store(
        {
            "start_date": "2000-01-01",
            "end_date": "2000-12-31",
            "subject_ids": [],
            "country1_alpha3": "AAA",
            "country_filter_mode": None,
            "keyword": None,
        }
    )
    assert criteria == ResolutionFilter(
        start_date="2000-01-01", end_date="2000-12-31", country="AAA"
    )
