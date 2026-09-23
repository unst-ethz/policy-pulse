"""Membership and voting-activity dates, against the real Postgres tables.

These replace `app/assets/joining_dates.csv`, which was a frozen extract of first/last *vote*
dates whose end was pinned to whatever the last resolution was the day it was generated. The
tests below assert the two properties that file could not have: that the dates track the data,
and that no country's votes fall outside the window the filter derives for it.

Assertions are about behaviour, not about upstream strings — states are renamed and readmitted,
so a failure here means our logic broke, not that the UN edited a record.
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.needs_postgres


@pytest.fixture(scope="module")
def data():
    from app import data as app_data

    return app_data


def test_every_voting_code_has_membership_periods(data):
    """Including the legacy codes (GER, SCG), which have no member_states row of their own."""
    missing = [c for c in data.available_countries if not data.get_membership_periods(c)]

    assert not missing, f"codes the membership filter would not narrow at all: {missing}"


def test_participation_range_covers_every_vote_actually_cast(data):
    """The property the CSV violated at both ends.

    The year range the UI clamps to must contain every vote a code cast, or the profile page
    hides data — BFA voted as Upper Volta from 1960, 24 years before its record begins.
    """
    uncovered = []
    for iso in data.available_countries:
        first, last = data.get_voting_activity(iso)
        first_year, last_year = data.get_participation_year_range(iso)
        if first.year < first_year or last.year > last_year:
            uncovered.append((iso, (first.year, last.year), (first_year, last_year)))

    assert not uncovered, f"votes outside the clamped year range: {uncovered}"


def test_end_date_tracks_the_data(data):
    """No frozen end: the latest activity in the app is the latest resolution in the database."""
    latest_activity = max(
        last for _, last in (data.get_voting_activity(iso) for iso in data.available_countries)
    )

    assert latest_activity == data.get_latest_data_date()


def test_legacy_code_takes_its_predecessor_period_off_the_successor(data):
    """West Germany's record sits under DEU, but its votes are GER's — so is its period."""
    ((ger_start, ger_end),) = data.get_membership_periods("GER")
    ((deu_start, deu_end),) = data.get_membership_periods("DEU")

    assert ger_start.year == 1973 and ger_end.year == 1990
    assert deu_start.year == 1990 and deu_start > ger_end
    assert deu_end is None, "Germany is still a member, so its period must stay open"


def test_successive_names_merge_into_one_period(data):
    """Burma to 1989-06-17 and Myanmar from 1989-06-18 are one membership, not two."""
    periods = data.get_membership_periods("MMR")

    assert len(periods) == 1, f"expected one continuous period, got {periods}"
    assert periods[0][0].year == 1948
    assert periods[0][1] is None


def test_a_real_gap_survives(data):
    """SYR left in 1958 (United Arab Republic) and returned in 1961, casting no votes between."""
    periods = data.get_membership_periods("SYR")

    assert len(periods) == 2, f"expected the 1958-1961 gap to survive, got {periods}"
    in_gap = data.membership_mask("SYR", pd.Series([pd.Timestamp("1959-06-01")]))
    assert not in_gap.any()

    outside_gap = data.membership_mask(
        "SYR", pd.Series([pd.Timestamp("1957-06-01"), pd.Timestamp("1990-06-01")])
    )
    assert outside_gap.all()


def test_unknown_code_does_not_empty_the_table(data):
    mask = data.membership_mask("ZZZ", pd.Series([pd.Timestamp("1960-01-01")]))

    assert mask.all()


def test_participation_range_stays_inside_the_year_slider(data):
    """A founding member joined in 1945, a year before the first resolution the app has.

    The range feeds a slider bounded by the data, so a 1945 start would render clamped while the
    store and URL disagreed with it.
    """
    outside = [
        (iso, years)
        for iso in data.available_countries
        if (years := data.get_participation_year_range(iso))
        and not (data.get_earliest_year() <= years[0] <= years[1] <= data.get_latest_year())
    ]

    assert not outside, f"ranges the year slider cannot represent: {outside}"


def test_participation_range_of_a_current_member_ends_with_the_data(data):
    first_year, last_year = data.get_participation_year_range("CHE")

    assert first_year == 2002
    assert last_year == data.get_latest_year()
