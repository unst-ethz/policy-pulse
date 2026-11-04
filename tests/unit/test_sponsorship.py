"""Sponsorship over the synthetic world in conftest.py."""

import pandas as pd

from app.un_data_stream.analysis import filtering, sponsorship
from app.un_data_stream.analysis.filtering import ResolutionFilter
from app.un_data_stream.data.repository import clean_sponsors

from .conftest import ENVIRONMENT, PEACE


def ids(df):
    return sorted(df["undl_id"])


def test_filter_by_sponsor(engine):
    def select(**criteria):
        return ids(filtering.filter_resolutions(engine, ResolutionFilter(**criteria)))

    assert select(sponsor="AAA") == ["101", "103", "104"]
    assert select(sponsor="AAA", original_sponsor=True) == ["101", "103"]
    assert select(sponsor="DDD") == []
    # 102 has only a placeholder and 105 no rows: both unknown
    assert select(sponsorship_known=True) == ["101", "103", "104", "106"]


def test_sponsor_count_is_null_when_unknown(engine):
    counts = engine.resolution_table.set_index("undl_id")["sponsor_count"]
    assert counts[["101", "103", "104", "106"]].tolist() == [2, 2, 1, 2]
    assert counts[["102", "105"]].isna().all()


def test_resolution_sponsors(engine):
    rows, note = sponsorship.resolution_sponsors(engine.snapshot, "101")
    assert rows["country_code"].tolist() == ["AAA", "BBB"] and note is None

    rows, note = sponsorship.resolution_sponsors(engine.snapshot, "102")
    assert rows.empty and note == "no draft backlinked, report to library"

    rows, note = sponsorship.resolution_sponsors(engine.snapshot, "105")
    assert rows.empty and note is None


def test_country_sponsorship(engine):
    record = sponsorship.country_sponsorship(engine, "AAA", None, None, [PEACE, ENVIRONMENT])
    assert (record.resolutions, record.with_data) == (6, 4)
    assert (record.sponsored, record.original) == (3, 2)
    assert record.per_year.to_dict("records") == [
        {"year": 2000, "with_data": 1, "original": 1, "additional": 0},
        {"year": 2001, "with_data": 2, "original": 1, "additional": 1},
        {"year": 2002, "with_data": 1, "original": 0, "additional": 0},
    ]
    # 101 and 104 count under PEACE through DISARMAMENT
    assert record.by_subject.to_dict() == {PEACE: 3}
    # votes on 101, 103, 104: BBB Y N N, CCC N Y Y, DDD A Y Y
    assert record.reception.to_dict("records") == [
        {"country": "BBB", "yes": 1, "no": 2, "abstain": 0},
        {"country": "CCC", "yes": 2, "no": 1, "abstain": 0},
        {"country": "DDD", "yes": 2, "no": 0, "abstain": 1},
    ]
    assert ids(record.not_yes) == ["104"]  # AAA sponsored it, then voted no


def test_country_sponsorship_in_a_period(engine):
    record = sponsorship.country_sponsorship(engine, "AAA", "2001-01-01", "2001-12-31", [PEACE])
    assert (record.resolutions, record.with_data, record.sponsored, record.original) == (2, 2, 2, 1)


def test_clean_sponsors():
    raw = pd.DataFrame(
        {
            "undl_id": ["1", "1", "1", "2"],
            "sponsor": ["the  Bahamas", "Palestine", "State of  Palestine", None],
            "country_code": ["BHS", None, None, None],
            "sponsor_type": ["state", "observer", "observer", None],
            "sponsor_kind": ["original"] * 3 + [None],
            "note": [None, None, None, "draft has no authors catalogued"],
        }
    )
    names = clean_sponsors(raw)["sponsor"].tolist()
    assert names[:3] == ["Bahamas", "State of Palestine", "State of Palestine"]
    assert pd.isna(names[3])
