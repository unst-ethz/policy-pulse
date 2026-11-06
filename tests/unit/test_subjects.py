"""Thesaurus domains over the synthetic world in conftest.py."""

from app.un_data_stream.analysis import filtering, sponsorship, subjects
from app.un_data_stream.analysis.filtering import ResolutionFilter

from .conftest import ENVIRONMENT, PEACE


def select(engine, **criteria):
    return filtering.filter_resolutions(engine, ResolutionFilter(**criteria))


def test_domain_years_count_descendants_once(engine):
    # PEACE holds 103 and 105 directly and 101/104 through DISARMAMENT; 105 had no vote
    out = subjects.domain_years(select(engine), engine.snapshot, [PEACE, ENVIRONMENT])
    assert out.sort_values(["subject_id", "year"]).to_dict("records") == [
        {"subject_id": ENVIRONMENT, "year": 2000, "resolutions": 1, "recorded": 1},
        {"subject_id": PEACE, "year": 2000, "resolutions": 1, "recorded": 1},
        {"subject_id": PEACE, "year": 2001, "resolutions": 2, "recorded": 2},
        {"subject_id": PEACE, "year": 2002, "resolutions": 1, "recorded": 0},
    ]


def test_domain_years_follow_the_period(engine):
    out = subjects.domain_years(
        select(engine, start_date="2001-01-01"), engine.snapshot, [PEACE, ENVIRONMENT]
    )
    assert set(out["subject_id"]) == {PEACE}
    assert out["resolutions"].sum() == 3


def test_subject_sponsors(engine):
    record = sponsorship.subject_sponsors(
        engine.snapshot, select(engine, subject_ids=(PEACE,)), select(engine)
    )
    # 101, 103, 104 have sponsors; 105 none. States only: the GA President is left out.
    assert (record.resolutions, record.with_data) == (4, 3)
    assert record.sponsors.to_dict("records") == [
        {"country": "AAA", "sponsored": 3, "original": 2, "own_total": 3},
        {"country": "BBB", "sponsored": 1, "original": 0, "own_total": 1},
    ]


def test_subject_sponsors_of_an_empty_selection(engine):
    record = sponsorship.subject_sponsors(
        engine.snapshot, select(engine, start_date="2030-01-01"), select(engine)
    )
    assert (record.resolutions, record.with_data, len(record.sponsors)) == (0, 0, 0)
