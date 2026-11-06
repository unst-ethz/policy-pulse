"""Facet counts over the synthetic world in conftest.py."""

from app.un_data_stream.analysis import facets
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
