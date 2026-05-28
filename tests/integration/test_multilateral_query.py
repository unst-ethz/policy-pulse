"""Integration tests for voting-rate statistics returned by query_multilateral_stats().

Scope: yes_rate, no_rate, abstention_rate, and participation_count on live data.
Multilateral alignment correctness is covered at the unit level in
test_multilateral_scores.py.

--- Arithmetic invariants ---
For any country that voted on at least one resolution:
    yes_rate + no_rate + abstention_rate == 1.0   (rates partition the votes)
    participation_count > 0

Countries with participation_count == 0 must have NaN for all rate columns.

--- Correctness ---
Rates are verified against a reference implementation (_reference_rates) that
re-derives expected values directly from raw vote strings in resolution_table,
sharing no code with the query engine.
"""

import numpy as np
import pandas as pd
import pytest

SAMPLE_COUNTRY = "USA"
SAMPLE_SIZE = 500

EXPECTED_COLUMNS = {
    "country", "multilateral_alignment", "abstention_rate",
    "yes_rate", "no_rate", "participation_count",
}


def _data_available() -> bool:
    try:
        from app import data as app_data
        return not app_data.query_engine.query_resolutions().empty
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _data_available(), reason="Local resolution data files not available"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    from app import data as app_data
    return app_data.query_engine


@pytest.fixture(scope="module")
def resolution_table(engine):
    return engine.resolution_table


@pytest.fixture(scope="module")
def sample_res_ids(resolution_table):
    return resolution_table["undl_id"].iloc[:SAMPLE_SIZE].tolist()


@pytest.fixture(scope="module")
def stats_full(engine):
    return engine.query_multilateral_stats()


@pytest.fixture(scope="module")
def stats_sample(engine, sample_res_ids):
    return engine.query_multilateral_stats(sample_res_ids)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def _reference_rates(resolution_table: pd.DataFrame, country: str, ids: list) -> dict:
    """Ground-truth voting rates derived from raw vote strings in resolution_table.

    Intentionally naive — simple enough to be obviously correct — and shares
    no code with query_multilateral_stats().
    """
    votes = (
        resolution_table.loc[resolution_table["undl_id"].isin(set(ids)), country]
        .astype(str).str.strip().str.upper()
    )
    voted = votes.isin(["Y", "N", "A"])
    n = int(voted.sum())
    if n == 0:
        return {"participation": 0, "abstention_rate": np.nan,
                "yes_rate": np.nan, "no_rate": np.nan}
    return {
        "participation":    n,
        "abstention_rate":  float((votes == "A").sum() / n),
        "yes_rate":         float((votes == "Y").sum() / n),
        "no_rate":          float((votes == "N").sum() / n),
    }


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_output_schema(stats_full):
    assert isinstance(stats_full, pd.DataFrame)
    assert EXPECTED_COLUMNS.issubset(stats_full.columns)


def test_one_row_per_country(stats_full, engine):
    assert len(stats_full) == len(engine.country_columns)
    assert stats_full["country"].nunique() == len(engine.country_columns)


# ---------------------------------------------------------------------------
# Arithmetic invariants
# ---------------------------------------------------------------------------

def test_rates_sum_to_one(stats_full):
    """yes_rate + no_rate + abstention_rate must equal 1.0 for every country that voted."""
    voted = stats_full["participation_count"] > 0
    rate_sum = (
        stats_full.loc[voted, "yes_rate"]
        + stats_full.loc[voted, "no_rate"]
        + stats_full.loc[voted, "abstention_rate"]
    )
    max_deviation = (rate_sum - 1.0).abs().max()
    assert max_deviation < 1e-5, (
        f"Rates don't sum to 1 for some countries; max deviation: {max_deviation:.2e}"
    )


def test_rates_in_range(stats_full):
    for col in ["yes_rate", "no_rate", "abstention_rate"]:
        vals = stats_full[col].dropna()
        assert (vals >= 0.0).all() and (vals <= 1.0).all(), f"{col} out of [0, 1]"


def test_participation_count_non_negative(stats_full):
    assert (stats_full["participation_count"] >= 0).all()


def test_non_voters_have_nan_rates(stats_full):
    non_voters = stats_full[stats_full["participation_count"] == 0]
    for col in ["yes_rate", "no_rate", "abstention_rate", "multilateral_alignment"]:
        assert non_voters[col].isna().all(), (
            f"Non-voters should have NaN {col}"
        )


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_participation_and_rates_match_reference(
    stats_sample, resolution_table, sample_res_ids
):
    """Rates for SAMPLE_COUNTRY must match the reference to 1e-5."""
    ref = _reference_rates(resolution_table, SAMPLE_COUNTRY, sample_res_ids)
    row = stats_sample.loc[stats_sample["country"] == SAMPLE_COUNTRY].iloc[0]

    assert int(row["participation_count"]) == ref["participation"]

    for col, ref_val in [
        ("abstention_rate", ref["abstention_rate"]),
        ("yes_rate",        ref["yes_rate"]),
        ("no_rate",         ref["no_rate"]),
    ]:
        if np.isnan(ref_val):
            assert pd.isna(row[col]), f"{col} should be NaN"
        else:
            assert abs(float(row[col]) - ref_val) < 1e-5, (
                f"{col}: engine={float(row[col]):.6f}, reference={ref_val:.6f}"
            )



def test_resolution_filter_changes_result(engine, stats_full, sample_res_ids):
    """Passing a subset of IDs must produce a strictly lower participation count."""
    stats_sub = engine.query_multilateral_stats(sample_res_ids)
    full_count = int(stats_full.loc[stats_full["country"] == SAMPLE_COUNTRY, "participation_count"].iloc[0])
    sub_count  = int(stats_sub.loc[stats_sub["country"] == SAMPLE_COUNTRY, "participation_count"].iloc[0])
    assert sub_count < full_count, (
        f"Subset participation ({sub_count}) should be < full ({full_count})"
    )
