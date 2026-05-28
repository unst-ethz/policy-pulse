"""Integration tests for consensus scores stored in resolution_table.

Scope: the consensus_score column on live data.
Computation correctness is covered at the unit level in test_resolution_matrix.py.

--- Correctness ---
A reference implementation (_reference_consensus_score) re-derives expected
values directly from raw vote strings in resolution_table, sharing no code
with DataProcessor._calculate_single_consensus_score.  Tests assert that the
stored column matches the reference for a sample of resolutions.
"""

import warnings
import numpy as np
import pandas as pd
import pytest

SAMPLE_SIZE = 200


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
def country_columns(engine):
    return engine.country_columns


@pytest.fixture(scope="module")
def sample_rows(resolution_table):
    return resolution_table.head(SAMPLE_SIZE)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def _reference_consensus_score(row: pd.Series, country_columns: list) -> float:
    """Ground-truth consensus score derived from raw vote strings.

    Intentionally naive — simple enough to be obviously correct — and shares
    no code with DataProcessor._calculate_single_consensus_score.
    """
    vote_map = {"Y": 1.0, "A": 0.0, "N": -1.0}
    votes = np.array(
        [vote_map.get(str(row[c]).strip().upper(), np.nan) for c in country_columns],
        dtype=np.float64,
    )
    diff = np.abs(votes[:, np.newaxis] - votes[np.newaxis, :])
    agree = 1.0 - diff / 2.0
    tril_idx = np.tril_indices(len(votes), k=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(agree[tril_idx]))


# ---------------------------------------------------------------------------
# Schema / range tests
# ---------------------------------------------------------------------------

def test_consensus_score_column_present(resolution_table):
    assert "consensus_score" in resolution_table.columns


def test_consensus_score_in_range(resolution_table):
    """All non-NaN stored scores must be in [0, 1]."""
    col = resolution_table["consensus_score"].dropna()
    assert (col >= 0.0).all() and (col <= 1.0).all()


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------

def test_consensus_scores_match_reference(sample_rows, country_columns):
    """Stored consensus scores must match the reference implementation to 1e-4."""
    for _, row in sample_rows.iterrows():
        expected = _reference_consensus_score(row, country_columns)
        actual = row["consensus_score"]

        if np.isnan(expected):
            assert pd.isna(actual), (
                f"Resolution {row['undl_id']}: expected NaN, got {actual}"
            )
        else:
            assert abs(float(actual) - expected) < 1e-4, (
                f"Resolution {row['undl_id']}: "
                f"expected {expected:.6f}, got {float(actual):.6f}"
            )
