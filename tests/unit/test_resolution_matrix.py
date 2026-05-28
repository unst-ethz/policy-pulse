"""Unit tests for per-resolution agreement matrix and consensus score computation.

--- Agreement matrix ---
For a single resolution, the (C × C) agreement matrix is computed via NumPy
broadcasting:  score(i, j) = 1 - |v_i - v_j| / 2  ∈ {0.0, 0.5, 1.0, nan}.
NaN propagates whenever either country did not vote.

--- Consensus score ---
The consensus score is the nanmean of the lower triangle of the agreement
matrix (k=-1 to exclude the diagonal). Taking the lower triangle gives each
unique country pair exactly once — the matrix is symmetric so the upper
triangle would produce identical values. The score is NaN when fewer than
two countries voted (all lower-triangle cells are NaN).

Tests use hard-coded expected values derived by hand for each scenario.
"""

import numpy as np
import pandas as pd
import pytest

# Each entry: (name, votes, matrix_checks, expected_c_score)
# matrix_checks: list of (row, col, expected_value) spot-checks
test_cases = [
    (
        "standard_with_single_nan",
        {"A": "Y", "B": "A", "C": "N", "D": "?", "E": "Y", "F": "N"},
        [
            (0, 0, 1.0),     # self-agreement
            (0, 4, 1.0),     # Y vs Y
            (2, 5, 1.0),     # N vs N
            (0, 1, 0.5),     # Y vs A
            (0, 2, 0.0),     # Y vs N
            (1, 2, 0.5),     # A vs N
            (0, 3, np.nan),  # Y vs NaN
        ],
        0.4,
    ),
    (
        "mixed_with_dual_nan",
        {"A": "Y", "B": "N", "C": "?", "D": "?"},
        [
            (0, 1, 0.0),     # Y vs N
            (0, 2, np.nan),  # Y vs NaN
            (1, 3, np.nan),  # N vs NaN
            (2, 3, np.nan),  # NaN vs NaN
        ],
        0.0,
    ),
    (
        "all_agree_yes",
        {"A": "Y", "B": "Y", "C": "Y"},
        [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)],
        1.0,
    ),
    (
        "all_agree_no",
        {"A": "N", "B": "N", "C": "N"},
        [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)],
        1.0,
    ),
    (
        "polarized_vote",
        {"A": "Y", "B": "Y", "C": "N", "D": "N"},
        [
            (0, 1, 1.0),  # Y vs Y
            (2, 3, 1.0),  # N vs N
            (0, 2, 0.0),  # Y vs N
        ],
        1 / 3,
    ),
    (
        "single_voter",
        {"A": "Y", "B": "?", "C": "?"},
        [(0, 1, np.nan), (1, 2, np.nan)],
        np.nan,
    ),
    (
        "no_voters",
        {"A": "?", "B": "?", "C": "?"},
        [(0, 1, np.nan), (1, 2, np.nan)],
        np.nan,
    ),
]


@pytest.mark.parametrize(
    "test_name, votes, matrix_checks, expected_c_score",
    test_cases,
    ids=[c[0] for c in test_cases],
)
def test_agreement_matrix_and_consensus_score(
    data_processor, test_name, votes, matrix_checks, expected_c_score
):
    """Agreement matrix spot-checks and consensus score must match expected values."""
    country_cols = list(votes.keys())
    row = pd.Series(votes, name="undl_id")

    matrix = data_processor._calculate_single_resolution_matrix(row, country_cols)

    for i, j, expected in matrix_checks:
        if np.isnan(expected):
            assert np.isnan(matrix[i, j]), f"Matrix cell ({i},{j}) should be NaN"
        else:
            assert matrix[i, j] == pytest.approx(expected), (
                f"Matrix cell ({i},{j}) should be {expected}"
            )

    c_score = data_processor._calculate_single_consensus_score(matrix)

    if np.isnan(expected_c_score):
        assert np.isnan(c_score), "Consensus score should be NaN"
    else:
        assert c_score == pytest.approx(expected_c_score), (
            "Consensus score does not match expected value"
        )
