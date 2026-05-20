import pytest
import time
import pandas as pd
import numpy as np

# Define test cases for parameterization.
# Columns: name, votes, matrix_checks, expected_c_score, expected_multilateral
# expected_multilateral: per-country row-mean agreement (NaN for non-voters or isolated voters)
test_cases = [
    (
        "standard_with_single_nan",
        {'A': 'Y', 'B': 'A', 'C': 'N', 'D': '?', 'E': 'Y', 'F': 'N'},
        [
            (0, 0, 1.0),    # Self-agreement
            (0, 4, 1.0),    # Y vs Y
            (2, 5, 1.0),    # N vs N
            (0, 1, 0.5),    # Y vs A
            (0, 2, 0.0),    # Y vs N
            (1, 2, 0.5),    # A vs N
            (0, 3, np.nan), # Y vs NaN
        ],
        0.4,
        # A: (0.5+0.0+1.0+0.0)/4, B: (0.5+0.5+0.5+0.5)/4, C: (0.0+0.5+0.0+1.0)/4,
        # D: NaN (non-voter), E: (1.0+0.5+0.0+0.0)/4, F: (0.0+0.5+1.0+0.0)/4
        [0.375, 0.5, 0.375, np.nan, 0.375, 0.375],
    ),
    (
        "mixed_with_dual_nan",
        {'A': 'Y', 'B': 'N', 'C': '?', 'D': '?'},
        [
            (0, 1, 0.0),    # Y vs N
            (0, 2, np.nan), # Y vs NaN
            (1, 3, np.nan), # N vs NaN
            (2, 3, np.nan), # NaN vs NaN
        ],
        0.0,
        # A: one valid pair (B), score 0.0; B: one valid pair (A), score 0.0; C, D: NaN
        [0.0, 0.0, np.nan, np.nan],
    ),
    (
        "all_agree_yes",
        {'A': 'Y', 'B': 'Y', 'C': 'Y'},
        [
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
        ],
        1.0,
        [1.0, 1.0, 1.0],
    ),
    (
        "all_agree_no",
        {'A': 'N', 'B': 'N', 'C': 'N'},
        [
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
        ],
        1.0,
        [1.0, 1.0, 1.0],
    ),
    (
        "polarized_vote",
        {'A': 'Y', 'B': 'Y', 'C': 'N', 'D': 'N'},
        [
            (0, 1, 1.0), # Y vs Y
            (2, 3, 1.0), # N vs N
            (0, 2, 0.0), # Y vs N
        ],
        1/3,
        # Every voter faces 1 same-side ally (1.0) and 2 opponents (0.0) → mean = 1/3
        [1/3, 1/3, 1/3, 1/3],
    ),
    (
        "single_voter",
        {'A': 'Y', 'B': '?', 'C': '?'},
        [
            (0, 1, np.nan),
            (1, 2, np.nan),
        ],
        np.nan,
        # A has no valid pairs (everyone else is NaN); B, C are non-voters
        [np.nan, np.nan, np.nan],
    ),
    (
        "no_voters",
        {'A': '?', 'B': '?', 'C': '?'},
        [
            (0, 1, np.nan),
            (1, 2, np.nan),
        ],
        np.nan,
        [np.nan, np.nan, np.nan],
    ),
]


@pytest.mark.parametrize(
    "test_name, votes, matrix_checks, expected_c_score, expected_multilateral",
    test_cases,
    ids=[case[0] for case in test_cases]
)
def test_agreement_and_consensus_score(
        data_processor, test_name, votes, matrix_checks, expected_c_score, expected_multilateral
):
    """
    Verify correctness of the agreement matrix and the consensus score (C-score)
    computation across various scenarios, including NaN handling.
    """
    country_cols = list(votes.keys())
    row = pd.Series(votes, name='undl_id')

    # 1. Test the agreement matrix calculation
    matrix = data_processor._calculate_single_resolution_matrix(row, country_cols)

    for i, j, expected in matrix_checks:
        if np.isnan(expected):
            assert np.isnan(matrix[i, j]), f"Matrix cell ({i},{j}) should be NaN"
        else:
            assert matrix[i, j] == pytest.approx(expected), f"Matrix cell ({i},{j}) should be {expected}"

    # 2. Test the consensus score calculation
    c_score = data_processor._calculate_single_consensus_score(matrix)

    if np.isnan(expected_c_score):
        assert np.isnan(c_score), "C-score should be NaN"
    else:
        assert c_score == pytest.approx(expected_c_score), "C-score does not match expected value"


@pytest.mark.parametrize(
    "test_name, votes, matrix_checks, expected_c_score, expected_multilateral",
    test_cases,
    ids=[case[0] for case in test_cases]
)
def test_multilateral_scores(
        data_processor, test_name, votes, matrix_checks, expected_c_score, expected_multilateral
):
    """
    Verify that calculate_agreement_data produces correct per-country multilateral scores
    (row-mean off-diagonal agreement) for each vote scenario, including NaN propagation
    for non-voters and isolated voters.
    """
    country_cols = list(votes.keys())
    df = pd.DataFrame([{'undl_id': 'TEST/1', **votes}])

    _, _, _, multilateral_scores = data_processor.calculate_agreement_data(df)

    assert multilateral_scores.shape == (1, len(country_cols))
    assert multilateral_scores.dtype == np.float32

    actual = multilateral_scores[0]
    for i, expected in enumerate(expected_multilateral):
        if np.isnan(expected):
            assert np.isnan(actual[i]), \
                f"Country {country_cols[i]}: multilateral score should be NaN, got {actual[i]}"
        else:
            assert actual[i] == pytest.approx(expected, abs=1e-6), \
                f"Country {country_cols[i]}: expected {expected:.4f}, got {actual[i]:.4f}"


def test_calculate_agreement_matrix_performance(data_processor, random_un_votes_dataframe):
    """
    Test the compute-time performance of the agreement-matrix calculation
    using the shared fixture.
    """
    start = time.perf_counter()
    matrices, c_scores, countries, multilateral_scores = data_processor.calculate_agreement_data(
        random_un_votes_dataframe
    )
    end = time.perf_counter()

    n_res = len(random_un_votes_dataframe)
    assert len(matrices) == n_res
    assert len(c_scores) == n_res
    assert len(countries) == 193
    assert multilateral_scores.shape == (n_res, 193)
    assert multilateral_scores.dtype == np.float32
    print(f"\nExecution time for {n_res} resolutions: {end - start:.4f}s")
