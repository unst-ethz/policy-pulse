import pytest
import time
import pandas as pd
import numpy as np

# Define test cases for parameterization
test_cases = [
    (
        "standard_with_single_nan",
        {'A': 'Y', 'B': 'A', 'C': 'N', 'D': '?', 'E': 'Y', 'F': 'N'},
        [
            (0, 0, 1.0),  # Self-agreement
            (0, 4, 1.0),  # Y vs Y
            (2, 5, 1.0),  # N vs N
            (0, 1, 0.5),  # Y vs A
            (0, 2, 0.0),  # Y vs N
            (1, 2, 0.5),  # A vs N
            (0, 3, np.nan),  # Y vs NaN
        ],
        0.4
    ),
    (
        "mixed_with_dual_nan",
        {'A': 'Y', 'B': 'N', 'C': '?', 'D': '?'},
        [
            (0, 1, 0.0),  # Y vs N
            (0, 2, np.nan),  # Y vs NaN
            (1, 3, np.nan),  # N vs NaN
            (2, 3, np.nan),  # NaN vs NaN
        ],
        0.0  # Only one valid pair (Y, N) with score 0.0
    ),
    (
        "all_agree_yes",
        {'A': 'Y', 'B': 'Y', 'C': 'Y'},
        [
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
        ],
        1.0
    ),
    (
        "all_agree_no",
        {'A': 'N', 'B': 'N', 'C': 'N'},
        [
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
        ],
        1.0
    ),
    (
        "polarized_vote",
        {'A': 'Y', 'B': 'Y', 'C': 'N', 'D': 'N'},
        [
            (0, 1, 1.0),  # Y vs Y
            (2, 3, 1.0),  # N vs N
            (0, 2, 0.0),  # Y vs N
        ],
        1/3
    ),
    (
        "single_voter",
        {'A': 'Y', 'B': '?', 'C': '?'},
        [
            (0, 1, np.nan),
            (1, 2, np.nan),
        ],
        np.nan  # No valid pairs
    ),
    (
        "no_voters",
        {'A': '?', 'B': '?', 'C': '?'},
        [
            (0, 1, np.nan),
            (1, 2, np.nan),
        ],
        np.nan  # No valid pairs
    ),
]


@pytest.mark.parametrize(
    "test_name, votes, matrix_checks, expected_c_score",
    test_cases,
    ids=[case[0] for case in test_cases]
)
def test_agreement_and_consensus_score(
        data_processor, test_name, votes, matrix_checks, expected_c_score
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


def test_calculate_agreement_matrix_performance(data_processor, random_un_votes_dataframe):
    """
    Test the compute-time performance of the agreement-matrix calculation
    using the shared fixture.
    """
    start = time.perf_counter()
    # Note: The function now returns three values, but we only need the first two for this test
    matrices, c_scores, countries = data_processor.calculate_agreement_data(random_un_votes_dataframe)
    end = time.perf_counter()

    n_res = len(random_un_votes_dataframe)
    assert len(matrices) == n_res
    assert len(c_scores) == n_res
    assert len(countries) == 193
    print(f"\nExecution time for {n_res} resolutions: {end - start:.4f}s")
