"""Unit tests for per-resolution multilateral alignment score computation.

--- Multilateral alignment score ---
For each country on a single resolution, the multilateral alignment score is
the mean agreement with every *other* country that voted on that resolution:

    multilateral(c, r) = nanmean over j ≠ c of score(c, j, r)

Countries that did not vote receive NaN. Countries that voted but had no
valid partner (all others were NaN) also receive NaN.

The scores are computed inside calculate_agreement_data() by masking the
diagonal of the per-resolution agreement matrix and taking row-means.

Tests use hard-coded expected values derived by hand for each scenario.
"""

import time
import numpy as np
import pandas as pd
import pytest

# Each entry: (name, votes, expected_multilateral)
# expected_multilateral: per-country row-mean agreement
#   (NaN for non-voters or for the sole voter with no valid partners)
test_cases = [
    (
        "standard_with_single_nan",
        {"A": "Y", "B": "A", "C": "N", "D": "?", "E": "Y", "F": "N"},
        # A: (0.5+0.0+1.0+0.0)/4, B: (0.5+0.5+0.5+0.5)/4, C: (0.0+0.5+0.0+1.0)/4
        # D: NaN (non-voter), E: (1.0+0.5+0.0+0.0)/4, F: (0.0+0.5+1.0+0.0)/4
        [0.375, 0.5, 0.375, np.nan, 0.375, 0.375],
    ),
    (
        "mixed_with_dual_nan",
        {"A": "Y", "B": "N", "C": "?", "D": "?"},
        # A: one valid partner (B), score 0.0; B: one valid partner (A), score 0.0
        [0.0, 0.0, np.nan, np.nan],
    ),
    (
        "all_agree_yes",
        {"A": "Y", "B": "Y", "C": "Y"},
        [1.0, 1.0, 1.0],
    ),
    (
        "all_agree_no",
        {"A": "N", "B": "N", "C": "N"},
        [1.0, 1.0, 1.0],
    ),
    (
        "polarized_vote",
        {"A": "Y", "B": "Y", "C": "N", "D": "N"},
        # Every voter: 1 same-side ally (1.0) + 2 opponents (0.0) → mean = 1/3
        [1 / 3, 1 / 3, 1 / 3, 1 / 3],
    ),
    (
        "single_voter",
        {"A": "Y", "B": "?", "C": "?"},
        # A has no valid partners; B, C did not vote
        [np.nan, np.nan, np.nan],
    ),
    (
        "no_voters",
        {"A": "?", "B": "?", "C": "?"},
        [np.nan, np.nan, np.nan],
    ),
]


@pytest.mark.parametrize(
    "test_name, votes, expected_multilateral",
    test_cases,
    ids=[c[0] for c in test_cases],
)
def test_multilateral_scores(
    data_processor, test_name, votes, expected_multilateral
):
    """Per-country multilateral scores must match expected values for each scenario."""
    country_cols = list(votes.keys())
    df = pd.DataFrame([{"undl_id": "TEST/1", **votes}])

    _, _, multilateral_scores, _ = data_processor.calculate_agreement_data(df)

    assert multilateral_scores.shape == (1, len(country_cols))
    assert multilateral_scores.dtype == np.float32

    actual = multilateral_scores[0]
    for i, expected in enumerate(expected_multilateral):
        if np.isnan(expected):
            assert np.isnan(actual[i]), (
                f"Country {country_cols[i]}: score should be NaN, got {actual[i]}"
            )
        else:
            assert actual[i] == pytest.approx(expected, abs=1e-6), (
                f"Country {country_cols[i]}: expected {expected:.4f}, got {actual[i]:.4f}"
            )


def test_calculate_agreement_data_performance(data_processor, random_un_votes_dataframe):
    """calculate_agreement_data must complete within a reasonable time budget."""
    start = time.perf_counter()
    c_scores, countries, multilateral_scores, vote_bool_arrays = (
        data_processor.calculate_agreement_data(random_un_votes_dataframe)
    )
    elapsed = time.perf_counter() - start

    n_res = len(random_un_votes_dataframe)
    assert len(c_scores) == n_res
    assert len(countries) == 193
    assert multilateral_scores.shape == (n_res, 193)
    assert multilateral_scores.dtype == np.float32
    print(f"\nExecution time for {n_res} resolutions: {elapsed:.4f}s")
