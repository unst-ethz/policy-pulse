import pytest
import time
import pandas as pd
import numpy as np


def test_calculate_single_resolution_matrix_correctness(data_processor):
    """Verify correctness of the agreement-score computation, including NaN handling."""

    country_cols = 'Country_A Country_B Country_C Country_D Country_E Country_F'.split(' ')
    # A: Yes, B: Abstain, C: No, D: Missing
    row = pd.Series({
        'undl_id': 'RES/1',
        'Country_A': 'Y',
        'Country_B': 'A',
        'Country_C': 'N',
        'Country_D': '?',  # Invalid/Missing
        'Country_E': 'Y',
        'Country_F': 'N'
    })

    matrix = data_processor._calculate_single_resolution_matrix(row, country_cols)

    assert matrix[0, 0] == 1.0  # Self-agreement
    assert matrix[0, 4] == 1.0  # Y vs Y
    assert matrix[2, 5] == 1.0  # N vs N
    assert matrix[0, 1] == 0.5  # Y vs A
    assert matrix[0, 2] == 0.0  # Y vs N
    assert matrix[1, 2] == 0.5  # A vs N
    assert np.isnan(matrix[0, 3])  # Y vs NaN
    assert np.isnan(matrix[1, 3])  # A vs NaN
    assert np.isnan(matrix[2, 3])  # N vs NaN


def test_calculate_agreement_matrix_performance(data_processor, random_un_votes_dataframe):
    """
    Test the compute-time performance of the agreement-matrix calculation
    using the shared fixture.
    """
    start = time.perf_counter()
    matrices, countries = data_processor.calculate_agreement_matrix(random_un_votes_dataframe)
    end = time.perf_counter()

    assert len(matrices) == len(random_un_votes_dataframe)
    assert len(countries) == 193
    print(f"\nExecution time for {len(random_un_votes_dataframe)} resolutions: {end - start:.4f}s")
