import pytest
import socket
import pandas as pd
import numpy as np

def is_internet_available(host="8.8.8.8", port=53, timeout=3):
    """
    Check if there is an internet connection by trying to connect to Google's public DNS.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_internet: mark test as requiring internet access"
    )

def pytest_collection_modifyitems(config, items):
    if is_internet_available():
        return

    skip_internet = pytest.mark.skip(reason="No internet connection available")
    for item in items:
        if "needs_internet" in item.keywords:
            item.add_marker(skip_internet)


@pytest.fixture(scope="module")
def random_un_votes_dataframe():
    """Generates a synthetic, pseudo-random DataFrame mimicking UN voting records."""
    np.random.seed(1946)

    n_countries = 193
    n_resolutions = 1_000

    countries = [f"Member_State_{i}" for i in range(n_countries)]
    undl_ids = [f"A/RES/78/{i}" for i in range(n_resolutions)]

    data = {
        'undl_id': undl_ids,
        'date': pd.date_range(start='2024-01-01', periods=n_resolutions),
        'title': [f"Resolution Topic {i}" for i in range(n_resolutions)]
    }

    # Fill country columns with random Y, N, A, or NaN (Missing)
    for country in countries:
        data[country] = np.random.choice(['Y', 'N', 'A', np.nan], n_resolutions, p=[0.4, 0.2, 0.3, 0.1])

    return pd.DataFrame(data)


@pytest.fixture
def data_processor(caplog):
    """
    Provides an initialized DataProcessor instance.
    The 'caplog' fixture is included to allow testing of log output.
    """
    import logging
    from app.un_data_stream.data.processor import DataProcessor

    # Using a dedicated test logger to avoid polluting main logs
    logger = logging.getLogger("un_data_test")
    config = {
        "env": "test",
        "threshold": 0.5
    }
    return DataProcessor(config, logger)
