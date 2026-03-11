import pytest
import logging
import pandas as pd
from rdflib import Graph
from app.un_data_stream.fetchers.ga_fetcher import GAResolutionFetcher
from app.un_data_stream.fetchers.sc_fetcher import SCResolutionFetcher
from app.un_data_stream.fetchers.thesaurus_fetcher import ThesaurusFetcher

# Configure a basic logger for the tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestLogger")

@pytest.mark.needs_internet
def test_ga_fetcher_e2e():
    """End-to-end test for GA Resolution Fetcher."""
    fetcher = GAResolutionFetcher(logger)
    
    # Configuration mimicking data_sources.yaml
    config = {
        'source_type': 'dynamic_api_file',
        'recid': '4060887',
        'file_name_pattern': '.*_ga_voting$',
        'format': '.csv'
    }
    
    df = fetcher.fetch(config)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Basic check for *some* expected columns
    expected = set('ms_code ms_vote date session resolution meeting title total_yes total_no'.split())
    assert expected.issubset(set(df.columns))
    logger.info(f"Fetched {len(df)} GA resolution records")

@pytest.mark.needs_internet
def test_sc_fetcher_e2e():
    """End-to-end test for SC Resolution Fetcher."""
    fetcher = SCResolutionFetcher(logger)
    
    # Configuration mimicking data_sources.yaml
    config = {
        'source_type': 'dynamic_api_file',
        'recid': '4055387',
        'file_name_pattern': '.*_sc_voting$',
        'format': '.csv'
    }
    
    df = fetcher.fetch(config)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Basic check for *some* expected columns
    expected = set('ms_code permanent_member ms_vote date resolution meeting total_yes total_no'.split())
    assert expected.issubset(set(df.columns))
    logger.info(f"Fetched {len(df)} SC resolution records")

@pytest.mark.needs_internet
def test_thesaurus_fetcher_e2e():
    """End-to-end test for Thesaurus Fetcher."""
    fetcher = ThesaurusFetcher(logger)
    
    # Configuration mimicking data_sources.yaml
    config = {
        'source_type': 'dynamic_api_file',
        'recid': '4075456',
        'file_name_pattern': 'unbist-.*_2$',
        'format': '.ttl'
    }
    
    graph = fetcher.fetch(config)
    
    assert isinstance(graph, Graph)
    assert len(graph) > 0

    # TODO: Add a few more relevant data checks
    logger.info(f"Fetched Thesaurus graph with {len(graph)} triples")
