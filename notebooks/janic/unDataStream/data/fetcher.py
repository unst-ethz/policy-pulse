"""
Data fetching orchestrator.

This module orchestrates fetching from multiple data sources using
registered fetchers for different dataset types.
"""

import logging
from typing import Dict, Any

from ..fetchers.ga_fetcher import GAResolutionFetcher
from ..fetchers.sc_fetcher import SCResolutionFetcher
from ..fetchers.thesaurus_fetcher import ThesaurusFetcher
from ..core.abstractions import DatasetFetcher


class DataFetcher:
    """Orchestrates fetching from multiple data sources."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Registry of dataset fetchers
        self._dataset_fetchers: Dict[str, DatasetFetcher] = {}
        self._register_default_fetchers()
        
        # Thesaurus fetcher (separate from datasets)
        self.thesaurus_fetcher = ThesaurusFetcher(logger)
    
    def _register_default_fetchers(self):
        """Register default dataset fetchers."""
        ga_fetcher = GAResolutionFetcher(self.logger)
        self._dataset_fetchers[ga_fetcher.get_dataset_type()] = ga_fetcher
        
        # Future fetchers can be added here:
        sc_fetcher = SCResolutionFetcher(self.logger)
        self._dataset_fetchers[sc_fetcher.get_dataset_type()] = sc_fetcher
    
    def register_fetcher(self, fetcher: DatasetFetcher):
        """Register a new dataset fetcher."""
        self._dataset_fetchers[fetcher.get_dataset_type()] = fetcher
    
    def fetch_resolutions(self) -> Dict[str, Any]:
        """Fetch all configured resolution datasets."""
        results = {}
        
        for dataset_type, source_config in self.config['data_sources']['resolutions'].items():
                
            if dataset_type in self._dataset_fetchers:
                fetcher = self._dataset_fetchers[dataset_type]
                results[dataset_type] = fetcher.fetch(source_config)
            else:
                self.logger.warning(f"No fetcher registered for dataset type: {dataset_type}")
        
        return results
    
    def fetch_thesaurus(self):
        """Fetch thesaurus data."""
        thesaurus_config = self.config['data_sources'].get('thesaurus')
        if not thesaurus_config:
            raise ValueError("No thesaurus configuration found")
        
        return self.thesaurus_fetcher.fetch(thesaurus_config)