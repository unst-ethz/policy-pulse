"""
Security Council resolution data fetcher.

This module handles fetching raw SC resolution data from external sources.
"""

import logging
import pandas as pd
from typing import Dict, Any

from ..core.abstractions import DatasetFetcher


class SCResolutionFetcher(DatasetFetcher):
    """Fetches Security Council resolution data"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def fetch(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Fetch SC resolution data from URL."""
        self.logger.info(f"Fetching SC resolutions from {source_config['url']}")
        try:
            df = pd.read_csv(source_config['url'])
            self.logger.info(f"Successfully fetched {len(df)} SC resolution records")
            return df
        except Exception as e:
            self.logger.error(f"Failed to fetch SC resolutions: {e}")
            raise
        
    def get_dataset_type(self) -> str:
        return "sc_resolutions"