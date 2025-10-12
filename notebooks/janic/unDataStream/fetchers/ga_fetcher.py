"""
General Assembly resolution data fetcher.

This module handles fetching raw GA resolution data from external sources.
"""

import logging
import pandas as pd
from typing import Dict, Any

from ..core.abstractions import DatasetFetcher


class GAResolutionFetcher(DatasetFetcher):
    """Fetches General Assembly resolution data."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def fetch(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Fetch GA resolution data from URL."""
        self.logger.info(f"Fetching GA resolutions from {source_config['url']}")
        try:
            df = pd.read_csv(source_config['url'])
            self.logger.info(f"Successfully fetched {len(df)} GA resolution records")
            
            df['session'] = df['session'].astype(str) # Ensure session is consistent
            df['date'] = pd.to_datetime(df['date']) # Convert date to datetime
            
            return df
        except Exception as e:
            self.logger.error(f"Failed to fetch GA resolutions: {e}")
            raise
    
        
    def get_dataset_type(self) -> str:
        return "ga_resolutions"