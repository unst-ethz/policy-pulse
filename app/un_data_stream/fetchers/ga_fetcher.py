"""
General Assembly resolution data fetcher.

This module handles fetching raw GA resolution data from external sources.
"""

import pandas as pd

from ..core.abstractions import DatasetFetcher


class GAResolutionFetcher(DatasetFetcher):
    """Fetches General Assembly resolution data."""
    
    def _fetch_and_parse(self, url: str) -> pd.DataFrame:
        """Fetch GA resolution data from URL."""
        
        df = pd.read_csv(url)
        self.logger.info(f"Successfully fetched {len(df)} GA resolution records")
        
        df['session'] = df['session'].astype(str) # Ensure session is consistent
        df['date'] = pd.to_datetime(df['date']) # Convert date to datetime
        
        return df

    def get_dataset_type(self) -> str:
        return "ga_resolutions"