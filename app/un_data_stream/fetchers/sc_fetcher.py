"""
Security Council resolution data fetcher.

This module handles fetching raw SC resolution data from external sources.
"""

import pandas as pd

from ..core.abstractions import DatasetFetcher


class SCResolutionFetcher(DatasetFetcher):
    """Fetches Security Council resolution data"""

    def _fetch_and_parse(self, url: str) -> pd.DataFrame:
        """Fetch SC resolution data from URL."""
        
        df = pd.read_csv(url)
        self.logger.info(f"Successfully fetched {len(df)} SC resolution records")
        return df
        
    def get_dataset_type(self) -> str:
        return "sc_resolutions"