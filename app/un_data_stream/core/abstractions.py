"""
Core abstractions for the UN Data Stream pipeline.

This module defines the abstract base classes that all dataset-specific
fetchers and processors must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class DatasetFetcher(ABC):
    """Abstract base class for dataset-specific fetchers."""
    
    @abstractmethod
    def fetch(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Fetch raw data for this dataset type."""
        pass
    
    @abstractmethod
    def get_dataset_type(self) -> str:
        """Return the dataset type identifier (e.g., 'ga_resolutions', 'sc_resolutions')."""
        pass


class DatasetProcessor(ABC):
    """Abstract base class for dataset-specific processors."""
    
    @abstractmethod
    def process(self, raw_data: pd.DataFrame, **kwargs) -> Dict[str, pd.DataFrame]:
        """Process raw data into normalized tables."""
        pass
    
    @abstractmethod 
    def get_dataset_type(self) -> str:
        """Return the dataset type this processor handles."""
        pass