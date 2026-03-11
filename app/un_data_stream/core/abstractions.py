"""
Core abstractions for the UN Data Stream pipeline.

This module defines the abstract base classes that all dataset-specific
fetchers and processors must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import pandas as pd
from rdflib import Graph
import logging

class DatasetFetcher(ABC):
    """
    Abstract base class for fetchers that retrieve data from the web.
    Implements the common URL resolution and fetching template.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def resolve_url(self, source_config: Dict[str, Any]) -> str:
        """
        Resolves the download URL from the source configuration.
        Handles both the dynamic and static URl case.
        """
        from .web_utils import fetch_latest_file_url_from_api  # Local import to avoid circular dependency
        
        if source_config.get('source_type') == 'dynamic_api_file':
            recid = source_config['recid']  # The UNDL's record ID for which we will query the API endpoint
            pattern = source_config['file_name_pattern'].strip() # Ensure no leading/trailing whitespace from YAML multiline
            file_format = source_config['format']
            
            self.logger.info(
                f"Looking for latest file for record {recid} with name "
                f"pattern '{pattern}' and format '{file_format}'"
            )
            
            url = fetch_latest_file_url_from_api(recid, pattern, file_format)
            
            if not url:
                error_msg = (f"Could not find a file matching name pattern '{pattern}' "
                             f"and format '{file_format}' for record {recid}. "
                             f"Verify the record ID exists and contains the expected file.")
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            return url
            
        else:
            # STATIC case (currently not used)
            url = source_config.get('url')
            if not url:
                error_msg = ("Source configuration missing 'url' (for static) or "
                             "'recid'/'file_name_pattern'/'format' (for dynamic_api_file)")
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            return url

    def fetch(self, source_config: Dict[str, Any]) -> Union[pd.DataFrame, Graph]:
        """
        Template method: resolves URL, logs, and delegates specific parsing.
        """
        url = self.resolve_url(source_config)
        self.logger.info(f"Fetching data from {url}")
        
        try:
            return self._fetch_and_parse(url)
        except Exception as e:
            self.logger.error(f"Failed to fetch/parse data from {url}: {e}")
            raise

    @abstractmethod
    def _fetch_and_parse(self, url: str) -> Union[pd.DataFrame, Graph]:
        """
        Subclasses must implement the specific logic to download and parse the file at 'url'.
        """
        pass

    @abstractmethod
    def get_dataset_type(self) -> str:
        """Return the dataset type identifier (e.g., 'ga_resolutions', 'sc_resolutions')."""
        pass


class DatasetProcessor(ABC):
    """Abstract base class for dataset-specific processors."""
    
    @abstractmethod
    def process(self, raw_data: Union[pd.DataFrame, Graph], **kwargs) -> Dict[str, pd.DataFrame]:
        """Process raw data into normalized tables."""
        pass
    
    @abstractmethod 
    def get_dataset_type(self) -> str:
        """Return the dataset type this processor handles."""
        pass
