"""
Thesaurus data fetcher.

This module handles fetching and parsing thesaurus data in RDF/TTL format.
"""

import logging
import requests
from io import BytesIO
from typing import Dict, Any
from rdflib import Graph


class ThesaurusFetcher:
    """Fetches thesaurus data (RDF/TTL format)."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def fetch(self, source_config: Dict[str, Any]) -> Graph:
        """Fetch and parse thesaurus graph."""
        self.logger.info(f"Fetching thesaurus from {source_config['url']}")

        thesaurus_url = source_config['url']

        graph = Graph()

        try:
            response = requests.get(thesaurus_url)
        except Exception as e:
            self.logger.info("Error fetching thesaurus file. The dataset might has been updated. Check the date in the URL.")
            self.logger.info("Thesaurus URL:", thesaurus_url)
            self.logger.info(f"Error: {e}")
            
            raise ValueError("Failed to fetch thesaurus")

        if response.status_code != 200:
            self.logger.info("Error fetching thesaurus file. The dataset might has been updated. Check the date in the URL. Response code:", response.status_code)
            self.logger.info("Thesaurus URL:", thesaurus_url)

            raise ValueError("Failed to fetch thesaurus")
        
        ttl_content = BytesIO(response.content)
        self.logger.info(f"Downloaded thesaurus file successfully")

        graph.parse(ttl_content, format="turtle")

        return graph