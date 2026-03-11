"""
Thesaurus data fetcher.

This module handles fetching and parsing thesaurus data in RDF/TTL format.
"""

from io import BytesIO

import requests
from rdflib import Graph

from ..core.abstractions import DatasetFetcher


class ThesaurusFetcher(DatasetFetcher):
    """Fetches thesaurus data (RDF/TTL format)."""
    
    def _fetch_and_parse(self, url: str) -> Graph:
        """Fetch and parse thesaurus graph."""
        
        graph = Graph()
        
        response = requests.get(url)
        response.raise_for_status()

        ttl_content = BytesIO(response.content)
        self.logger.info(f"Downloaded thesaurus file successfully")

        graph.parse(ttl_content, format="turtle")
        
        return graph

    def get_dataset_type(self) -> str:
        return "thesaurus"