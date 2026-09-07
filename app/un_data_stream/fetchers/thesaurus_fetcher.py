"""
Thesaurus data fetcher.

This module handles fetching and parsing thesaurus data in RDF/TTL format.
"""

from io import BytesIO

from rdflib import Graph

from ..core.abstractions import DatasetFetcher
from ..core.web_utils import download_bytes


class ThesaurusFetcher(DatasetFetcher):
    """Fetches thesaurus data (RDF/TTL format)."""
    
    def _fetch_and_parse(self, url: str) -> Graph:
        """Fetch and parse thesaurus graph."""
        
        graph = Graph()
        
        ttl_content = BytesIO(download_bytes(url))
        self.logger.info(f"Downloaded thesaurus file successfully")

        graph.parse(ttl_content, format="turtle")
        
        return graph

    def get_dataset_type(self) -> str:
        return "thesaurus"
