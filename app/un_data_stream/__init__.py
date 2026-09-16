"""
UN Data Stream - the app's data layer.

Loads the normalized UN resolution tables out of Postgres (written by the `undl-ingest` jobs),
precomputes the agreement/alignment arrays, and answers queries over them in memory.

This package used to fetch from the UN Digital Library and process raw exports itself; that work
now lives in the `undl-ingest` repo, and the app is a read-only consumer of the storage layer.
"""

from .analysis.query_engine import ResolutionQueryEngine
from .data.processor import DataProcessor
from .data.repository import DataRepository

__version__ = "2.0.0"
__author__ = "UN-ETH Project Team"

__all__ = [
    'DataRepository',
    'DataProcessor',
    'ResolutionQueryEngine',
]
