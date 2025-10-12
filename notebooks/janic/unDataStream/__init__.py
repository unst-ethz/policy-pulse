"""
UN Data Stream - Modular data processing pipeline for UN resolution data.

This package provides a flexible, extensible architecture for fetching, processing,
and analyzing UN resolution data from multiple sources (GA, SC, HRC, etc.).
"""

# Core abstractions
from .core.abstractions import DatasetFetcher, DatasetProcessor

# Fetchers
from .fetchers.ga_fetcher import GAResolutionFetcher
from .fetchers.sc_fetcher import SCResolutionFetcher
from .fetchers.thesaurus_fetcher import ThesaurusFetcher

# Processors
from .processors.ga_processor import GAResolutionProcessor
from .processors.sc_processor import SCResolutionProcessor
from .processors.thesaurus_processor import ThesaurusProcessor

# Data orchestration
from .data.fetcher import DataFetcher
from .data.processor import DataProcessor
from .data.merger import DataMerger
from .data.repository import DataRepository

# Analysis tools
from .analysis.query_engine import ResolutionQueryEngine
from .analysis.analyzer import ResolutionAnalyzer

__version__ = "1.0.0"
__author__ = "UN-ETH Project Team"

# Public API - main classes that users will interact with
__all__ = [
    # Main entry points
    'DataRepository',
    'DataFetcher', 
    'DataProcessor',
    'DataMerger',
    
    # Analysis tools
    'ResolutionQueryEngine',
    'ResolutionAnalyzer',
    
    # Abstract base classes (for extending)
    'DatasetFetcher',
    'DatasetProcessor',
    
    # Concrete implementations
    'GAResolutionFetcher',
    'GAResolutionProcessor',
    'SCResolutionFetcher', 
    'SCResolutionProcessor',
    'ThesaurusFetcher',
    'ThesaurusProcessor'
]