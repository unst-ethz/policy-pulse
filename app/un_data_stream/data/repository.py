"""
Data repository for managing processed UN data.

This module handles storage, retrieval, and caching of processed UN data,
orchestrating the entire data processing pipeline.
"""

import logging
import pickle
import sys
import yaml
import requests
import pandas as pd
from pathlib import Path
from typing import Any, Dict

from .fetcher import DataFetcher
from .processor import DataProcessor
from .merger import DataMerger


class DataRepository:
    """Handles storage and retrieval of processed UN data."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        
        # Initialize data attributes
        self.resolution_table: pd.DataFrame
        self.resolution_subject_table: pd.DataFrame
        self.subject_table: pd.DataFrame
        self.closure_table: pd.DataFrame

        # Load configuration
        self._load_config()

        # Initialize Logging
        self._setup_logging()

        self.logger.info("Initializing UNDataRepository")

        # Check if links are still valid
        if not self._check_URLS():
            self.logger.error("One or more data URLs are invalid. The dataset might have been updated. Check the date in the URL.")
            raise ValueError("Invalid data URLs in configuration.")
        
        # Check if data is already processed and available
        if self._has_cached_data():
            # Cached data found -> load
            self._load_cached_data() 
            self.logger.info("Initialization Complete with Cached Data.")
            return
        
        self._build_data()
        self.logger.info("Initialization complete with fetched Data.")

    def get_data(self) -> Dict[str, Any]:
        """Return processed data as a dictionary of DataFrames."""
        return {
            'resolution': self.resolution_table,
            'resolution_subject': self.resolution_subject_table,
            'subject': self.subject_table,
            'closure': self.closure_table,
            'agreement_matrices': self.agreement_matrices,
            'country_columns': self.country_columns
        }
    
    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def _setup_logging(self):
        """Setup logging configuration with file and console handlers."""
        # Create logger
        self.logger = logging.getLogger('UNResolutionAnalyzer')
        
        if not self.config['logs']:
            self.logger.disabled = True
            return

        self.logger.setLevel(logging.DEBUG if self.config['debug'] else logging.INFO)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )

        log_dir = Path(self.config['paths']['logs'])
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / 'un_resolution_analyzer.log'

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        if self.config['debug']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(simple_formatter)
            self.logger.addHandler(console_handler)

        self.logger.info("Logging setup complete.")

    def _check_URLS(self) -> bool:
        """Check if the URLs in the configuration are reachable."""
        urls = [
            source['url'] 
            for source in self.config['data_sources'].values() 
            if 'url' in source
        ]
        all_valid = True
        for url in urls:
            try:
                response = requests.head(url, allow_redirects=True, timeout=10)
                if response.status_code != 200:
                    self.logger.error(f"URL not reachable: {url} (Status code: {response.status_code})")
                    all_valid = False
                else:
                    self.logger.info(f"URL is valid: {url}")
            except requests.RequestException as e:
                self.logger.error(f"Error reaching URL: {url} ({e})")
                all_valid = False
        return all_valid
    
    def _has_cached_data(self) -> bool:
        """Check if cached data files exist."""
        data_path = Path(self.config['paths']['data'])
        required_files = [
            'resolution_table.csv',
            'resolution_subject_table.csv',
            'subject_table.csv',
            'closure_table.csv',
            'agreement_matrices.pkl'
        ]
        all_exist = all((data_path / file).exists() for file in required_files)
        if all_exist:
            self.logger.info("Cached Data files found.")
        else:
            self.logger.info("Cached Data files not found.")
        return all_exist
    
    def _load_cached_data(self):
        """Load cached data files into DataFrames."""
        data_path = Path(self.config['paths']['data'])
        data_path.mkdir(exist_ok=True)

        # Load CSV files
        self.resolution_table = pd.read_csv(data_path / 'resolution_table.csv')
        self.resolution_subject_table = pd.read_csv(data_path / 'resolution_subject_table.csv')
        self.subject_table = pd.read_csv(data_path / 'subject_table.csv')
        self.closure_table = pd.read_csv(data_path / 'closure_table.csv')
        self.logger.info("Cached data loaded successfully.")

        # Load agreement matrices
        with open(data_path / 'agreement_matrices.pkl', 'rb') as f:
            agreement_data = pickle.load(f)

        self.agreement_matrices = agreement_data['agreement_matrices']
        self.country_columns = agreement_data['country_columns']
    
    def _save_cached_data(self):
        """Save data files into DataFrames."""
        data_path = Path(self.config['paths']['data'])
        data_path.mkdir(exist_ok=True)

        # Save the tables in the defined folder
        self.resolution_table.to_csv(data_path / 'resolution_table.csv', index=False)
        self.resolution_subject_table.to_csv(data_path / 'resolution_subject_table.csv', index=False)
        self.subject_table.to_csv(data_path / 'subject_table.csv', index=False)
        self.closure_table.to_csv(data_path / 'closure_table.csv', index=False)

        with open(data_path / 'agreement_matrices.pkl', 'wb') as f:
            agreement_data = {
                'agreement_matrices': self.agreement_matrices,
                'country_columns': self.country_columns
            }
            pickle.dump(agreement_data, f)

    def _build_data(self):
        """Build processed data tables from raw sources."""
        
        # Fetch Data
        fetcher = DataFetcher(self.config, self.logger)
        resolutions_raw = fetcher.fetch_resolutions()
        thesaurus_graph = fetcher.fetch_thesaurus()

        # Process data
        processor = DataProcessor(self.config, self.logger)
        merger = DataMerger(self.logger)
        
        # Process thesaurus first (needed for subject matching)
        thesaurus_tables = processor.process_thesaurus(thesaurus_graph)
        self.subject_table = thesaurus_tables.get('subject_table', pd.DataFrame())
        self.closure_table = thesaurus_tables.get('closure_table', pd.DataFrame())
        
        # Process individual resolution datasets
        processed_datasets = processor.process_resolutions(resolutions_raw, subject_table=self.subject_table)
        
        # Continue with GA for now
        # unified_resolutions = merger.merge_resolutions(processed_datasets)
        ga_resolutions = processed_datasets.get('ga_resolutions', pd.DataFrame())

        # Normalize ga_resolutions
        self.resolution_table, self.resolution_subject_table = processor.normalize_resolutions(ga_resolutions)

        # Calculate the agreement matrix
        self.agreement_matrices, self.country_columns = processor.calculate_agreement_matrix(self.resolution_table)

        # Save processed data
        self._save_cached_data()