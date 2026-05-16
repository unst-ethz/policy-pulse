"""
Data repository for managing processed UN data.

This module handles storage, retrieval, and caching of processed UN data,
orchestrating the entire data processing pipeline.
"""

import bz2
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from .fetcher import DataFetcher
from .merger import DataMerger
from .processor import DataProcessor


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

        if not self._has_cached_data() or not self._check_data_version():
            self.logger.info("Initializing from raw data sources.")
            # Check if configured data sources are valid
            if not self._resolve_and_validate_data_urls():
                self.logger.error(
                    "One or more configured data sources are invalid. "
                    "Please review settings in data_sources.yaml."
                )
                raise ValueError("Failed to resolve one or more data sources. Check logs for details.")

            self._build_data()
        else:
            # Cached data found and version matches -> load
            self.logger.info("Initializing from cached data.")
            self._load_cached_data()

        self.logger.info("Initialization complete. Below are memory footprints:")
        self.logger.info(
            f"Resolution Table: {self.resolution_table.memory_usage(index=True).sum() / (1024**2):.2f} MB"
        )
        self.logger.info(
            f"Resolution Subject Table: {self.resolution_subject_table.memory_usage(index=True).sum() / (1024**2):.2f} MB"
        )
        self.logger.info(
            f"Subject Table: {self.subject_table.memory_usage(index=True).sum() / (1024**2):.2f} MB"
        )
        self.logger.info(
            f"Closure Table: {self.closure_table.memory_usage(index=True).sum() / (1024**2):.2f} MB"
        )
        self.logger.info(
            f"Broader Table: {self.broader_table.memory_usage(index=True).sum() / (1024**2):.2f} MB"
        )
        self.logger.info(
            f"Agreement Matrices: {sum(map(lambda x: x[1].nbytes / (1024**2), self.agreement_matrices.items()))} MB"
        )

    def get_data(self) -> Dict[str, Any]:
        """Return processed data as a dictionary of DataFrames."""
        return {
            'resolution': self.resolution_table,
            'resolution_subject': self.resolution_subject_table,
            'subject': self.subject_table,
            'closure': self.closure_table,
            'broader': self.broader_table,
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

    def _resolve_and_validate_data_urls(self) -> bool:
        """
        Check if the data sources in the configuration are valid and reachable.
        Resolves dynamic URLs via API to ensure files exist.
        """
        
        # We use a temporary DataFetcher to resolve URLs using the registered logic
        # This can handle both static and dynamic sources.
        fetcher_orchestrator = DataFetcher(self.config, self.logger)
        
        all_valid = True
        
        # 1. Check Resolutions
        resolutions_config = self.config['data_sources'].get('resolutions', {})
        for dataset_type, source_config in resolutions_config.items():
            try:
                # Get the specific fetcher for this dataset type
                if dataset_type in fetcher_orchestrator._dataset_fetchers:
                    dataset_fetcher = fetcher_orchestrator._dataset_fetchers[dataset_type]
                    # Attempt to resolve the URL (this triggers the UNDL's API check for dynamic sources)
                    url = dataset_fetcher.resolve_url(source_config)
                    self.logger.info(f"Successfully resolved URL for {dataset_type}: {url}")
                else:
                    self.logger.warning(f"No fetcher registered for {dataset_type}, skipping URL check.")
            except Exception as e:
                self.logger.error(f"Failed to resolve/check URL for {dataset_type}: {e}")
                all_valid = False

        # 2. Check Thesaurus
        thesaurus_config = self.config['data_sources'].get('thesaurus')
        if thesaurus_config:
            try:
                # Thesaurus fetcher is separate in DataFetcher
                url = fetcher_orchestrator.thesaurus_fetcher.resolve_url(thesaurus_config)
                self.logger.info(f"Successfully resolved URL for thesaurus: {url}")
            except Exception as e:
                self.logger.error(f"Failed to resolve/check URL for thesaurus: {e}")
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
            'broader_table.csv',
            'agreement_matrices.pkl'
        ]
        all_exist = all((data_path / file).exists() for file in required_files)
        if all_exist:
            self.logger.info("Cached Data files found.")
        else:
            self.logger.info("Cached Data files not found.")
        return all_exist

    def _check_data_version(self) -> bool:
        """
        Check if the cached data version matches the config version.
        
        Returns:
             bool: True if versions match, False otherwise.
        """
        data_path = Path(self.config['paths']['data'])
        metadata_file = data_path / 'metadata.json'
        
        if not metadata_file.exists():
            self.logger.warning("No metadata file found for cached data.")
            return False
            
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            cached_version = metadata.get('version')
            config_version = str(self.config.get('version'))
            
            if str(cached_version) == config_version:
                self.logger.info(f"Data version match: {cached_version}")
                return True
            else:
                self.logger.warning(f"Data version mismatch. Cached: {cached_version}, Config: {config_version}")
                return False
        except Exception as e:
             self.logger.error(f"Error checking data version: {e}")
             return False
    
    def _load_cached_data(self):
        """Load cached data files into DataFrames."""
        data_path = Path(self.config['paths']['data'])
        data_path.mkdir(exist_ok=True)

        # Load CSV files
        self.logger.info("Loading resolution table")
        self.resolution_table = pd.read_csv(data_path / 'resolution_table.csv')
        self.logger.info("Loading resolution subject table")
        self.resolution_subject_table = pd.read_csv(
            data_path / "resolution_subject_table.csv"
        )
        self.logger.info("Loading subject table")
        self.subject_table = pd.read_csv(data_path / "subject_table.csv")
        self.logger.info("Loading closure table")
        self.closure_table = pd.read_csv(data_path / "closure_table.csv")
        self.logger.info("Loading broader table")
        self.broader_table = pd.read_csv(data_path / "broader_table.csv")

        self.logger.info("Loading agreement matrices")
        # Load pickle file containing agreement matrices and labels
        pkl_path = data_path / 'agreement_matrices.pkl'
        try:
            # Try loading with compression first
            with bz2.BZ2File(pkl_path, 'rb') as f:
                agreement_data = pickle.load(f)
        except (EOFError, OSError):
            with open(pkl_path, 'rb') as f:
                agreement_data = pickle.load(f)

            self.logger.warning("Cache is in old format. Rewriting...")
            # If it uses float64, convert to float32
            for key in agreement_data["agreement_matrices"]:
                agreement_data["agreement_matrices"][key] = agreement_data[
                    "agreement_matrices"
                ][key].astype("float32")

            # Write back new cache in compressed format
            with bz2.BZ2File(pkl_path, "wb") as f:
                pickle.dump(agreement_data, f)

        self.agreement_matrices = agreement_data["agreement_matrices"]
        self.country_columns = agreement_data["country_columns"]

        self.logger.info("Cached data loaded successfully.")
    
    def _save_cached_data(self):
        """Save data files into DataFrames."""
        data_path = Path(self.config['paths']['data'])
        data_path.mkdir(exist_ok=True)

        # Save the tables in the defined folder
        self.resolution_table.to_csv(data_path / 'resolution_table.csv', index=False)
        self.resolution_subject_table.to_csv(data_path / 'resolution_subject_table.csv', index=False)
        self.subject_table.to_csv(data_path / 'subject_table.csv', index=False)
        self.closure_table.to_csv(data_path / 'closure_table.csv', index=False)
        self.broader_table.to_csv(data_path / 'broader_table.csv', index=False)

        pkl_path = data_path / 'agreement_matrices.pkl'
        # Save agreement matrices with bz2 compression to reduce disk space
        with bz2.BZ2File(pkl_path, 'wb') as f:
            agreement_data = {
                'agreement_matrices': self.agreement_matrices,
                'country_columns': self.country_columns
            }
            pickle.dump(agreement_data, f)
            
        # Save Metadata (Version)
        metadata = {
            'version': self.config.get('version', 'unknown'),
            'last_updated': self.config.get('last_updated', 'unknown')
        }
        with open(data_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        self.logger.info(f"Saved data metadata (version {metadata['version']})")

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
        self.broader_table = thesaurus_tables.get('broader_table', pd.DataFrame())
        
        # Process individual resolution datasets
        processed_datasets = processor.process_resolutions(resolutions_raw, subject_table=self.subject_table)
        
        # Continue with GA for now
        # unified_resolutions = merger.merge_resolutions(processed_datasets)
        ga_resolutions = processed_datasets.get('ga_resolutions', pd.DataFrame())

        # Normalize ga_resolutions
        self.resolution_table, self.resolution_subject_table = processor.normalize_resolutions(ga_resolutions)

        # Calculate the agreement matrix
        self.agreement_matrices, self.country_columns = processor.calculate_agreement_matrix(self.resolution_table)

        
        expanded_subjects = set()
        
        for subject_id in self.resolution_subject_table["subject_id"].unique().tolist():
            ancestors = self.closure_table[
                self.closure_table['descendant_id'] == subject_id
                ]['ancestor_id'].to_list()
            for i in ancestors:
                expanded_subjects.add(i)

        expanded_subjects = list(expanded_subjects)
        self.subject_table = self.subject_table[self.subject_table['subject_id'].isin(expanded_subjects)]
        self.closure_table = self.closure_table[self.closure_table['ancestor_id'].isin(expanded_subjects)]
        self.broader_table = self.broader_table[self.broader_table['parent_id'].isin(expanded_subjects)]


        # Save processed data
        self._save_cached_data()