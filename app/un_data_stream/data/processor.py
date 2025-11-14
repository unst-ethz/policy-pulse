"""
Data processing orchestrator.

This module orchestrates processing of individual datasets using
registered processors for different dataset types.
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


from ..processors.ga_processor import GAResolutionProcessor
from ..processors.sc_processor import SCResolutionProcessor
from ..processors.thesaurus_processor import ThesaurusProcessor
from ..core.abstractions import DatasetProcessor


class DataProcessor:
    """Orchestrates processing of individual datasets."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Registry of dataset processors
        self._dataset_processors: Dict[str, DatasetProcessor] = {}
        self._register_default_processors()
        
        # Thesaurus processor (separate from datasets)
        self.thesaurus_processor = ThesaurusProcessor(logger)
    
    def _register_default_processors(self):
        """Register default dataset processors."""
        ga_processor = GAResolutionProcessor(self.logger)
        self._dataset_processors[ga_processor.get_dataset_type()] = ga_processor
        
        # Future processors can be added here:
        sc_processor = SCResolutionProcessor(self.logger)
        self._dataset_processors[sc_processor.get_dataset_type()] = sc_processor
    
    def register_processor(self, processor: DatasetProcessor):
        """Register a new dataset processor."""
        self._dataset_processors[processor.get_dataset_type()] = processor
    
    def process_resolutions(self, raw_datasets: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, pd.DataFrame]:
        """Process individual resolution datasets."""
        processed_datasets = {}
        
        for dataset_type, raw_data in raw_datasets.items():
            if dataset_type in self._dataset_processors:
                processor = self._dataset_processors[dataset_type]
                processed_data = processor.process(raw_data, **kwargs)
                processed_datasets.update(processed_data)
            else:
                self.logger.warning(f"No processor registered for dataset type: {dataset_type}")
        
        return processed_datasets
    
    def process_thesaurus(self, thesaurus_graph) -> Dict[str, pd.DataFrame]:
        """Process thesaurus data."""
        return self.thesaurus_processor.process(thesaurus_graph)
    
    def normalize_resolutions(self, resolutions_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize the resolutions dataframe into separate tables.
        
        Args:
            resolutions_df : pd.DataFrame
                DataFrame with one row per resolution-subject pair, containing
                resolution metadata and subject_id column
            
        Returns:
            tuple of (resolutions_normalized_df, resolution_subjects_df)
                - resolutions_normalized_df: One row per resolution with metadata
                - resolution_subjects_df: Resolution-subject pairs mapping table
        """
        
        # Identify columns that belong to resolution metadata vs subject mapping
        subject_columns = ['subjects', 'subject_id']
        resolution_columns = [col for col in resolutions_df.columns if col not in subject_columns]
        
        # 1. Create normalized resolutions table (one row per resolution)
        resolutions_normalized_df = resolutions_df[resolution_columns].drop_duplicates()
        
        # 2. Create resolution-subject mapping table
        # Only keep rows with valid subject_ids
        valid_mappings = resolutions_df[resolutions_df['subject_id'].notna()]
        resolution_subjects_df = valid_mappings[['undl_id', 'subject_id']].copy()
        
        # Remove duplicates (in case same subject appears multiple times for a resolution)
        resolution_subjects_df = resolution_subjects_df.drop_duplicates()
        
        # Check for resolutions without subjects
        resolutions_without_subjects = set(resolutions_normalized_df['undl_id']) - set(resolution_subjects_df['undl_id'])
        if resolutions_without_subjects:
            self.logger.info(f"\nWarning: {len(resolutions_without_subjects)} resolutions have no mapped subjects")
        
        return resolutions_normalized_df, resolution_subjects_df
    
    def calculate_agreement_matrix(self, resolutions_df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], List[str]]:
      """
      Calculate agreement matrices for all resolutions.
      
      Args:
          resolutions_df : pd.DataFrame
              DataFrame with one row per resolution, containing voting columns
              for each member state and metadata columns
      
      Returns:
          Dict[str, np.ndarray]
              Dictionary mapping undl_id to 2D agreement matrix where
              matrix[i,j] represents agreement score between country i and country j
      """
      self.logger.info("Starting agreement matrix calculation")
      start_time = time.time()
      
      # Step 1: Identify country columns (exclude metadata)
      metadata_columns = {
          'undl_id', 'date', 'session', 'resolution', 'draft', 
          'committee_report', 'meeting', 'title', 'agenda_title', 
          'subjects', 'total_yes', 'total_no', 'total_abstentions', 
          'total_non_voting', 'total_ms', 'undl_link', 'subject_id',
          'description', 'agenda', 'modality', 'source_dataset'
      }
      
      country_columns = [col for col in resolutions_df.columns 
                        if col not in metadata_columns]
      
      self.logger.info(f"Found {len(country_columns)} country columns")
      self.logger.info(f"Processing {len(resolutions_df)} resolutions")
      
      # Step 2: Calculate agreement matrix for each resolution
      agreement_matrices = {}
      
      for idx, row in resolutions_df.iterrows():
          undl_id = row['undl_id']
          agreement_matrix = self._calculate_single_resolution_matrix(row, country_columns)
          agreement_matrices[undl_id] = agreement_matrix
          
      
      elapsed_time = time.time() - start_time
      self.logger.info(f"Calculated {len(agreement_matrices)} agreement matrices in {elapsed_time:.2f}s")
      
      return agreement_matrices, country_columns
    
    def _calculate_single_resolution_matrix(self, resolution_row: pd.Series, 
                                        country_columns: List[str]) -> np.ndarray:
      """
      Calculate agreement matrix for a single resolution.
      
      Args:
          resolution_row: Series containing votes for all countries
          country_columns: List of country column names
      
      Returns:
          np.ndarray: 2D agreement matrix (n_countries x n_countries)
      """
      n_countries = len(country_columns)
      agreement_matrix = np.full((n_countries, n_countries), np.nan)
      
      # Vote mapping: Y=1, A=0, N=-1, others=NaN
      vote_mapping = {"Y": 1, "A": 0, "N": -1}
      
      # Convert votes to numeric values
      votes = np.array([vote_mapping.get(resolution_row[country], np.nan) 
                      for country in country_columns])
      
      # Calculate pairwise agreements
      for i in range(n_countries):
          for j in range(i, n_countries):
            if i == j:
              agreement_matrix[i, j] = 1.0
            elif not (np.isnan(votes[i]) or np.isnan(votes[j])):
              diff = abs(votes[i] - votes[j])
              score = 1.0 - (diff / 2.0)
              agreement_matrix[i, j] = score
              agreement_matrix[j, i] = score  # Ensure symmetry
            # else: leave as NaN for missing votes

      return agreement_matrix
