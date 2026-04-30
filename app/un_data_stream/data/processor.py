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

from app.un_data_stream.data.progress import progressbar


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
    
    def calculate_agreement_data(
            self,
            resolutions_df: pd.DataFrame
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], List[str]]:
      """
      Calculate agreement matrices and consensus scores for all resolutions.
      
      Args:
          resolutions_df : pd.DataFrame
              DataFrame with one row per resolution, containing voting columns
              for each member state and metadata columns
      
      Returns:
          Tuple[Dict[str, np.ndarray], Dict[str, float], List[str]]:
              - agreement_matrices: Dict mapping undl_id to its agreement matrix.
              - consensus_scores: Dict mapping undl_id to its score.
              - country_columns: List of country columns used in the matrices.
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
      
      # Step 2: Calculate agreement matrix and consensus score for each resolution
      agreement_matrices = {}
      consensus_scores = {}

      for idx, row in progressbar(resolutions_df.iterrows(), total=len(resolutions_df)):
          undl_id = row['undl_id']
          agreement_matrix = self._calculate_single_resolution_matrix(row, country_columns)
          c_score = self._calculate_single_consensus_score(agreement_matrix)
          agreement_matrices[undl_id] = agreement_matrix
          consensus_scores[undl_id] = c_score


      elapsed_time = time.time() - start_time
      n_res = len(agreement_matrices)
      assert n_res == len(resolutions_df)
      self.logger.info(
          f"Calculated {n_res} agreement matrices and consensus scores in {elapsed_time:.2f}s"
      )
      
      return agreement_matrices, consensus_scores, country_columns

    @staticmethod
    def _calculate_single_resolution_matrix(
            resolution_row: pd.Series,
            country_columns: List[str]
    ) -> np.ndarray:
      """
      Quickly calculate the agreement matrix for a single resolution.

      Uses vectorized calculation via NumPy broadcasting.
      
      Args:
          resolution_row: Series containing votes for all countries
          country_columns: List of country column names
      
      Returns:
          np.ndarray: 2D agreement matrix (n_countries x n_countries)
      """
      # 1. Map votes to numeric values
      vote_mapping = {"Y": 1.0, "A": 0.0, "N": -1.0}

      # Extract the country votes as a NumPy array (floats to accommodate NaN)
      # Using .get() for safety, defaulting to np.nan
      votes = np.array([vote_mapping.get(resolution_row[c], np.nan) for c in country_columns])

      # 2. Use broadcasting to compute all pairwise absolute differences
      # votes[:, np.newaxis] creates a column vector (N, 1)
      # votes[np.newaxis, :] creates a row vector (1, N)
      # The subtraction results in an (N, N) matrix of all combinations
      abs_diff_mat = np.abs(votes[:, np.newaxis] - votes[np.newaxis, :])

      # 3. Apply the agreement-score formula to the entire matrix at once
      agreement_matrix = 1.0 - (abs_diff_mat / 2.0)

      return agreement_matrix

    @staticmethod
    def _calculate_single_consensus_score(agreement_matrix: np.ndarray) -> float:
        """
        Calculate the 'consensus' score for a given resolution, based on
        the resolution's agreement matrix.

        The consensus score is simply the average vote-agreement score
        across all country pairs for which both members voted on the resolution
        at hand.

        Args:
            agreement_matrix: np.ndarray: 2D agreement matrix (n_countries x n_countries)

        Returns:
            float
        """

        # We take the mean of the lower triangle of the matrix (excluding the diagonal)
        # to get the average of all unique pairwise scores, ignoring NaNs.
        n_countries = agreement_matrix.shape[0]
        tril_indices = np.tril_indices(n_countries, k=-1)  # k=-1 to exclude diagonal
        lower_triangle_values = agreement_matrix[tril_indices]

        with np.errstate(invalid='ignore'):  # Suppress warning for all-NaN slice
            consensus_score = np.nanmean(lower_triangle_values)

        return float(consensus_score)
