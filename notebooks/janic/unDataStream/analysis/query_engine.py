"""
Query engine for resolution data analysis.

This module provides advanced querying capabilities for UN resolution data,
including subject-based filtering, trend analysis, and temporal queries.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path

from ..data import DataRepository


class ResolutionQueryEngine:
    """Advanced query engine for resolution data analysis."""
    
    def __init__(self, repo: DataRepository):
        """
        Initialize query engine with processed data.
        
        Args:
            data: Dictionary containing processed DataFrames
        """
        data = repo.get_data()
        self.logger = repo.logger

        self.resolution_table = data.get('resolution', pd.DataFrame())
        self.resolution_subject_table = data.get('resolution_subject', pd.DataFrame())
        self.subject_table = data.get('subject', pd.DataFrame())
        self.closure_table = data.get('closure', pd.DataFrame())
        self.agreement_matrices = data.get('agreement_matrices', pd.DataFrame())
        self.country_columns = data.get('country_columns', [])

    def query_resolutions(self, start_date: Optional[str] = None, end_date: Optional[str] = None, subject_ids: Optional[List[str]] = None, language: str = "en", include_descendants: bool = True) -> pd.DataFrame:
        """
        Query resolutions based on date range and subject filters.
        
        Args:
            start_date: Start date in format 'YYYY-MM-DD' (None = from beginning)
            end_date: End date in format 'YYYY-MM-DD' (None = until today)
            subject_ids: List of subject URIs to filter by (None = all subjects)
            include_descendants: If True, include all descendants of specified subjects
        
        Returns:
            pd.DataFrame: Filtered resolutions with all metadata
        """

        # Start with all resolutions
        filtered_df = self.resolution_table.copy()

        filtered_df['date'] = pd.to_datetime(filtered_df['date'])

        # 1. Apply date filters
        if start_date:
            filtered_df = filtered_df[filtered_df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            filtered_df = filtered_df[filtered_df['date'] <= pd.to_datetime(end_date)]
        
        # 2. Apply subject filters
        if subject_ids is not None and len(subject_ids) > 0:
            if include_descendants:
                # Expand subject_ids to include all descendants
                expanded_subjects = set(subject_ids)
                
                for subject_id in subject_ids:
                    # Find all descendants of this subject
                    descendants = self.closure_table[
                        self.closure_table['ancestor_id'] == subject_id
                    ]['descendant_id'].unique()
                    expanded_subjects.update(descendants)
                
                self.logger.info(f"Expanded {len(subject_ids)} subjects to {len(expanded_subjects)} (including descendants)")
                subject_filter = list(expanded_subjects)
            else:
                subject_filter = subject_ids
            
            # Find resolutions with these subjects
            matching_resolution_ids = self.resolution_subject_table[
                self.resolution_subject_table['subject_id'].isin(subject_filter)
            ]['undl_id'].unique()
            
            # Filter resolutions
            filtered_df = filtered_df[
                filtered_df['undl_id'].isin(matching_resolution_ids)
            ]
            self.logger.info(f"After subject filter: {len(filtered_df)} resolutions")
        
        self.logger.info(f"\nFinal result: {len(filtered_df)} resolutions")
        return filtered_df

    def query_agreement_matrix(self, resolution_ids: Optional[List[str]]) -> Dict[str, Any]:
        """
        Retrieve agreement matrices for specified resolutions.
        
        Args:
            resolution_ids: List of undl_id strings for which to retrieve matrices
        Returns:
            Dict[str, Any]: Mapping of undl_id to its agreement matrix (as numpy array)
        """
        if resolution_ids is None or len(resolution_ids) == 0:
            # Return all matrices
            return self.agreement_matrices
        
        matrices = {}
        for res_id in resolution_ids:
            matrix = self.agreement_matrices.get(res_id)
            if matrix is not None:
                matrices[res_id] = matrix
            else:
                self.logger.warning(f"No agreement matrix found for resolution ID: {res_id}")
        return matrices
    
    def query_agreement_between_countries(self, country_code: str, resolution_ids: Optional[List[str]] = None, average: bool = False) -> pd.DataFrame:
        """
        Get agreement scores between a selected country and all other countries.
        
        Args:
            country_code: Country code (column name) to analyze
            resolution_ids: List of resolution IDs to analyze (None = all resolutions)
            average: If True, return averaged scores across all resolutions;
                    if False, return scores for each resolution separately
        
        Returns:
            pd.DataFrame: Agreement scores with columns:
                - If average=False: ['undl_id', 'target_country', 'agreement_score']
                - If average=True: ['target_country', 'average_agreement_score']
        """
        if not self.agreement_matrices:
            self.logger.error("No agreement matrices available")
            return pd.DataFrame()
        
        if country_code not in self.country_columns:
            self.logger.error(f"Country '{country_code}' not found in country columns")
            available_countries = self.country_columns[:10]  # Show first 10
            self.logger.info(f"Available countries (first 10): {available_countries}")
            return pd.DataFrame()
        
        # Get country index
        country_index = self.country_columns.index(country_code)
        
        # Determine which resolutions to analyze
        if resolution_ids is None:
            target_resolutions = list(self.agreement_matrices.keys())
        else:
            # Filter to only existing resolution IDs
            target_resolutions = [rid for rid in resolution_ids 
                                if rid in self.agreement_matrices]
            
            missing_ids = set(resolution_ids or []) - set(target_resolutions)
            if missing_ids:
                self.logger.warning(f"Missing agreement matrices for {len(missing_ids)} resolutions")
        
        if not target_resolutions:
            self.logger.warning("No valid resolutions found for agreement analysis")
            return pd.DataFrame()
        
        self.logger.info(f"Analyzing agreement for '{country_code}' across {len(target_resolutions)} resolutions")
        
        # Create list to store all resolution data
        all_resolution_data = []
        
        for resolution_id in target_resolutions:
            matrix = self.agreement_matrices[resolution_id]
            
            # Extract row for target country (agreement with all others)
            country_agreements = matrix[country_index, :]
            
            # Create record for this resolution
            resolution_data = {'undl_id': resolution_id}
            
            # Add agreement score with each other country
            for other_idx, other_country in enumerate(self.country_columns):
                if other_idx != country_index:  # Skip self-agreement
                    agreement_score = country_agreements[other_idx]
                    resolution_data[other_country] = agreement_score if not np.isnan(agreement_score) else np.nan
            
            all_resolution_data.append(resolution_data)
        
        if not all_resolution_data:
            self.logger.warning("No valid agreement scores found")
            return pd.DataFrame()
        
        # Convert to DataFrame
        scores_df = pd.DataFrame(all_resolution_data)
        
        if average:
            # Calculate average agreement per country (excluding undl_id column)
            country_cols = [col for col in scores_df.columns if col != 'undl_id']
            avg_scores = scores_df[country_cols].mean(skipna=True)
            
            # Create single-row DataFrame with averages
            avg_df = pd.DataFrame([avg_scores.values], columns=avg_scores.index)
            
            # Add metadata columns
            avg_df.insert(0, 'resolution_count', len(target_resolutions))
            avg_df.insert(0, 'source_country', country_code)
            
            self.logger.info(f"Calculated average agreements across {len(target_resolutions)} resolutions")
            return avg_df
        else:
            # Sort by resolution ID
            scores_df = scores_df.sort_values('undl_id')
            
            self.logger.info(f"Retrieved agreement scores for {len(scores_df)} resolutions")
            return scores_df