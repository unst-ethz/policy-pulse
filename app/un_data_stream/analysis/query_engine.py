"""
Query engine for resolution data analysis.

This module provides querying capabilities for UN resolution data,
including subject-based and date filtering, bilateral vote agreement,
lookups and multilateral alignment statistics.
"""

from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from ..data import DataRepository


class ResolutionQueryEngine:
    """Advanced query engine for resolution data analysis."""
    
    def __init__(self, repo: DataRepository):
        """
        Initialize query engine with processed data.

        Args:
            repo: DataRepository instance providing all precomputed tables and matrices.
        """
        data = repo.get_data()
        self.logger = repo.logger

        self.resolution_table = data.get('resolution', pd.DataFrame())
        self.resolution_subject_table = data.get('resolution_subject', pd.DataFrame())
        self.subject_table = data.get('subject', pd.DataFrame())
        self.closure_table = data.get('closure', pd.DataFrame())
        self.agreement_matrices = data.get('agreement_matrices', {})
        self.country_columns = data.get('country_columns', [])

        self._multilateral_scores = data.get('multilateral_scores')

        # Precompute voted/abstained bool arrays (R x C) from resolution_table once at startup.
        # multilateral_scores rows, _voted rows and _abstained rows all share resolution_table's
        # row order (calculate_agreement_data iterates resolutions_df in order), so one index
        # dict covers all three arrays.
        if not self.resolution_table.empty and self.country_columns:
            votes = (
                self.resolution_table[self.country_columns]
                .astype(str)
                .apply(lambda s: s.str.strip().str.upper())
                .replace({"NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA, "": pd.NA})
            )

            self._yes = (votes == "Y").to_numpy()
            self._no = (votes == "N").to_numpy()
            self._abstained = (votes == "A").to_numpy()
            self._voted = self._yes | self._no | self._abstained

            self._row_index: dict[str, int] = {
                rid: i for i, rid in enumerate(self.resolution_table["undl_id"].tolist())
            }
        else:
            self._yes = np.empty((0, 0), dtype=bool)
            self._no = np.empty((0, 0), dtype=bool)
            self._abstained = np.empty((0, 0), dtype=bool)
            self._voted = np.empty((0, 0), dtype=bool)
            self._row_index = {}

    def query_resolutions(
            self,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            subject_ids: Optional[List[str]] = None,
            language: str = "en",
            include_descendants: bool = True
    ) -> pd.DataFrame:
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
            # Separate the synthetic "no subject" sentinel from real subject IDs
            include_no_subject = "__no_subject__" in subject_ids
            real_subject_ids = [s for s in subject_ids if s != "__no_subject__" and s != "__all_subjects__"]

            matching_ids = set()

            if include_no_subject:
                # Resolutions that have no entry in the subject table at all
                all_with_subject = set(self.resolution_subject_table["undl_id"].unique())
                no_subject_ids = set(filtered_df["undl_id"].unique()) - all_with_subject
                matching_ids.update(no_subject_ids)
                self.logger.info(f"No-subject resolutions: {len(no_subject_ids)}")

            if real_subject_ids:
                if include_descendants:
                    expanded_subjects = set(real_subject_ids)
                    for subject_id in real_subject_ids:
                        descendants = self.closure_table[
                            self.closure_table['ancestor_id'] == subject_id
                        ]['descendant_id'].unique()
                        expanded_subjects.update(descendants)
                    self.logger.info(f"Expanded {len(real_subject_ids)} subjects to {len(expanded_subjects)} (including descendants)")
                    subject_filter = list(expanded_subjects)
                else:
                    subject_filter = real_subject_ids

                subject_resolution_ids = self.resolution_subject_table[
                    self.resolution_subject_table['subject_id'].isin(subject_filter)
                ]['undl_id'].unique()
                matching_ids.update(subject_resolution_ids)

            filtered_df = filtered_df[filtered_df['undl_id'].isin(matching_ids)]
            self.logger.info(f"After subject filter: {len(filtered_df)} resolutions")
        
        self.logger.info(f"Final result: {len(filtered_df)} resolutions")
        return filtered_df

    def query_agreement_matrix(self, resolution_ids: Optional[List[str]]) -> Dict[str, np.ndarray]:
        """
        Retrieve bilateral vote-agreement matrices for specified resolutions.

        Args:
            resolution_ids: List of undl_id strings to retrieve. If None or empty,
                returns the full dict of all bilateral matrices.
        Returns:
            Dict[str, np.ndarray]: Mapping of undl_id to its (C x C) bilateral matrix.
        """
        if resolution_ids is None or len(resolution_ids) == 0:
            return self.agreement_matrices

        matrices = {}
        for res_id in resolution_ids:
            matrix = self.agreement_matrices.get(res_id)
            if matrix is not None:
                matrices[res_id] = matrix
            else:
                self.logger.warning(f"No bilateral matrix found for resolution ID: {res_id}")
        return matrices
    
    def query_agreement_between_countries(
            self,
            country_code: str,
            resolution_ids: Optional[List[str]] = None,
            average: bool = False
    ) -> pd.DataFrame:
        """
        Get bilateral agreement scores between a selected country and all other countries.
        
        Args:
            country_code: Country code (column name) to analyze
            resolution_ids: List of resolution IDs to analyze (None = all resolutions)
            average: If True, return averaged scores across all resolutions;
                    if False, return scores for each resolution separately
        
        Returns:
            pd.DataFrame in wide format — one column per country (ISO3 code), excluding
            country_code itself:
                - If average=False: one row per resolution; columns ['undl_id', <iso3>, ...]
                  where each cell is the bilateral agreement score between country_code and
                  that country on that resolution.
                - If average=True: a single row; columns ['source_country',
                  'resolution_count', <iso3>, ...] where each cell is the mean bilateral
                  agreement score across all selected resolutions.
        """
        if not self.agreement_matrices:
            self.logger.error("No bilateral matrices available")
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
                self.logger.warning(f"Missing bilateral matrices for {len(missing_ids)} resolutions")
        
        if not target_resolutions:
            self.logger.warning("No valid resolutions found for agreement analysis")
            return pd.DataFrame()
        
        self.logger.info(f"Analyzing agreement for '{country_code}' across {len(target_resolutions)} resolutions")
        
        # Create list to store all resolution data
        all_resolution_data = []
        
        for resolution_id in target_resolutions:
            matrix = self.agreement_matrices[resolution_id]  # (C, C)

            # Extract row for target country (agreement with all others)
            country_agreements = matrix[country_index, :]    # (C,)

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
        scores_df = pd.DataFrame(all_resolution_data)        # (R', C-1) + undl_id column

        if average:
            # Calculate average agreement per country (excluding undl_id column)
            country_cols = [col for col in scores_df.columns if col != 'undl_id']
            avg_scores = scores_df[country_cols].mean(skipna=True)  # (C-1,)
            
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
        
    def query_multilateral_stats(self, resolution_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        For each country, compute the average multilateral alignment and abstention rate
        across the given resolutions, based on the full precomputed (R x C) arrays.

        Args:
            resolution_ids: List of undl_ids to include. If None or empty, uses all
                resolutions in the dataset.

        Returns:
            pd.DataFrame with one row per country and columns:
                - country: ISO3 country code
                - multilateral_alignment: mean pairwise agreement with all other voting
                  countries, averaged across selected resolutions (NaN if no participation)
                - abstention_rate: fraction of votes cast as abstentions (NaN if no votes)
                - participation_count: number of selected resolutions the country voted on
        """
        if self._multilateral_scores is None or not self.country_columns:
            return pd.DataFrame()

        if resolution_ids is None or len(resolution_ids) == 0:
            rows = list(self._row_index.values())
        else:
            rows = [self._row_index[r] for r in resolution_ids if r in self._row_index]

        if not rows:
            return pd.DataFrame()

        align_slice = self._multilateral_scores[rows]              # (R', C) float32
        avg_alignment = np.nanmean(align_slice, axis=0)            # (C,) float64

        voted_slice = self._voted[rows]                            # (R', C) bool
        abstained_slice = self._abstained[rows]                    # (R', C) bool
        participation = voted_slice.sum(axis=0)                    # (C,) int
        abstentions = abstained_slice.sum(axis=0)                  # (C,) int
        abstention_rate = np.where(participation > 0, abstentions / participation, np.nan)

        return pd.DataFrame({
            "country": self.country_columns,
            "multilateral_alignment": avg_alignment,
            "abstention_rate": abstention_rate,
            "participation_count": participation,
        })

    def get_available_countries(self) -> List[str]:
        """Get list of available country codes in the dataset."""
        return self.country_columns